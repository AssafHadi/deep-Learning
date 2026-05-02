from __future__ import annotations

import os
import io
import json
import math
import time
import shutil
import random
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # pragma: no cover
    Image = ImageOps = UnidentifiedImageError = None

from core.state import model_state_context
from models.ann import *
from models.cnn import *
from models.lstm import *
from services.preprocessing import *
from services.training import *
from services.evaluation import *
from services.prediction import *
from visualization.plots import *
from storage.persistence import *

from pages.home import ann_inject_css, ann_hero, cnn_inject_css, cnn_hero, cnn_status_bar, cnn_render_uploaded_images_section, lstm_inject_css, lstm_hero, lstm_info_card, lstm_top_status_bar

PAGE_NAME = "Predict"

def render_ann_predict_ui():
    st.title("Predict")
    model = st.session_state.trained_model
    prepared = st.session_state.prepared_data
    if model is None or prepared is None:
        st.info("Train or load a project first.")
        return

    st.subheader("Manual Single Prediction")
    schema = prepared["feature_schema"]
    manual_values = {}
    cols = st.columns(2)
    feature_cols = prepared["feature_columns_original"]

    for idx, col in enumerate(feature_cols):
        with cols[idx % 2]:
            meta = schema[col]
            if meta["is_numeric"]:
                manual_values[col] = st.text_input(f"{col}", value=str(meta["example"] or "0"))
            else:
                options = meta["categories"] or []
                options = [""] + options
                manual_values[col] = st.selectbox(f"{col}", options=options, index=0)

    if st.button("Run Single Prediction", use_container_width=True):
        row = {}
        for col in feature_cols:
            meta = schema[col]
            val = manual_values[col]
            if meta["is_numeric"]:
                try:
                    row[col] = float(val)
                except Exception:
                    row[col] = np.nan
            else:
                row[col] = val if val != "" else np.nan
        pred_df = pd.DataFrame([row])
        out = ann_predict_with_pipeline(model, prepared, pred_df, threshold=float(st.session_state.config["threshold"]))
        st.dataframe(out, use_container_width=True)

    st.divider()
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Upload prediction file", type=["csv", "xlsx", "xls"], key="pred_file")
    if uploaded is not None:
        try:
            pred_df = ann_read_uploaded_table(uploaded)
            out = ann_predict_with_pipeline(model, prepared, pred_df, threshold=float(st.session_state.config["threshold"]))
            st.dataframe(out.head(100), use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def render_cnn_predict_ui():
    cnn_hero()
    cnn_status_bar()
    st.subheader("Predict")

    if st.session_state.trained_model is None or not st.session_state.training_complete:
        st.warning("No trained model is loaded yet. You can still upload one or many images now, but prediction will start only after training or loading a model.")

    cnn_render_uploaded_images_section(
        uploader_label="Upload one image or many images for prediction",
        uploader_key="predict_page_image_uploader",
        show_title=False,
        show_gradcam=True,
    )


def render_lstm_predict_ui():
    lstm_hero("Predict", "Generate numeric forecasts or class predictions from the latest sequence window.")
    training = st.session_state.get("training")
    processed = st.session_state.get("processed")
    if training is None or processed is None:
        st.info("Train the model first.")
        return

    cfg = st.session_state["config"]
    source = st.radio("Prediction source", ["Use last window from loaded data", "Upload new data for prediction"], horizontal=True)
    pred_df = None

    if source == "Upload new data for prediction":
        new_file = st.file_uploader("Upload new prediction dataset", type=["csv", "xlsx", "xls"], key="pred_uploader")
        if new_file is not None:
            try:
                pred_df = lstm_read_uploaded_data(new_file)
            except Exception as e:
                st.error(f"Prediction file load failed: {e}")
                return
    else:
        raw_df = st.session_state.get("raw_df")
        pred_df = raw_df.copy() if raw_df is not None else None

    if pred_df is None:
        st.info("Provide a source dataset for prediction.")
        return

    button_label = "Run Forecast" if processed.get("task") == "regression" else "Run Classification Prediction"
    if st.button(button_label, type="primary", use_container_width=True):
        try:
            X_input, numeric = lstm_prepare_prediction_input(pred_df, processed, cfg)
            pred_raw = training["model"].predict(X_input, verbose=0)

            if processed.get("task") == "classification":
                class_names = processed["class_names"]
                if len(class_names) == 2:
                    score = float(np.asarray(pred_raw).reshape(-1)[0])
                    probs = np.array([1.0 - score, score])
                    pred_idx = int(score >= 0.5)
                else:
                    probs = np.asarray(pred_raw).reshape(-1)
                    pred_idx = int(np.argmax(probs))
                pred_table = pd.DataFrame({
                    "class": class_names,
                    "probability": probs,
                }).sort_values("probability", ascending=False).reset_index(drop=True)
                st.session_state["prediction"] = {
                    "task": "classification",
                    "predicted_class": class_names[pred_idx],
                    "probability_table": pred_table,
                }
                st.success("Classification prediction complete.")
            else:
                pred_inv = lstm_inverse_3d(pred_raw, processed["target_scaler"], int(cfg["horizon"]), len(processed["target_cols"]))[0]
                pred_table = pd.DataFrame(pred_inv, columns=processed["target_cols"], index=[f"step_{i+1}" for i in range(int(cfg["horizon"]))])

                implied_prices = None
                if cfg["transform_mode"] in {"pct_change", "log_return", "raw"}:
                    last_prices = numeric.iloc[-1]
                    implied_prices = lstm_implied_prices_from_returns(last_prices, pred_inv, cfg["transform_mode"], processed["target_cols"])

                st.session_state["prediction"] = {
                    "task": "regression",
                    "forecast_table": pred_table,
                    "implied_prices": implied_prices,
                }
                st.success("Forecast complete.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    pred_state = st.session_state.get("prediction")
    if pred_state is None:
        return

    if pred_state.get("task") == "classification":
        st.metric("Predicted Class", pred_state["predicted_class"])
        st.write("**Class probabilities**")
        st.dataframe(pred_state["probability_table"], use_container_width=True, hide_index=True)
        return

    st.write("**Forecast in modeled units**")
    st.dataframe(pred_state["forecast_table"], use_container_width=True)
    if pred_state["implied_prices"] is not None:
        st.write("**Implied future values in original scale**")
        st.dataframe(pred_state["implied_prices"], use_container_width=True)


def _apply_model_ui(model_name: str) -> None:
    if model_name == "ANN":
        ann_inject_css()
    elif model_name == "CNN":
        cnn_inject_css()
    elif model_name == "LSTM":
        lstm_inject_css()


def render(model_name: str) -> None:
    with model_state_context(model_name):
        _apply_model_ui(model_name)
        if model_name == "ANN":
            return render_ann_predict_ui()
        if model_name == "CNN":
            return render_cnn_predict_ui()
        if model_name == "LSTM":
            return render_lstm_predict_ui()
        st.error(f"Unknown model: {model_name}")
