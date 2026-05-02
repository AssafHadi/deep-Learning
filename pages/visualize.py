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
from visualization.plots import _lstm_target_label_from_columns
from storage.persistence import *

from pages.home import ann_inject_css, ann_hero, cnn_inject_css, cnn_hero, cnn_status_bar, cnn_render_uploaded_images_section, lstm_inject_css, lstm_hero, lstm_info_card, lstm_top_status_bar

PAGE_NAME = "Visualize"

def render_ann_visualize_ui():
    st.title("Visualize")
    df = st.session_state.raw_df
    if df is not None:
        ann_plot_data_profile(df, st.session_state.config.get("target_column"))

    st.subheader("Training Curves")
    history = st.session_state.training_history
    if history:
        base_metrics = ["loss"] + sorted({
            (k[4:] if k.startswith("val_") else k)
            for k in history.keys()
            if k != "val_loss" and not k.startswith("val_val_")
        } - {"loss"})
        for i in range(0, len(base_metrics), 2):
            row = st.columns(2)
            for j, metric in enumerate(base_metrics[i:i+2]):
                with row[j]:
                    st.pyplot(ann_make_training_curve_fig(history, metric), clear_figure=True, use_container_width=True)
    else:
        st.info("No training history available yet.")

    results = st.session_state.results
    if not results:
        return

    st.subheader("Model Performance Visuals")
    if results["task"] == "classification":
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(ann_make_confusion_fig(results["y_true"], results["y_pred"], results["class_names"]), clear_figure=True, use_container_width=True)
        with c2:
            st.pyplot(ann_make_predicted_distribution_fig(results["y_pred"]), clear_figure=True, use_container_width=True)

        if len(results["class_names"]) == 2:
            fig1, fig2 = ann_make_binary_curve_figs(results["y_true"], results.get("y_score"))
            c3, c4 = st.columns(2)
            with c3:
                st.pyplot(fig1, clear_figure=True, use_container_width=True)
            with c4:
                st.pyplot(fig2, clear_figure=True, use_container_width=True)
    else:
        fig1, fig2, fig3, fig4 = ann_make_regression_figures(np.asarray(results["y_true"]), np.asarray(results["y_pred"]))
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig1, clear_figure=True, use_container_width=True)
        with c2:
            st.pyplot(fig2, clear_figure=True, use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.pyplot(fig3, clear_figure=True, use_container_width=True)
        with c4:
            st.pyplot(fig4, clear_figure=True, use_container_width=True)


def render_cnn_visualize_ui():
    cnn_hero()
    cnn_status_bar()
    st.subheader("Visualize")

    df = st.session_state.dataset_df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Training", "Validation", "Explainability"])

    with tab1:
        cnn_plot_class_distribution(df)
        cnn_plot_image_dimensions(df)
        cnn_show_sample_gallery(df, n_per_class=5)

    with tab2:
        cnn_plot_training_curves(st.session_state.history)

    with tab3:
        if st.session_state.eval_artifacts is None:
            st.info("Train the model to unlock validation plots.")
        else:
            ev = st.session_state.eval_artifacts
            cnn_plot_confusion_matrices(ev["confusion_matrix"], ev["class_names"])
            cnn_plot_per_class_metrics(ev["classification_report"], ev["class_names"])
            cnn_plot_prediction_confidence_distribution(ev)
            cnn_plot_correct_incorrect_summary(ev)
            cnn_plot_roc_pr(ev)
            cnn_plot_embedding_map(st.session_state.trained_model, ev, st.session_state.model_config)
            cnn_show_misclassified(ev, max_items=12)
            cnn_show_correct_incorrect_examples(ev, max_each=4)

    with tab4:
        model = st.session_state.trained_model
        if model is None:
            st.info("Train or load a model first.")
        else:
            source = st.selectbox("Explainability Image Source", ["Validation sample", "Upload custom image"])
            cfg = st.session_state.model_config
            if source == "Validation sample":
                ev = st.session_state.eval_artifacts
                if ev is None:
                    st.info("No validation results available.")
                else:
                    path = st.selectbox("Select validation image", ev["val_paths"][:min(200, len(ev["val_paths"]))])
                    img = Image.open(path)
                    arr = cnn_prepare_single_image(img, cfg)
                    probs = model.predict(arr, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    st.write(f"**Predicted:** {st.session_state.class_names[pred_idx]} ({probs[pred_idx]:.4f})")
                    c1, c2 = st.columns(2, gap="small")
                    c1.image(img, caption="Original", use_container_width=True)
                    heatmap = cnn_gradcam_heatmap(model, arr)
                    overlay = cnn_overlay_heatmap_on_image(img.resize((cfg["image_size"], cfg["image_size"])), heatmap)
                    if overlay is not None:
                        c2.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            else:
                up = st.file_uploader("Upload image", key="viz_custom_img", help="JPG/PNG/BMP/TIFF/WEBP/GIF work best. HEIC usually needs conversion.")
                if up is not None:
                    img, err = cnn_open_uploaded_image(up)
                    if err:
                        st.error(err)
                        return
                    arr = cnn_prepare_single_image(img, cfg)
                    probs = model.predict(arr, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    st.write(f"**Predicted:** {st.session_state.class_names[pred_idx]} ({probs[pred_idx]:.4f})")
                    c1, c2 = st.columns(2, gap="small")
                    c1.image(img, caption="Original", use_container_width=True)
                    heatmap = cnn_gradcam_heatmap(model, arr)
                    overlay = cnn_overlay_heatmap_on_image(img.resize((cfg["image_size"], cfg["image_size"])), heatmap)
                    if overlay is not None:
                        c2.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)


def render_lstm_visualize_ui():
    lstm_hero("Visualize", "Task-aware plots: regression diagnostics for forecasting, classification diagnostics for class prediction.")
    training = st.session_state.get("training")
    processed = st.session_state.get("processed")
    if training is None or processed is None:
        st.info("Train the model first.")
        return

    cfg = st.session_state["config"]

    st.subheader("Data and training overview")
    ov1, ov2 = st.columns(2)
    if isinstance(processed["dates"].iloc[0], pd.Timestamp):
        frame = processed["transformed"].copy()
        frame.index = processed["dates"]
        with ov1:
            if processed.get("task") == "regression":
                y_label = _lstm_target_label_from_columns(processed["target_cols"], cfg.get("transform_mode", "raw"))
                lstm_show_lstm_fig(lstm_fig_line(frame[processed["target_cols"]], "Target Series Over Time", ylabel=y_label))
            else:
                lstm_show_lstm_fig(lstm_fig_line(frame[processed["feature_cols"][: min(4, len(processed["feature_cols"]))]], "Input Feature Series Over Time", ylabel="Transformed / Scaled Feature Value"))
    else:
        with ov1:
            st.info("Date-based overview is only shown when a valid date column exists.")
    with ov2:
        lstm_show_lstm_fig(lstm_fig_training_history(training["history"]))

    if processed.get("task") == "classification":
        st.subheader("Classification diagnostics")
        split = st.selectbox("Visualization split", ["train", "val", "test"], index=2)
        class_names = processed["class_names"]
        y_true = training["actuals"][split]
        y_pred = training["predictions"][split]
        y_prob = training["probabilities"][split]

        a, b = st.columns(2)
        with a:
            lstm_show_lstm_fig(lstm_fig_lstm_confusion(y_true, y_pred, class_names, f"{split.title()} Confusion Matrix"))
        with b:
            lstm_show_lstm_fig(lstm_fig_class_distribution(y_true, y_pred, class_names, f"{split.title()} Class Distribution"))

        c, d = st.columns(2)
        with c:
            lstm_show_lstm_fig(lstm_fig_class_probability_hist(y_prob, y_pred, class_names, f"{split.title()} Prediction Confidence"))
        with d:
            report = pd.DataFrame(classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)).T
            st.dataframe(report, use_container_width=True)

        if len(class_names) == 2:
            fig1, fig2 = lstm_fig_binary_roc_pr(y_true, y_prob, split.title())
            e, f = st.columns(2)
            with e:
                lstm_show_lstm_fig(fig1)
            with f:
                lstm_show_lstm_fig(fig2)
        return

    st.subheader("Prediction diagnostics")
    split = st.selectbox("Visualization split", ["train", "val", "test"], index=2)
    target = st.selectbox("Visualization target", processed["target_cols"], index=0)
    horizon_step = st.selectbox("Visualization horizon step", list(range(1, int(cfg["horizon"]) + 1)), index=0)

    idx = processed["target_cols"].index(target)
    h = horizon_step - 1
    actual = training["actuals"][split][:, h, idx]
    pred = training["predictions"][split][:, h, idx]
    residuals = actual - pred

    a, b = st.columns(2)
    with a:
        lstm_show_lstm_fig(lstm_fig_actual_vs_predicted(actual, pred, f"{split.title()} | {target} | Actual vs Predicted"))
    with b:
        lstm_show_lstm_fig(lstm_fig_scatter(actual, pred, f"{split.title()} | {target} | Scatter"))

    c, d = st.columns(2)
    with c:
        lstm_show_lstm_fig(lstm_fig_residuals_vs_pred(pred, residuals, f"{split.title()} | {target} | Residuals vs Predicted"))
    with d:
        lstm_show_lstm_fig(lstm_fig_rolling_rmse(actual, pred, f"{split.title()} | {target} | Rolling RMSE", window=max(10, int(cfg['lookback']))))

    e, f = st.columns(2)
    with e:
        lstm_show_lstm_fig(lstm_fig_distribution(actual, pred, f"{split.title()} | {target} | Distribution"))
    with f:
        lstm_show_lstm_fig(lstm_fig_qq(residuals, f"{split.title()} | {target} | QQ Plot"))

    if cfg["transform_mode"] in {"pct_change", "log_return"} and int(cfg["horizon"]) >= 1:
        st.subheader("Signal-driven strategy simulation (MATLAB-style idea, simplified)")
        dates = training["times"][split]
        if np.asarray(dates).ndim == 2:
            dates = np.asarray(dates)[:, 0]
        actual_first = training["actuals"][split][:, 0, :]
        pred_first = training["predictions"][split][:, 0, :]
        bt_returns = lstm_returns_for_backtest(cfg["transform_mode"], actual_first, pred_first)
        equity, dd = lstm_backtest_strategies(bt_returns[0], bt_returns[1], dates, processed["target_cols"], processed["freq_name"])
        g, hcol = st.columns(2)
        with g:
            lstm_show_lstm_fig(lstm_fig_equity_curves(equity, "Strategy Equity Curves"))
        with hcol:
            lstm_show_lstm_fig(lstm_fig_drawdowns(dd, "Strategy Drawdowns"))


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
            return render_ann_visualize_ui()
        if model_name == "CNN":
            return render_cnn_visualize_ui()
        if model_name == "LSTM":
            return render_lstm_visualize_ui()
        st.error(f"Unknown model: {model_name}")
