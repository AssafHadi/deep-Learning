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

PAGE_NAME = "Preprocess"

def render_ann_preprocess_ui():
    st.title("Preprocess")
    df = st.session_state.raw_df
    cfg = st.session_state.config

    if df is None:
        st.info("Upload a dataset first.")
        return

    cols = df.columns.tolist()
    target_key = "__ann_preprocess_target"
    feature_key = "__ann_preprocess_features"
    drop_key = "__ann_preprocess_drop_columns"

    if target_key not in st.session_state or st.session_state[target_key] not in cols:
        st.session_state[target_key] = cfg["target_column"] if cfg["target_column"] in cols else cols[0]

    current_target = st.selectbox("Target Column", cols, key=target_key)
    available_cols = [c for c in cols if c != current_target]

    if feature_key not in st.session_state:
        default_features = [c for c in cfg["feature_columns"] if c in available_cols]
        if not default_features:
            default_features = available_cols.copy()
        st.session_state[feature_key] = default_features
    else:
        st.session_state[feature_key] = [c for c in st.session_state[feature_key] if c in available_cols]
        if not st.session_state[feature_key]:
            st.session_state[feature_key] = available_cols.copy()

    st.multiselect(
        "Feature Columns",
        available_cols,
        key=feature_key,
    )

    if drop_key not in st.session_state:
        st.session_state[drop_key] = [c for c in cfg["drop_columns"] if c in available_cols]
    else:
        st.session_state[drop_key] = [c for c in st.session_state[drop_key] if c in available_cols]

    st.multiselect(
        "Drop Columns",
        available_cols,
        key=drop_key,
        help="Optional: remove IDs, leakage columns, or irrelevant fields.",
    )

    cfg["target_column"] = st.session_state[target_key]
    cfg["feature_columns"] = [c for c in st.session_state[feature_key] if c != cfg["target_column"]]
    cfg["drop_columns"] = [c for c in st.session_state[drop_key] if c != cfg["target_column"]]

    c1, c2, c3, c4 = st.columns(4)
    cfg["auto_datetime"] = c1.checkbox("Expand datetime columns", value=bool(cfg["auto_datetime"]))
    cfg["drop_duplicates"] = c2.checkbox("Drop duplicates", value=bool(cfg["drop_duplicates"]))
    cfg["shuffle_data"] = c3.checkbox("Shuffle before split", value=bool(cfg["shuffle_data"]))
    cfg["use_class_weights"] = c4.checkbox("Class weights", value=bool(cfg["use_class_weights"]))

    c5, c6 = st.columns(2)
    cfg["test_size"] = c5.slider("Test Size", 0.10, 0.40, float(cfg["test_size"]), 0.05)
    cfg["random_seed"] = c6.number_input("Random Seed", min_value=0, max_value=999999, value=int(cfg["random_seed"]), step=1)

    st.session_state.config = cfg

    if st.button("Prepare Dataset", use_container_width=True):
        try:
            prepared = ann_prepare_dataset(df, cfg)
            st.session_state.prepared_data = prepared
            st.success("Dataset prepared successfully.")
        except Exception as e:
            st.error(f"Preparation failed: {e}")

    prepared = st.session_state.prepared_data
    if prepared:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Task", prepared["task"].title())
        c2.metric("Train Rows", prepared["X_train"].shape[0])
        c3.metric("Test Rows", prepared["X_test"].shape[0])
        c4.metric("Engineered Features", prepared["X_train"].shape[1])

        with st.expander("Prepared dataset details", expanded=True):
            st.write({
                "target_column": prepared["target_column"],
                "original_features": prepared["feature_columns_original"],
                "datetime_expanded_columns": prepared["datetime_expanded_columns"],
                "numeric_features": prepared["num_cols"],
                "categorical_features": prepared["cat_cols"],
            })


def render_cnn_preprocess_ui():
    cnn_hero()
    cnn_status_bar()
    st.subheader("Preprocess")

    cfg = st.session_state.model_config

    c1, c2, c3 = st.columns(3)
    with c1:
        cfg["image_size"] = st.selectbox("Input Size", [160, 192, 224, 256, 300],
                                         index=[160, 192, 224, 256, 300].index(cfg["image_size"]))
        cfg["batch_size"] = st.selectbox("Batch Size", [8, 16, 24, 32, 48],
                                         index=[8, 16, 24, 32, 48].index(cfg["batch_size"]))
    with c2:
        cfg["color_mode"] = st.selectbox("Color Mode", ["RGB", "Grayscale → 3-channel"],
                                         index=0 if cfg["color_mode"] == "RGB" else 1)
        cfg["seed"] = st.number_input("Seed", 1, 999999, int(cfg["seed"]))
    with c3:
        cfg["shuffle_buffer"] = st.selectbox("Shuffle Buffer", [256, 512, 1024, 2048],
                                             index=[256, 512, 1024, 2048].index(cfg["shuffle_buffer"]))

    st.write("### Augmentation")
    a1, a2, a3, a4 = st.columns(4)
    aug = cfg["augmentation"]
    with a1:
        aug["flip"] = st.checkbox("Horizontal Flip", value=aug["flip"])
    with a2:
        aug["rotation"] = st.slider("Rotation", 0.0, 0.25, float(aug["rotation"]), 0.01)
    with a3:
        aug["zoom"] = st.slider("Zoom", 0.0, 0.30, float(aug["zoom"]), 0.01)
    with a4:
        aug["contrast"] = st.slider("Contrast", 0.0, 0.30, float(aug["contrast"]), 0.01)

    st.caption("These augmentations mimic the standard MATLAB-style image pipeline: resize, random transform, transfer learn.")


def render_lstm_preprocess_ui():
    lstm_hero("Preprocess", "Choose columns, clean the series, split it sequentially, and build LSTM windows.")
    df = st.session_state.get("raw_df")
    if df is None:
        st.info("Load data first.")
        return

    cfg = st.session_state["config"]
    guessed_date = lstm_detect_date_column(df)
    numeric_cols = lstm_get_numeric_columns(df, exclude=[guessed_date] if guessed_date else [])
    all_cols = list(df.columns)

    if guessed_date and (cfg.get("date_col") is None or cfg.get("date_col") not in df.columns):
        cfg["date_col"] = guessed_date
    if not cfg.get("feature_cols"):
        cfg["feature_cols"] = numeric_cols[:]
    if cfg.get("task_mode", "Regression") == "Regression" and not cfg.get("target_cols"):
        cfg["target_cols"] = numeric_cols[: min(1, len(numeric_cols))]
    if cfg.get("task_mode", "Regression") == "Classification" and not cfg.get("classification_target_col"):
        non_date_cols = [c for c in all_cols if c != cfg.get("date_col")]
        cfg["classification_target_col"] = non_date_cols[0] if non_date_cols else None

    with st.form("preprocess_form"):
        st.markdown("#### Column selection and sequence setup")
        a1, a2, a3 = st.columns(3)
        with a1:
            date_options = [None] + all_cols
            date_index = date_options.index(cfg["date_col"]) if cfg.get("date_col") in date_options else 0
            cfg["date_col"] = st.selectbox("Date column", options=date_options, index=date_index)
        with a2:
            cfg["feature_cols"] = st.multiselect(
                "Feature columns",
                options=numeric_cols,
                default=[c for c in cfg.get("feature_cols", []) if c in numeric_cols],
                help="Numeric sequence inputs used by the LSTM.",
            )
        with a3:
            if cfg.get("task_mode", "Regression") == "Classification":
                options = [c for c in all_cols if c != cfg.get("date_col")]
                current = cfg.get("classification_target_col")
                idx = options.index(current) if current in options else 0
                cfg["classification_target_col"] = st.selectbox("Classification target column", options=options, index=idx)
                cfg["target_cols"] = [cfg["classification_target_col"]]
            else:
                cfg["target_cols"] = st.multiselect(
                    "Regression target column(s)",
                    options=numeric_cols,
                    default=[c for c in cfg.get("target_cols", []) if c in numeric_cols],
                    help="Numeric future value(s) the LSTM forecasts.",
                )

        b1, b2 = st.columns(2)
        with b1:
            cfg["lookback"] = st.number_input("Lookback window", min_value=2, max_value=365, value=int(cfg["lookback"]), step=1)
        with b2:
            cfg["horizon"] = st.number_input("Forecast horizon", min_value=1, max_value=30, value=int(cfg["horizon"]), step=1)

        st.markdown("#### Cleaning, scaling, and split")
        c1, c2, c3 = st.columns(3)
        with c1:
            cfg["transform_mode"] = st.selectbox(
                "Feature/target transform",
                ["raw", "pct_change", "log_return"],
                index=["raw", "pct_change", "log_return"].index(cfg["transform_mode"]),
                help="For classification, this is applied only to numeric input features, not to the class target.",
            )
            cfg["missing_method"] = st.selectbox(
                "Missing value handling",
                ["ffill_bfill", "interpolate", "drop", "ffill", "bfill", "median_impute"],
                index=["ffill_bfill", "interpolate", "drop", "ffill", "bfill", "median_impute"].index(cfg["missing_method"]),
            )
            resample_options = ["None", "D", "W", "M"]
            if cfg.get("task_mode") == "Classification":
                resample_options = ["None"]
                cfg["resample_rule"] = "None"
            cfg["resample_rule"] = st.selectbox("Resample rule", resample_options, index=resample_options.index(cfg["resample_rule"]))
        with c2:
            cfg["scale_method"] = st.selectbox("Scaling", ["standard", "minmax", "robust"], index=["standard", "minmax", "robust"].index(cfg["scale_method"]))
            cfg["train_frac"] = st.slider("Train fraction", min_value=0.50, max_value=0.85, value=float(cfg["train_frac"]), step=0.05)
            cfg["val_frac"] = st.slider("Validation fraction", min_value=0.05, max_value=0.30, value=float(cfg["val_frac"]), step=0.05)
        with c3:
            cfg["clip_outliers"] = st.checkbox("Clip outliers by quantile", value=bool(cfg["clip_outliers"]))
            cfg["clip_low_q"] = st.number_input("Lower quantile", min_value=0.0, max_value=0.20, value=float(cfg["clip_low_q"]), step=0.005)
            cfg["clip_high_q"] = st.number_input("Upper quantile", min_value=0.80, max_value=1.0, value=float(cfg["clip_high_q"]), step=0.005)

        submitted = st.form_submit_button("Run Preprocessing", use_container_width=True)

    if submitted:
        try:
            if cfg["train_frac"] + cfg["val_frac"] >= 0.95:
                raise ValueError("Train fraction + validation fraction must leave room for a test set.")
            processed = lstm_preprocess_dataset(df, cfg)
            st.session_state["processed"] = processed
            st.session_state["training"] = None
            st.session_state["prediction"] = None
            st.success("Preprocessing complete.")
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")

    processed = st.session_state.get("processed")
    if processed is not None:
        st.markdown("### Processed Dataset Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Task", processed["task"].title())
        c2.metric("Train Sequences", len(processed["X_train"]))
        c3.metric("Validation Sequences", len(processed["X_val"]))
        c4.metric("Test Sequences", len(processed["X_test"]))

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Lookback", int(cfg["lookback"]))
        c6.metric("Horizon", int(cfg["horizon"]))
        c7.metric("Features", len(processed["feature_cols"]))
        c8.metric("Target", processed["target_col"])

        st.pyplot(lstm_fig_split_overview(processed["dates"], processed["train_end_idx"], processed["val_end_idx"], "Sequential Train / Validation / Test Split"), use_container_width=True)
        st.dataframe(
            pd.DataFrame({
                "setting": ["Task", "Date column", "Features", "Target", "Frequency", "Transform", "Scaling"],
                "value": [
                    processed["task"].title(),
                    cfg.get("date_col"),
                    ", ".join(processed["feature_cols"]),
                    processed["target_col"],
                    processed["freq_name"],
                    cfg["transform_mode"],
                    cfg["scale_method"],
                ],
            }),
            use_container_width=True,
            hide_index=True,
        )


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
            return render_ann_preprocess_ui()
        if model_name == "CNN":
            return render_cnn_preprocess_ui()
        if model_name == "LSTM":
            return render_lstm_preprocess_ui()
        st.error(f"Unknown model: {model_name}")
