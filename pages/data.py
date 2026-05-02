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

from core.state import model_state_context, lstm_reset_after_data_change
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

PAGE_NAME = "Data Upload"

def render_ann_data_ui():
    st.title("Data Upload")
    st.caption("Load a CSV/XLSX file containing oil & gas tabular data.")
    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = ann_read_uploaded_table(uploaded)
            st.session_state.raw_df = df
            if not st.session_state.config["feature_columns"]:
                st.session_state.config["feature_columns"] = [c for c in df.columns]
            st.success(f"Loaded {uploaded.name} successfully.")
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    df = st.session_state.raw_df
    if df is None:
        st.info("Upload a file to continue.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(df))
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Cells", int(df.isna().sum().sum()))

    ann_plot_data_profile(df, st.session_state.config.get("target_column"))


def render_cnn_data_ui():
    cnn_hero()
    cnn_status_bar()
    st.subheader("Data Upload")

    cfg = st.session_state.model_config
    col1, col2 = st.columns([1, 1])
    with col1:
        st.session_state.project_name = st.text_input("Project Name", value=st.session_state.project_name)
    with col2:
        cfg["val_ratio"] = st.slider("Validation Ratio", 0.1, 0.4, float(cfg["val_ratio"]), 0.05)

    st.write("### Upload Dataset ZIP")
    uploaded_zip = st.file_uploader("Upload ZIP dataset for training", type=["zip"], key="dataset_zip_uploader")

    if uploaded_zip is not None:
        if st.button("Load Dataset", type="primary", use_container_width=True):
            with st.spinner("Extracting and indexing dataset..."):
                cnn_seed_everything(cfg["seed"])
                extract_dir = cnn_save_uploaded_zip(uploaded_zip, st.session_state.project_name)
                df = cnn_infer_dataset_structure(extract_dir)
                df, dropped_files = cnn_filter_valid_images(df)
                if df.empty:
                    raise ValueError("No valid readable images were found after validation. Remove corrupted, unsupported, or HEIC/HEIF files and try again.")
                counts = df["label"].value_counts()
                valid_labels = counts[counts >= 2].index.tolist()
                df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
                if df.empty:
                    raise ValueError("After removing invalid images, each class must still contain at least 2 readable images.")
                df = cnn_finalize_splits(df, cfg["val_ratio"], cfg["seed"])
                st.session_state.dataset_root = str(extract_dir)
                st.session_state.dataset_df = df
                st.session_state.class_names = sorted(df["label"].unique().tolist())
                st.session_state.data_summary = cnn_dataset_summary(df)
                st.session_state.training_complete = False
                st.session_state.trained_model = None
                st.session_state.history = None
                st.session_state.eval_artifacts = None
                st.session_state.invalid_dataset_files = dropped_files
            st.success("Dataset loaded successfully.")
            if st.session_state.get("invalid_dataset_files"):
                dropped_df = pd.DataFrame(st.session_state.invalid_dataset_files)
                st.warning(f"Ignored {len(dropped_df)} unreadable/unsupported image files from the ZIP.")
                st.dataframe(dropped_df.head(20), use_container_width=True)



    df = st.session_state.dataset_df
    if df is not None:
        st.write("### Dataset Preview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Images", len(df))
        c2.metric("Classes", df["label"].nunique())
        c3.metric("Train", int((df["split"] == "train").sum()))
        c4.metric("Validation", int((df["split"] == "val").sum()))

        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns([1.0, 1.0], gap="small")
        with col1:
            cnn_plot_class_distribution(df)
        with col2:
            cnn_plot_image_dimensions(df)

        st.write("### Sample Gallery")
        cnn_show_sample_gallery(df, n_per_class=5)


def render_lstm_data_ui():
    lstm_hero("Data Upload", "Load oil and gas market data and inspect it before you poison the model with garbage.")
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"], key="main_uploader")

    if uploaded is not None:
        try:
            df = lstm_read_uploaded_data(uploaded)
            st.session_state["raw_df"] = df
            st.session_state["loaded_filename"] = uploaded.name
            lstm_reset_after_data_change()
            detected_date = lstm_detect_date_column(df)
            if detected_date and not st.session_state["config"].get("date_col"):
                st.session_state["config"]["date_col"] = detected_date
            st.success(f"Loaded {uploaded.name} with shape {df.shape}.")
        except Exception as e:
            st.error(f"File load failed: {e}")

    df = st.session_state.get("raw_df")
    if df is None:
        st.info("Upload a dataset first.")
        return

    st.write(f"**Loaded file:** {st.session_state.get('loaded_filename')}  ")
    st.dataframe(df.head(20), use_container_width=True)

    detected_date = lstm_detect_date_column(df)
    numeric_cols = lstm_get_numeric_columns(df, exclude=[detected_date] if detected_date else [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", df.shape[1])
    c3.metric("Detected Date Column", detected_date or "None")
    c4.metric("Numeric-like Columns", len(numeric_cols))

    left, right = st.columns([1.2, 1])
    with left:
        st.pyplot(lstm_fig_missing_bars(df, "Missing Values by Column"), use_container_width=True)
    with right:
        profile = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(x) for x in df.dtypes],
                "missing": df.isna().sum().values,
                "missing_%": (df.isna().mean().values * 100.0).round(2),
            }
        )
        st.dataframe(profile, use_container_width=True, hide_index=True)


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
            return render_ann_data_ui()
        if model_name == "CNN":
            return render_cnn_data_ui()
        if model_name == "LSTM":
            return render_lstm_data_ui()
        st.error(f"Unknown model: {model_name}")
