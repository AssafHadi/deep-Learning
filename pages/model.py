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

PAGE_NAME = "Model"


def render_ann_model_ui():
    st.title("Model")
    cfg = st.session_state.config

    with st.form("model_form"):
        left, right = st.columns(2)

        with left:
            cfg["project_name"] = st.text_input(
                "Project Name",
                value=cfg["project_name"],
            )

            cfg["task_mode"] = st.selectbox(
                "Task",
                ["Auto Detect", "Regression", "Classification"],
                index=["Auto Detect", "Regression", "Classification"].index(cfg["task_mode"]),
            )

            cfg["hidden_layers"] = st.text_input(
                "Hidden Layers (comma separated)",
                value=cfg["hidden_layers"],
            )

            cfg["activation"] = st.selectbox(
                "Activation",
                ["relu", "tanh", "elu", "selu"],
                index=["relu", "tanh", "elu", "selu"].index(cfg["activation"]),
            )

            cfg["dropout"] = st.slider(
                "Dropout",
                0.0,
                0.7,
                float(cfg["dropout"]),
                0.05,
            )

            cfg["batch_norm"] = st.checkbox(
                "Use Batch Normalization",
                value=bool(cfg["batch_norm"]),
            )

            cfg["l2_reg"] = st.number_input(
                "L2 Regularization",
                min_value=0.0,
                max_value=1.0,
                value=float(cfg["l2_reg"]),
                step=0.0001,
                format="%.4f",
            )

        with right:
            cfg["optimizer"] = st.selectbox(
                "Optimizer",
                ["adam", "sgd", "rmsprop"],
                index=["adam", "sgd", "rmsprop"].index(cfg["optimizer"]),
            )

            cfg["validation_split"] = st.slider(
                "Validation Split",
                0.05,
                0.40,
                float(cfg["validation_split"]),
                0.05,
            )

            cfg["use_class_weights"] = st.checkbox(
                "Use Class Weights (classification)",
                value=bool(cfg["use_class_weights"]),
            )

            cfg["threshold"] = st.slider(
                "Binary Threshold",
                0.05,
                0.95,
                float(cfg["threshold"]),
                0.01,
            )

        submitted = st.form_submit_button("Save Model Settings", use_container_width=True)

    if submitted:
        st.session_state.config = cfg
        st.success("Model settings updated.")

    st.markdown("### Current architecture preview")
    layers = ann_parse_hidden_layers(cfg["hidden_layers"])
    st.code(
        "Input -> " + " -> ".join([f"Dense({x})" for x in layers]) + " -> Output",
        language="text",
    )


def render_cnn_model_ui():
    cnn_hero()
    cnn_status_bar()
    st.subheader("Model")

    cfg = st.session_state.model_config
    c1, c2, c3 = st.columns(3)

    with c1:
        cfg["backbone"] = st.selectbox(
            "Backbone",
            ["MobileNetV2", "EfficientNetB0", "ResNet50"],
            index=["MobileNetV2", "EfficientNetB0", "ResNet50"].index(cfg["backbone"]),
        )

        cfg["weights"] = st.selectbox(
            "Initial Weights",
            ["imagenet", None],
            index=0 if cfg["weights"] == "imagenet" else 1,
        )

        cfg["optimizer"] = st.selectbox(
            "Optimizer",
            ["Adam", "RMSprop", "SGD"],
            index=["Adam", "RMSprop", "SGD"].index(cfg["optimizer"]),
        )

    with c2:
        cfg["dense_units"] = st.select_slider(
            "Dense Units",
            options=[64, 128, 256, 512],
            value=cfg["dense_units"],
        )

        cfg["dropout"] = st.slider(
            "Dropout",
            0.0,
            0.7,
            float(cfg["dropout"]),
            0.05,
        )

        cfg["label_smoothing"] = 0.0
        st.caption("Label smoothing is disabled to keep sparse-label training compatible with your TensorFlow/Keras build.")

    with c3:
        cfg["learning_rate"] = st.select_slider(
            "Learning Rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3],
            value=cfg["learning_rate"],
        )

        cfg["fine_tune"] = st.checkbox(
            "Enable Fine-Tuning",
            value=cfg["fine_tune"],
        )

        cfg["use_class_weights"] = st.checkbox(
            "Use Class Weights",
            value=cfg["use_class_weights"],
        )

    if cfg["fine_tune"]:
        c4, c5 = st.columns(2)

        with c4:
            cfg["unfreeze_layers"] = st.slider(
                "Unfreeze Last N Layers",
                5,
                120,
                int(cfg["unfreeze_layers"]),
                5,
            )

        with c5:
            cfg["fine_tune_lr"] = st.select_slider(
                "Fine-Tune LR",
                options=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
                value=cfg["fine_tune_lr"],
            )

    st.info("Recommended starting point for small oil & gas image datasets: MobileNetV2 + ImageNet weights + fine-tuning.")


def render_lstm_model_ui():
    lstm_hero("Model", "Choose the LSTM task and architecture. Data columns and splits belong in Preprocess.")
    df = st.session_state.get("raw_df")

    if df is None:
        st.info("Load data first.")
        return

    cfg = st.session_state["config"]
    task_options = ["Regression", "Classification"]

    if cfg.get("task_mode") not in task_options:
        cfg["task_mode"] = "Regression"

    with st.form("model_form"):
        c1, c2 = st.columns(2)

        with c1:
            cfg["project_name"] = st.text_input("Project name", value=cfg["project_name"])
            cfg["task_mode"] = st.selectbox("Task", task_options, index=task_options.index(cfg["task_mode"]))
            cfg["lstm_units_1"] = st.number_input("LSTM units 1", min_value=8, max_value=512, value=int(cfg["lstm_units_1"]), step=8)
            cfg["lstm_units_2"] = st.number_input("LSTM units 2", min_value=8, max_value=512, value=int(cfg["lstm_units_2"]), step=8)

        with c2:
            cfg["lstm_units_3"] = st.number_input("LSTM units 3", min_value=8, max_value=512, value=int(cfg["lstm_units_3"]), step=8)
            cfg["dense_units"] = st.number_input("Dense units", min_value=8, max_value=512, value=int(cfg["dense_units"]), step=8)
            cfg["dropout"] = st.slider("Dropout", min_value=0.0, max_value=0.8, value=float(cfg["dropout"]), step=0.05)
            cfg["seed"] = st.number_input("Random seed", min_value=0, max_value=999999, value=int(cfg["seed"]), step=1)

        submitted = st.form_submit_button("Save Model Settings", use_container_width=True)

    if submitted:
        st.session_state["config"] = cfg
        st.session_state["processed"] = None
        st.session_state["training"] = None
        st.session_state["prediction"] = None
        st.success("Model settings saved. Re-run preprocessing before training.")

    output = "future numeric value(s)" if cfg["task_mode"] == "Regression" else "future class label"

    st.markdown("### Architecture preview")
    st.code(
        f"Input sequence → LSTM({cfg['lstm_units_1']}) → LSTM({cfg['lstm_units_2']}) → LSTM({cfg['lstm_units_3']}) → Dense({cfg['dense_units']}) → {output}",
        language="text",
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
            return render_ann_model_ui()

        if model_name == "CNN":
            return render_cnn_model_ui()

        if model_name == "LSTM":
            return render_lstm_model_ui()

        st.error(f"Unknown model: {model_name}")
