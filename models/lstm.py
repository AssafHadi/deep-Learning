from __future__ import annotations

import os
import io
import re
import json
import math
import time
import shutil
import random
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # pragma: no cover
    Image = ImageOps = UnidentifiedImageError = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix, f1_score,
        mean_absolute_error, mean_squared_error, precision_recall_curve,
        precision_score, r2_score, recall_score, roc_auc_score, roc_curve,
        balanced_accuracy_score, auc, average_precision_score, log_loss,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, label_binarize
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.decomposition import PCA
except Exception:  # pragma: no cover
    pass

MODEL_NAME = "LSTM"

# Model-specific constants and algorithms extracted from the original implementation.
LSTM_MODEL_NAME = "LSTM"


_LSTM_TF_CACHE: Dict[str, object] = {}


LSTM_APP_TITLE = "Oil & Gas LSTM Studio"


LSTM_APP_SUBTITLE = "MATLAB-style workflow for multivariate energy forecasting with LSTM"


LSTM_BASE_DIR = Path(".oil_gas_lstm_studio")


LSTM_PROJECTS_DIR = LSTM_BASE_DIR / "projects"


LSTM_TEMP_IMPORT_DIR = LSTM_BASE_DIR / "imports"


for p in [LSTM_BASE_DIR, LSTM_PROJECTS_DIR, LSTM_TEMP_IMPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


LSTM_PLOT_W = 5.15


LSTM_PLOT_H = 2.55


LSTM_PLOT_WIDE = 6.4


LSTM_PLOT_DPI = 130


LSTM_MAX_PLOT_POINTS = 520


LSTM_LSTM_COLORS = {
    "blue": "#1d4ed8",
    "orange": "#f97316",
    "green": "#16a34a",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "gray": "#64748b",
    "grid": "#cbd5e1",
    "text": "#0f172a",
}


def lstm_get_tf():
    if "tf" not in _LSTM_TF_CACHE:
        import tensorflow as tf

        _LSTM_TF_CACHE["tf"] = tf
    return _LSTM_TF_CACHE["tf"]


def lstm_default_config() -> Dict:
    return {
        "task_mode": "Regression",
        "date_col": None,
        "feature_cols": [],
        "target_cols": [],
        "classification_target_col": None,
        "lookback": 30,
        "horizon": 1,
        "train_frac": 0.70,
        "val_frac": 0.15,
        "transform_mode": "raw",
        "missing_method": "ffill_bfill",
        "resample_rule": "None",
        "scale_method": "standard",
        "clip_outliers": False,
        "clip_low_q": 0.01,
        "clip_high_q": 0.99,
        "lstm_units_1": 64,
        "lstm_units_2": 64,
        "lstm_units_3": 32,
        "dense_units": 32,
        "dropout": 0.20,
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 32,
        "patience": 10,
        "loss": "mse",
        "seed": 42,
        "project_name": "oil_gas_lstm_project",
    }


def lstm_make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError("Unknown scaler")


def lstm_create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: np.ndarray,
    lookback: int,
    horizon: int,
    train_end_idx: int,
    val_end_idx: int,
) -> Dict:
    X_seq, y_seq = [], []
    target_times = []
    target_last_row_idx = []
    split_tags = []

    n_rows = len(X)
    for end in range(lookback, n_rows - horizon + 1):
        x_block = X[end - lookback : end, :]
        y_block = y[end : end + horizon, :]
        last_target_idx = end + horizon - 1
        X_seq.append(x_block)
        y_seq.append(y_block)
        target_times.append(timestamps[end : end + horizon])
        target_last_row_idx.append(last_target_idx)
        if last_target_idx < train_end_idx:
            split_tags.append("train")
        elif last_target_idx < val_end_idx:
            split_tags.append("val")
        else:
            split_tags.append("test")

    return {
        "X_seq": np.asarray(X_seq, dtype=np.float32),
        "y_seq": np.asarray(y_seq, dtype=np.float32),
        "target_times": np.asarray(target_times, dtype=object),
        "target_last_row_idx": np.asarray(target_last_row_idx),
        "split_tags": np.asarray(split_tags),
    }


def lstm_create_classification_sequences(
    X: np.ndarray,
    y_labels: np.ndarray,
    timestamps: np.ndarray,
    lookback: int,
    horizon: int,
    train_end_idx: int,
    val_end_idx: int,
) -> Dict:
    X_seq, y_seq = [], []
    target_times = []
    target_last_row_idx = []
    split_tags = []

    n_rows = len(X)
    for end in range(lookback, n_rows - horizon + 1):
        x_block = X[end - lookback : end, :]
        label_idx = end + horizon - 1
        X_seq.append(x_block)
        y_seq.append(y_labels[label_idx])
        target_times.append(timestamps[label_idx])
        target_last_row_idx.append(label_idx)
        if label_idx < train_end_idx:
            split_tags.append("train")
        elif label_idx < val_end_idx:
            split_tags.append("val")
        else:
            split_tags.append("test")

    return {
        "X_seq": np.asarray(X_seq, dtype=np.float32),
        "y_seq": np.asarray(y_seq, dtype=np.int64),
        "target_times": np.asarray(target_times, dtype=object),
        "target_last_row_idx": np.asarray(target_last_row_idx),
        "split_tags": np.asarray(split_tags),
    }


def lstm_inverse_3d(flat_pred: np.ndarray, scaler, horizon: int, num_targets: int) -> np.ndarray:
    arr = flat_pred.reshape(-1, horizon, num_targets)
    inv = np.empty_like(arr, dtype=np.float64)
    for step in range(horizon):
        inv[:, step, :] = scaler.inverse_transform(arr[:, step, :])
    return inv


def lstm_flatten_y(y_3d: np.ndarray) -> np.ndarray:
    return y_3d.reshape(y_3d.shape[0], -1)


def lstm_model_summary_text(model) -> str:
    lines: List[str] = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def lstm_build_model(input_shape: Tuple[int, int], num_targets: int, horizon: int, cfg: Dict, n_classes: Optional[int] = None):
    tf = lstm_get_tf()
    tf.keras.backend.clear_session()
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(int(cfg["seed"]))

    task = str(cfg.get("task_mode", "Regression")).lower()

    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(int(cfg["lstm_units_1"]), return_sequences=True)(inp)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(float(cfg["dropout"]))(x)

    x = tf.keras.layers.LSTM(int(cfg["lstm_units_2"]), return_sequences=True)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(float(cfg["dropout"]))(x)

    x = tf.keras.layers.LSTM(int(cfg["lstm_units_3"]), return_sequences=False)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(float(cfg["dropout"]))(x)

    x = tf.keras.layers.Dense(int(cfg["dense_units"]), activation="relu")(x)
    x = tf.keras.layers.Dropout(float(cfg["dropout"]))(x)

    if task == "classification":
        n_classes = int(n_classes or 2)
        if n_classes == 2:
            out = tf.keras.layers.Dense(1, activation="sigmoid", name="class_probability")(x)
            loss_fn = "binary_crossentropy"
            metrics = [tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        else:
            out = tf.keras.layers.Dense(n_classes, activation="softmax", name="class_probabilities")(x)
            loss_fn = "sparse_categorical_crossentropy"
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    else:
        out = tf.keras.layers.Dense(horizon * num_targets, name="forecast")(x)
        loss_name = cfg["loss"]
        if loss_name == "huber":
            loss_fn = tf.keras.losses.Huber()
        else:
            loss_fn = loss_name
        metrics = [tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")]

    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
        loss=loss_fn,
        metrics=metrics,
    )
    return model


# Public architecture aliases requested by the layered design.
build_lstm_model = lstm_build_model
create_lstm_sequences = lstm_create_sequences
get_lstm_default_config = lstm_default_config
