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

MODEL_NAME = "ANN"

# Model-specific constants and algorithms extracted from the original implementation.
ANN_MODEL_NAME = "ANN"


ANN_APP_TITLE = "Oil & Gas ANN Studio"


ANN_APP_SUBTITLE = "MATLAB-style workflow for tabular oil, gas, drilling, reservoir, and production data"


ANN_APP_DIR = Path(".oil_gas_ann_studio")


ANN_PROJECTS_DIR = ANN_APP_DIR / "projects"


ANN_PROJECTS_DIR.mkdir(parents=True, exist_ok=True)


_ANN_TF = None


_ANN_PLT = None


def ann_get_tf():
    global _ANN_TF
    if _ANN_TF is None:
        import tensorflow as tf
        _ANN_TF = tf
    return _ANN_TF


def ann_get_plt():
    global _ANN_PLT
    if _ANN_PLT is None:
        import matplotlib.pyplot as plt
        plt.rcParams["figure.dpi"] = 125
        plt.rcParams["savefig.dpi"] = 125
        plt.rcParams["axes.titlesize"] = 15
        plt.rcParams["axes.labelsize"] = 12.5
        plt.rcParams["xtick.labelsize"] = 10.5
        plt.rcParams["ytick.labelsize"] = 10.5
        plt.rcParams["legend.fontsize"] = 10.5
        _ANN_PLT = plt
    return _ANN_PLT


def ann_default_config() -> Dict[str, Any]:
    return {
        "project_name": "oil_gas_ann_project",
        "task_mode": "Auto Detect",
        "target_column": None,
        "feature_columns": [],
        "drop_columns": [],
        "auto_datetime": True,
        "drop_duplicates": True,
        "shuffle_data": True,
        "test_size": 0.20,
        "random_seed": 42,
        "hidden_layers": "256,128,64",
        "activation": "relu",
        "dropout": 0.20,
        "batch_norm": True,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 120,
        "batch_size": 32,
        "validation_split": 0.20,
        "early_stopping": True,
        "patience": 15,
        "use_class_weights": True,
        "threshold": 0.50,
        "l2_reg": 0.0,
    }


def ann_set_seed(seed: int):
    np.random.seed(seed)
    try:
        tf = ann_get_tf()
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def ann_parse_hidden_layers(text: str) -> List[int]:
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            v = int(item)
            if v > 0:
                values.append(v)
        except Exception:
            continue
    return values or [128, 64]


def ann_infer_task(y: pd.Series) -> str:
    if y is None:
        return "regression"
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    y_nonnull = y.dropna()
    if y_nonnull.empty:
        return "regression"
    unique_count = y_nonnull.nunique()
    ratio = unique_count / max(len(y_nonnull), 1)
    integer_like = np.allclose(y_nonnull, np.round(y_nonnull), equal_nan=True)
    if integer_like and unique_count <= 20 and ratio <= 0.2:
        return "classification"
    return "regression"


def ann_mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.nanmean(mape))


def ann_build_model(input_dim: int, task: str, config: Dict[str, Any], n_classes: int = 1):
    tf = ann_get_tf()
    layers = ann_parse_hidden_layers(config["hidden_layers"])
    reg = tf.keras.regularizers.l2(float(config["l2_reg"])) if float(config["l2_reg"]) > 0 else None

    model = tf.keras.Sequential(name="oil_gas_ann")
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for units in layers:
        model.add(tf.keras.layers.Dense(units, activation=None, kernel_regularizer=reg))
        if config.get("batch_norm", True):
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(config["activation"]))
        if float(config["dropout"]) > 0:
            model.add(tf.keras.layers.Dropout(float(config["dropout"])))

    if task == "classification":
        if n_classes == 2:
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ]
        else:
            model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
            loss = "sparse_categorical_crossentropy"
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ]
    else:
        model.add(tf.keras.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = [
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
        ]

    opt_name = config["optimizer"].lower()
    lr = float(config["learning_rate"])
    if opt_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif opt_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


# Public architecture aliases requested by the layered design.
build_ann_model = ann_build_model
infer_ann_task = ann_infer_task
get_ann_default_config = ann_default_config
