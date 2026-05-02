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

from models.ann import *
from models.cnn import *
from models.lstm import *
from services.preprocessing import *
from services.evaluation import *

def ann_predict_with_pipeline(model, prepared: Dict[str, Any], input_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    feature_cols = prepared["feature_columns_original"]
    raw = ann_ensure_feature_frame(input_df, feature_cols)
    if prepared["datetime_expanded_columns"]:
        raw, _ = ann_expand_datetime_columns(raw)

    # Align to training-time columns after datetime expansion
    required_after_dt = prepared["feature_columns_after_datetime"]
    raw = ann_ensure_feature_frame(raw, required_after_dt)

    X = prepared["preprocessor"].transform(raw)
    X = np.asarray(X).astype("float32")

    task = prepared["task"]
    out = input_df.copy()

    if task == "classification":
        class_names = prepared["class_names"]
        proba = model.predict(X, verbose=0)
        if len(class_names) == 2:
            score = proba.reshape(-1)
            pred = (score >= threshold).astype(int)
            labels = [class_names[i] for i in pred]
            out["prediction"] = labels
            out["prediction_score_positive"] = score
            out[f"prob_{class_names[0]}"] = 1 - score
            out[f"prob_{class_names[1]}"] = score
        else:
            pred = np.argmax(proba, axis=1)
            labels = [class_names[i] for i in pred]
            out["prediction"] = labels
            for i, cls in enumerate(class_names):
                out[f"prob_{cls}"] = proba[:, i]
    else:
        pred = model.predict(X, verbose=0).reshape(-1)
        out["prediction"] = pred

    return out


def cnn_gradcam_heatmap(model, img_array, layer_name=None):
    tf = cnn_get_tf()
    if layer_name is None:
        layer_name = cnn_find_last_conv_layer_name(model)
    if layer_name is None:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def cnn_overlay_heatmap_on_image(base_img: Image.Image, heatmap, alpha=0.40):
    if heatmap is None:
        return None
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]
    colored = Image.fromarray(np.uint8(colored * 255)).resize(base_img.size)
    base = base_img.convert("RGB")
    return Image.blend(base, colored, alpha=alpha)


def cnn_extract_features(model, paths, cfg, batch_limit=256):
    tf = cnn_get_tf()
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("gap").output)
    labels_dummy = np.zeros(len(paths), dtype=np.int32)
    ds = cnn_make_tf_dataset(paths[:batch_limit], labels_dummy[:batch_limit], cfg, training=False)
    feats = feature_model.predict(ds, verbose=0)
    return feats


def lstm_prepare_prediction_input(df_new: pd.DataFrame, processed: Dict, cfg: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
    feature_cols = processed["feature_cols"]
    date_col = cfg["date_col"]
    work = df_new.copy()

    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    missing = [c for c in feature_cols if c not in work.columns]
    if missing:
        raise ValueError(f"Prediction data is missing required feature columns: {missing}")

    numeric = lstm_coerce_numeric(work[feature_cols], feature_cols)
    numeric = lstm_apply_missing(numeric, cfg["missing_method"])

    if processed.get("task") == "regression":
        numeric = lstm_apply_transform(numeric, cfg["transform_mode"])
        numeric = lstm_apply_missing(numeric, cfg["missing_method"])
    else:
        numeric = lstm_apply_transform(numeric, cfg["transform_mode"])
        numeric = lstm_apply_missing(numeric, cfg["missing_method"])

    numeric = numeric.dropna().reset_index(drop=True)
    if len(numeric) < int(cfg["lookback"]):
        raise ValueError("Prediction data has fewer rows than the lookback window.")

    X_scaled = processed["feature_scaler"].transform(numeric[feature_cols])
    last_window = X_scaled[-int(cfg["lookback"]) :, :]
    return np.expand_dims(last_window.astype(np.float32), axis=0), numeric


def lstm_implied_prices_from_returns(last_prices: pd.Series, pred_values: np.ndarray, transform_mode: str, target_cols: List[str]) -> pd.DataFrame:
    pred_values = np.asarray(pred_values)
    horizon = pred_values.shape[0]
    out = []
    running = last_prices[target_cols].astype(float).copy()
    for h in range(horizon):
        step_vals = pred_values[h]
        if transform_mode == "pct_change":
            running = running * (1.0 + step_vals)
        elif transform_mode == "log_return":
            running = running * np.exp(step_vals)
        elif transform_mode == "raw":
            running = pd.Series(step_vals, index=target_cols)
        out.append(running.copy())
    df = pd.DataFrame(out)
    df.index = [f"step_{i+1}" for i in range(horizon)]
    return df


# Public workflow API
def predict_ann(*args, **kwargs):
    return ann_predict_with_pipeline(*args, **kwargs)

def predict_cnn(*args, **kwargs):
    return cnn_prepare_single_image(*args, **kwargs)

def predict_lstm(*args, **kwargs):
    return lstm_prepare_prediction_input(*args, **kwargs)
