# UI_test.py (Single-file: UI + ANN backend + MULTIVARIATE LSTM backend)
# ✅ SAFE LSTM + PROFESSIONAL VISUALIZE
# ✅ SPEED UPDATE: Lazy import sklearn + matplotlib + tensorflow (faster startup / faster page switching)

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Quiet TF noise (optional, safe)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ============================================================
# Performance: Lazy import caches
# ============================================================
_TF_CACHE: Dict[str, Any] = {}
_SK_CACHE: Dict[str, Any] = {}
_PLT_CACHE: Dict[str, Any] = {}


def _get_tf():
    """Lazy-load TensorFlow only when training/loading models is needed."""
    if "tf" in _TF_CACHE:
        return _TF_CACHE["tf"]

    import tensorflow as tf  # heavy
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping

    _TF_CACHE["tf"] = tf
    _TF_CACHE["layers"] = layers
    _TF_CACHE["Sequential"] = Sequential
    _TF_CACHE["EarlyStopping"] = EarlyStopping
    return tf


def _get_tf_keras():
    """Returns (tf, layers, Sequential, EarlyStopping) lazily."""
    tf = _get_tf()
    return tf, _TF_CACHE["layers"], _TF_CACHE["Sequential"], _TF_CACHE["EarlyStopping"]


def _get_sklearn():
    """Lazy-load scikit-learn only when training/evaluation uses it."""
    if _SK_CACHE.get("loaded"):
        return _SK_CACHE

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    _SK_CACHE.update(
        {
            "loaded": True,
            "train_test_split": train_test_split,
            "ColumnTransformer": ColumnTransformer,
            "OneHotEncoder": OneHotEncoder,
            "StandardScaler": StandardScaler,
            "LabelEncoder": LabelEncoder,
            "MinMaxScaler": MinMaxScaler,
            "Pipeline": Pipeline,
            "SimpleImputer": SimpleImputer,
            "accuracy_score": accuracy_score,
            "precision_score": precision_score,
            "recall_score": recall_score,
            "f1_score": f1_score,
            "confusion_matrix": confusion_matrix,
            "mean_absolute_error": mean_absolute_error,
            "mean_squared_error": mean_squared_error,
            "r2_score": r2_score,
        }
    )
    return _SK_CACHE


def _get_plt():
    """Lazy-load matplotlib only when Visualize page is rendered."""
    if "plt" in _PLT_CACHE:
        return _PLT_CACHE["plt"]
    import matplotlib.pyplot as plt  # heavy-ish
    _PLT_CACHE["plt"] = plt
    return plt


# ============================================================
# JSON-safe utilities (fix "not JSON serializable" forever)
# ============================================================
def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


# ============================================================
# Small utilities
# ============================================================
def _safe_index(options: List[Any], value: Any, default: int = 0) -> int:
    try:
        return options.index(value)
    except Exception:
        return default


def _clamp_int(x: Any, lo: int, hi: int, default: int) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


# ============================================================
# Navigation (Fixes the “double click” issue)
# ============================================================
def request_nav(page: str) -> None:
    st.session_state["_nav_to"] = page


def goto(page: str) -> None:
    st.query_params["page"] = page
    st.rerun()


def apply_pending_nav() -> None:
    nav_to = st.session_state.pop("_nav_to", None)
    if nav_to:
        goto(nav_to)


# ============================================================
# ANN BACKEND
# ============================================================
def infer_task(y: pd.Series) -> str:
    y_no_na = y.dropna()
    if y_no_na.empty:
        return "unknown"

    if not pd.api.types.is_numeric_dtype(y_no_na):
        return "classification"

    n = len(y_no_na)
    nunique = int(y_no_na.nunique(dropna=True))
    unique_ratio = nunique / max(1, n)

    is_integer_like = pd.api.types.is_integer_dtype(y_no_na) or np.allclose(
        y_no_na.astype(float).values, np.round(y_no_na.astype(float).values), atol=1e-9
    )

    if is_integer_like and nunique <= 20 and unique_ratio <= 0.2:
        return "classification"

    return "regression"


def _expand_datetime_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Convert datetime columns into numeric features to avoid one-hot explosion.
    - If a column is datetime dtype -> expand.
    - If an object column looks like datetime (parsable) -> expand.
    """
    X = X.copy()

    def add_parts(colname: str, dt: pd.Series):
        X[f"{colname}__year"] = dt.dt.year
        X[f"{colname}__month"] = dt.dt.month
        X[f"{colname}__day"] = dt.dt.day
        X[f"{colname}__dayofweek"] = dt.dt.dayofweek
        X[f"{colname}__hour"] = dt.dt.hour

    dt_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    for c in dt_cols:
        dt = pd.to_datetime(X[c], errors="coerce")
        add_parts(c, dt)
        X.drop(columns=[c], inplace=True)

    obj_cols = X.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        s = X[c]
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() >= 0.8:
            full = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
            add_parts(c, full)
            X.drop(columns=[c], inplace=True)

    return X


def _make_preprocessor(X: pd.DataFrame):
    sk = _get_sklearn()
    ColumnTransformer = sk["ColumnTransformer"]
    OneHotEncoder = sk["OneHotEncoder"]
    StandardScaler = sk["StandardScaler"]
    Pipeline = sk["Pipeline"]
    SimpleImputer = sk["SimpleImputer"]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def _map_activation(name: str) -> str:
    name = (name or "").strip().lower()
    if name in ["relu", "re-lu"]:
        return "relu"
    if name == "tanh":
        return "tanh"
    if name == "sigmoid":
        return "sigmoid"
    return "relu"


def _build_ann_model(
    input_dim: int,
    task: str,
    n_classes: int,
    ann_config: Optional[Dict[str, Any]],
    lr: float,
):
    tf, layers, _, _ = _get_tf_keras()

    ann_config = ann_config or {}
    hidden_layers = _clamp_int(ann_config.get("hidden_layers", 3), 1, 12, 3)

    neurons = ann_config.get("neurons", [256, 128, 64])
    if not isinstance(neurons, list):
        neurons = [256, 128, 64]
    neurons = (neurons + [64] * hidden_layers)[:hidden_layers]
    neurons = [max(1, int(n)) for n in neurons]

    act = _map_activation(ann_config.get("activation", "ReLU"))
    out_choice = (ann_config.get("output_activation", "Auto") or "Auto").strip().lower()

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for n in neurons:
        model.add(layers.Dense(n, activation=act))
        model.add(layers.Dropout(0.2))

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(lr))

    if task == "classification":
        if n_classes == 2:
            out_units = 1
            default_out_act = "sigmoid"
            loss = "binary_crossentropy"
        else:
            out_units = n_classes
            default_out_act = "softmax"
            loss = "sparse_categorical_crossentropy"

        out_act = default_out_act
        if out_choice in ["sigmoid", "softmax", "linear"]:
            out_act = out_choice

        model.add(layers.Dense(out_units, activation=out_act))
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    else:
        model.add(layers.Dense(1, activation="linear"))
        model.compile(optimizer=optimizer, loss="mse")

    return model


def train_ann_from_df(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    task_choice: str = "auto",
    test_size: float = 0.2,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
    early_stop: bool = True,
    patience: int = 5,
    seed: int = 42,
    ann_config: Optional[Dict[str, Any]] = None,
):
    sk = _get_sklearn()
    train_test_split = sk["train_test_split"]
    LabelEncoder = sk["LabelEncoder"]
    accuracy_score = sk["accuracy_score"]
    precision_score = sk["precision_score"]
    recall_score = sk["recall_score"]
    f1_score = sk["f1_score"]
    confusion_matrix = sk["confusion_matrix"]
    mean_absolute_error = sk["mean_absolute_error"]
    mean_squared_error = sk["mean_squared_error"]
    r2_score = sk["r2_score"]

    tf = _get_tf()

    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()

    if target_col not in df.columns:
        raise ValueError(f"Target column not found: {target_col}")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column not found: {c}")

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X = _expand_datetime_features(X)

    task = infer_task(y) if task_choice == "auto" else task_choice
    if task == "unknown":
        raise ValueError("Target column is empty after removing NA.")

    label_encoder: Optional[Any] = None

    if task == "classification":
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y.astype(str).fillna("NA"))
        n_classes = int(len(label_encoder.classes_))
        if n_classes < 2:
            raise ValueError("Classification requires at least 2 classes in target.")

        stratify = y_enc if n_classes >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify
        )
    else:
        y_enc = pd.to_numeric(y, errors="coerce").astype(float).values
        keep = ~np.isnan(y_enc)
        X = X.loc[keep].reset_index(drop=True)
        y_enc = y_enc[keep]
        if len(y_enc) < 2:
            raise ValueError("Regression has too few usable numeric rows after cleaning.")
        n_classes = 1

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=test_size, random_state=seed, shuffle=True
        )

    preprocessor, numeric_cols, categorical_cols = _make_preprocessor(X)

    X_train_np = preprocessor.fit_transform(X_train)
    X_test_np = preprocessor.transform(X_test)
    if X_train_np.shape[1] < 1:
        raise ValueError("After preprocessing, there are 0 usable input features.")

    model = _build_ann_model(
        input_dim=int(X_train_np.shape[1]),
        task=task,
        n_classes=int(n_classes),
        ann_config=ann_config,
        lr=float(lr),
    )

    callbacks = []
    if early_stop:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
        )

    history_obj = model.fit(
        X_train_np,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )
    history = history_obj.history

    results: Dict[str, Any] = {
        "task": task,
        "n_features_after_encoding": int(X_train_np.shape[1]),
        "numeric_features": int(len(numeric_cols)),
        "categorical_features": int(len(categorical_cols)),
    }

    if task == "classification":
        probs = model.predict(X_test_np, verbose=0)
        if n_classes == 2:
            y_pred = (probs.reshape(-1) > 0.5).astype(int)
        else:
            y_pred = np.argmax(probs, axis=1)

        results.update(
            {
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "f1_score": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
                "classes": int(n_classes),
                "class_labels": label_encoder.classes_.tolist() if label_encoder is not None else None,
            }
        )
    else:
        preds = model.predict(X_test_np, verbose=0).reshape(-1)
        results.update(
            {
                "mae": float(mean_absolute_error(y_test, preds)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
                "r2_score": float(r2_score(y_test, preds)) if len(y_test) >= 2 else 0.0,
            }
        )

    return model, preprocessor, label_encoder, results, {
        "loss": [float(x) for x in history.get("loss", [])],
        "val_loss": [float(x) for x in history.get("val_loss", [])],
    }


def predict_from_df(
    df_features: pd.DataFrame,
    model,
    preprocessor,
    task: str,
    label_encoder: Optional[Any] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    X = _expand_datetime_features(df_features)
    X_np = preprocessor.transform(X)

    if task == "classification":
        probs = model.predict(X_np, verbose=0)
        if probs.ndim == 2 and probs.shape[1] > 1:
            pred_idx = np.argmax(probs, axis=1)
            conf = np.max(probs, axis=1)
        else:
            p = probs.reshape(-1)
            pred_idx = (p > 0.5).astype(int)
            conf = p

        if label_encoder is not None:
            decoded = label_encoder.inverse_transform(pred_idx.astype(int))
            return decoded, conf
        return pred_idx, conf

    preds = model.predict(X_np, verbose=0).reshape(-1)
    return preds, None


# ============================================================
# MULTIVARIATE LSTM BACKEND (SAFE PATCH ✅ never crashes UI)
# ============================================================
def _clean_df_minimal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df


def _smart_split_sizes(n: int, test_size: float) -> Tuple[float, List[str]]:
    warns: List[str] = []
    test_size = float(test_size)
    test_size = min(max(test_size, 0.05), 0.5)

    if n < 2:
        return test_size, warns

    test_count = int(np.ceil(n * test_size))
    train_count = n - test_count
    if train_count < 1:
        test_count = n - 1
        test_size = test_count / n
        warns.append(f"Auto-adjusted test split because dataset is very small (n={n}).")
    return test_size, warns


@dataclass
class LSTMConfig:
    target_col: str
    feature_cols: List[str]
    date_col: Optional[str] = None

    lookback: int = 10
    horizon: int = 1
    test_size: float = 0.2

    lstm_units: int = 64
    dropout: float = 0.2
    epochs: int = 100
    batch_size: int = 32

    patience: int = 10
    val_split: float = 0.1
    seed: int = 42


def _set_seed(seed: int) -> None:
    tf = _get_tf()
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    if s.dtype == "object":
        s = s.astype(str).str.replace(r"[^\d.-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").astype(float)


def _prepare_multivariate_frame_safe(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    date_col: Optional[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    warns: List[str] = []

    df = _clean_df_minimal(df)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column '{c}' not found.")

    work = df.copy()

    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        before = len(work)
        work = work.dropna(subset=[date_col])
        if len(work) < before:
            warns.append("Dropped rows with invalid time values in the time column.")
        work = work.sort_values(date_col)

    X = work[feature_cols].copy()
    X = _expand_datetime_features(X)
    X = pd.get_dummies(X, dummy_na=True)

    y = _coerce_numeric_series(work[target_col])

    aligned = X.copy()
    aligned["_target_"] = y.values

    aligned = aligned.replace([np.inf, -np.inf], np.nan)
    aligned = aligned.ffill().bfill()
    aligned = aligned.dropna(axis=0, how="any")

    if len(aligned) < 10:
        raise ValueError(f"After cleaning, only {len(aligned)} usable rows remain. Need more data for LSTM.")

    input_cols = [c for c in aligned.columns if c != "_target_"] + ["_target_"]
    return aligned, input_cols, warns


def make_sequences_multivariate(arr: np.ndarray, lookback: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if len(arr) <= lookback + horizon - 1:
        raise ValueError("Not enough data for lookback+horizon.")

    X, y = [], []
    target_idx = arr.shape[1] - 1
    for i in range(lookback, len(arr) - horizon + 1):
        X.append(arr[i - lookback : i, :])
        y.append(arr[i + horizon - 1, target_idx])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_lstm_multivariate(lookback: int, n_channels: int, lstm_units: int, dropout: float):
    _, layers, Sequential, _ = _get_tf_keras()
    model = Sequential(
        [
            layers.Input(shape=(lookback, n_channels)),
            layers.LSTM(int(lstm_units), return_sequences=True),
            layers.Dropout(float(dropout)),
            layers.LSTM(int(lstm_units)),
            layers.Dropout(float(dropout)),
            layers.Dense(1),
        ]
    )
    tf = _get_tf()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_lstm_from_df(
    df: pd.DataFrame,
    cfg: LSTMConfig,
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Dict[str, list], Dict[str, Any], Dict[str, np.ndarray]]:
    sk = _get_sklearn()
    MinMaxScaler = sk["MinMaxScaler"]
    mean_squared_error = sk["mean_squared_error"]
    mean_absolute_error = sk["mean_absolute_error"]
    r2_score = sk["r2_score"]

    _, _, _, EarlyStopping = _get_tf_keras()

    metrics: Dict[str, Any] = {"status": "ok", "message": "", "warnings": []}
    history: Dict[str, list] = {"loss": [], "val_loss": []}
    outputs: Dict[str, np.ndarray] = {}

    try:
        _set_seed(int(cfg.seed))

        aligned, input_cols, warns = _prepare_multivariate_frame_safe(df, cfg.target_col, cfg.feature_cols, cfg.date_col)
        metrics["warnings"].extend(warns)

        Xy = aligned[[c for c in input_cols if c != "_target_"]].to_numpy(dtype=np.float32)
        y = aligned["_target_"].to_numpy(dtype=np.float32).reshape(-1, 1)
        full = np.concatenate([Xy, y], axis=1)

        n = len(full)
        test_size_adj, warns2 = _smart_split_sizes(n, cfg.test_size)
        metrics["warnings"].extend(warns2)

        split_idx = int(n * (1 - test_size_adj))
        split_idx = max(2, min(split_idx, n - 1))

        train_full = full[:split_idx]
        test_full = full[split_idx:]

        scaler_all = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))

        train_scaled = scaler_all.fit_transform(train_full)
        test_scaled = scaler_all.transform(test_full)

        scaler_y.fit(train_full[:, [-1]])
        scaled_full = np.vstack([train_scaled, test_scaled]).astype(np.float32)

        lookback = int(cfg.lookback)
        horizon = int(cfg.horizon)

        if len(scaled_full) <= lookback + horizon:
            lookback = max(1, len(scaled_full) - horizon - 1)
            metrics["warnings"].append("Auto-reduced lookback due to limited data length.")

        X_seq, y_seq = make_sequences_multivariate(scaled_full, lookback=lookback, horizon=horizon)

        if len(X_seq) < 2:
            return None, None, history, {
                "status": "error",
                "message": "Not enough sequences after preprocessing for LSTM training.",
                "warnings": metrics.get("warnings", []),
            }, outputs

        n_seq = len(X_seq)
        test_size_adj2, warns3 = _smart_split_sizes(n_seq, test_size_adj)
        metrics["warnings"].extend(warns3)

        seq_split = int(n_seq * (1 - test_size_adj2))
        seq_split = max(1, min(seq_split, n_seq - 1))

        X_train, y_train = X_seq[:seq_split], y_seq[:seq_split]
        X_test, y_test = X_seq[seq_split:], y_seq[seq_split:]

        model = build_lstm_multivariate(
            lookback=lookback,
            n_channels=int(X_train.shape[2]),
            lstm_units=int(cfg.lstm_units),
            dropout=float(cfg.dropout),
        )

        cb = EarlyStopping(monitor="val_loss", patience=int(cfg.patience), restore_best_weights=True)
        hist = model.fit(
            X_train,
            y_train,
            epochs=int(cfg.epochs),
            batch_size=int(cfg.batch_size),
            validation_split=float(cfg.val_split),
            callbacks=[cb],
            verbose=0,
        )
        history = {k: [float(x) for x in v] for k, v in hist.history.items()}

        train_pred = model.predict(X_train, verbose=0).reshape(-1, 1)
        test_pred = model.predict(X_test, verbose=0).reshape(-1, 1)

        y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
        train_pred_actual = scaler_y.inverse_transform(train_pred)
        test_pred_actual = scaler_y.inverse_transform(test_pred)

        y_test_flat = y_test_actual.reshape(-1)
        pred_flat = test_pred_actual.reshape(-1)
        mask = np.isfinite(y_test_flat) & np.isfinite(pred_flat)

        rmse = float(np.sqrt(mean_squared_error(y_test_flat[mask], pred_flat[mask]))) if mask.sum() else 0.0
        mae = float(mean_absolute_error(y_test_flat[mask], pred_flat[mask])) if mask.sum() else 0.0
        try:
            r2 = float(r2_score(y_test_flat[mask], pred_flat[mask])) if mask.sum() >= 2 else 0.0
        except Exception:
            r2 = 0.0

        metrics.update(
            {
                "status": "ok",
                "task": "regression",
                "rmse": rmse,
                "mae": mae,
                "r2_score": r2,
                "lookback_used": int(lookback),
                "horizon_used": int(horizon),
                "channels_used": int(X_train.shape[2]),
                "rows_used": int(len(aligned)),
                "train_sequences": int(len(X_train)),
                "test_sequences": int(len(X_test)),
            }
        )

        pack = {
            "scaler_all": scaler_all,
            "scaler_y": scaler_y,
            "input_feature_columns": [c for c in input_cols if c != "_target_"],
            "target_column": cfg.target_col,
        }

        last_feature_scaled = scaled_full[-1, :-1].astype(np.float32)

        outputs = {
            "scaled_full": scaled_full.astype(np.float32),
            "lookback": np.array([lookback], dtype=np.int32),
            "horizon": np.array([horizon], dtype=np.int32),
            "last_feature_scaled": last_feature_scaled.astype(np.float32),
            "y_train_actual": y_train_actual.astype(np.float32),
            "y_test_actual": y_test_actual.astype(np.float32),
            "train_pred_actual": train_pred_actual.astype(np.float32),
            "test_pred_actual": test_pred_actual.astype(np.float32),
        }

        return model, pack, history, metrics, outputs

    except Exception as e:
        return None, None, history, {
            "status": "error",
            "message": f"LSTM training failed: {e}",
            "warnings": metrics.get("warnings", []),
        }, outputs


def forecast_future_multivariate(
    model,
    scaler_y,
    scaled_full: np.ndarray,
    lookback: int,
    n_steps: int,
    last_feature_scaled: np.ndarray,
) -> np.ndarray:
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if len(scaled_full) <= lookback:
        raise ValueError("Not enough history to forecast. Increase data or reduce lookback.")

    n_channels = scaled_full.shape[1]
    curr = scaled_full[-lookback:, :].reshape(1, lookback, n_channels).astype(np.float32)

    preds_scaled = []
    for _ in range(n_steps):
        p = model.predict(curr, verbose=0).reshape(-1)[0]
        preds_scaled.append(p)

        next_row = np.concatenate([last_feature_scaled, np.array([p], dtype=np.float32)], axis=0).reshape(1, 1, n_channels)
        curr = np.concatenate([curr[:, 1:, :], next_row], axis=1)

    preds_scaled = np.array(preds_scaled, dtype=np.float32).reshape(-1, 1)
    preds_inv = scaler_y.inverse_transform(preds_scaled)
    return preds_inv


# ============================================================
# Storage (projects persistence)
# ============================================================
DATA_DIR = Path(".neural_studio")
DATA_DIR.mkdir(exist_ok=True)
PROJECTS_FILE = DATA_DIR / "projects.json"
CURRENT_FILE = DATA_DIR / "current_project.json"


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj):
    safe = _json_safe(obj)
    path.write_text(json.dumps(safe, indent=2, ensure_ascii=False), encoding="utf-8")


def get_projects() -> List[Dict[str, Any]]:
    return _read_json(PROJECTS_FILE, [])


def save_projects(projects: List[Dict[str, Any]]) -> None:
    _write_json(PROJECTS_FILE, projects)


def get_current_project() -> Optional[Dict[str, Any]]:
    return _read_json(CURRENT_FILE, None)


def set_current_project(project: Dict[str, Any]) -> None:
    _write_json(CURRENT_FILE, project)


def upsert_project(project: Dict[str, Any]) -> None:
    projects = get_projects()
    idx = next((i for i, p in enumerate(projects) if p.get("id") == project.get("id")), None)
    if idx is None:
        projects.insert(0, project)
    else:
        projects[idx] = project
    save_projects(projects)
    set_current_project(project)


# ============================================================
# UI helpers (styling)
# ============================================================
def inject_css():
    st.markdown(
        """
        <style>
          .block-container { padding-top: 1rem; padding-bottom: 6rem; max-width: 1200px; }
          @media (min-width: 1024px) { .block-container { padding-bottom: 2rem; } }

          .hero {
            border-radius: 18px;
            padding: 28px;
            color: white;
            background: linear-gradient(135deg, #0f172a 0%, #312e81 45%, #0891b2 100%);
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.10);
          }
          .hero .blur1 {
            position:absolute; top:-90px; right:-90px;
            width:300px; height:300px; border-radius:999px;
            background: rgba(34, 211, 238, 0.18);
            filter: blur(40px);
          }
          .hero .blur2 {
            position:absolute; bottom:-90px; left:-90px;
            width:300px; height:300px; border-radius:999px;
            background: rgba(99, 102, 241, 0.18);
            filter: blur(40px);
          }
          .hero h1 { margin: 0.25rem 0 0.25rem 0; font-size: 2.0rem; }
          @media (min-width: 768px){ .hero h1{ font-size: 2.8rem; } }

          .ns-card {
            border-radius: 16px;
            padding: 18px;
            border: 1px solid #e2e8f0;
            background: white;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
          }
          .muted { color: #64748b; }
          .badge {
            display:inline-block; padding: 4px 10px; border-radius:999px;
            font-size: 12px; border: 1px solid #e2e8f0; background: #f8fafc; color:#0f172a;
          }
          .badge.good { background: #ecfdf5; border-color:#a7f3d0; color:#065f46; }
          .badge.warn { background: #fffbeb; border-color:#fde68a; color:#92400e; }

          /* Force "same-line" action buttons to look equal */
          .btn-row > div { width:100%; }
          .btn-row button { width:100% !important; height: 48px !important; border-radius: 14px !important; }

          .btn-grad > button {
            background: linear-gradient(90deg, #06b6d4 0%, #6366f1 100%) !important;
            color: white !important;
            border: 0 !important;
          }
          .btn-emerald > button {
            background: linear-gradient(90deg, #10b981 0%, #14b8a6 100%) !important;
            color: white !important;
            border: 0 !important;
          }

          .bottom-nav {
            position: fixed;
            left: 0; right: 0; bottom: 0;
            background: rgba(255,255,255,0.92);
            border-top: 1px solid #e2e8f0;
            padding: 10px 12px;
            z-index: 9999;
          }
          @media (min-width: 1024px){ .bottom-nav { display: none; } }
          .bottom-nav .row {
            max-width: 1200px; margin: 0 auto;
            display: grid; grid-template-columns: repeat(9, 1fr);
            gap: 6px;
          }
          .bottom-nav a {
            text-decoration: none; color: #64748b;
            font-size: 11px; text-align:center;
            padding: 8px 6px; border-radius: 14px;
            border: 1px solid transparent;
            white-space: nowrap;
          }
          .bottom-nav a.active {
            color: #4f46e5;
            border-color: #c7d2fe;
            background: #eef2ff;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def status_bar(current: str, processing: bool = False):
    steps = ["data_loaded", "preprocessed", "configured", "trained", "evaluated"]
    labels = {
        "data_loaded": "Data Loaded",
        "preprocessed": "Preprocessed",
        "configured": "Model Configured",
        "trained": "Trained",
        "evaluated": "Evaluated",
    }
    current_idx = steps.index(current) if current in steps else -1

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    cols = st.columns(5)
    for i, s in enumerate(steps):
        done = i <= current_idx
        with cols[i]:
            st.markdown(
                f"""
                <div style="display:flex; gap:10px; align-items:center;">
                  <div style="
                    width:32px;height:32px;border-radius:999px;
                    display:flex;align-items:center;justify-content:center;
                    font-weight:700; font-size:14px;
                    color:{'white' if done else '#64748b'};
                    background:{'linear-gradient(90deg,#06b6d4,#6366f1)' if done else '#f1f5f9'};
                    border:1px solid {'rgba(99,102,241,0.35)' if done else '#e2e8f0'};
                  ">{i + 1}</div>
                  <div style="font-size:13px; font-weight:600; color:{'#0f172a' if done else '#64748b'};">
                    {labels[s]}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    if processing:
        st.caption("Processing…")
    st.markdown("</div>", unsafe_allow_html=True)


def bottom_nav(active: str):
    items = [
        ("home", "Home"),
        ("data", "Data"),
        ("preprocess", "Preprocess"),
        ("model", "Model"),
        ("train", "Train"),
        ("evaluate", "Evaluate"),
        ("predict", "Predict"),
        ("visualize", "Visualize"),
        ("save", "Save"),
    ]
    links = []
    for key, label in items:
        cls = "active" if key == active else ""
        links.append(f'<a class="{cls}" href="?page={key}">{label}</a>')
    st.markdown(
        f"""
        <div class="bottom-nav">
          <div class="row">
            {''.join(links)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def new_project() -> Dict[str, Any]:
    p = {
        "id": str(uuid.uuid4()),
        "name": f"Project {time.strftime('%Y-%m-%d')}",
        "task_type": "classification",
        "model_type": "ann",
        "status": "data_loaded",
        "dataset": {"filename": "—", "rows": 0, "cols": 0, "missing": 0, "path": None, "file_type": None, "sheet": None},
        "columns": {"target": None, "time": None, "features": []},
        "preprocess": {
            "missing_strategy": "Drop rows",
            "split": 0.8,
            "seed": 42,
            "lookback": 20,
            "horizon": 1,
        },
        "train_config": {"epochs": 20, "batch_size": 32, "lr": 0.001, "early_stop": True, "patience": 5},
        "train_logs": [],
        "history": {"loss": [], "val_loss": []},
        "evaluation_metrics": None,
        "artifacts": None,
        "feature_meta": {},
        "target_profile": {},
        "viz_cache": {},
        "ann_config": {"hidden_layers": 3, "neurons": [256, 128, 64], "activation": "ReLU", "output_activation": "Auto"},
        "lstm_config": {"units": 64, "layers": 2, "dropout": 0.2, "bidirectional": False},
    }
    upsert_project(p)
    return p


def ensure_current_project() -> Optional[Dict[str, Any]]:
    return get_current_project()


def project_badge(p: Dict[str, Any]) -> str:
    model = p.get("model_type", "—").upper()
    task = p.get("task_type", "—")
    return f"{model} • {task}"


# ============================================================
# Caching
# ============================================================
@st.cache_data(show_spinner=False)
def cached_load_dataset(path: str, file_type: str, sheet: Optional[str]):
    if file_type == "csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)


@st.cache_resource(show_spinner=False)
def cached_load_ann_artifacts(model_path: str, preprocessor_path: str, label_encoder_path: str):
    tf = _get_tf()
    model = tf.keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)
    return model, preprocessor, label_encoder


@st.cache_resource(show_spinner=False)
def cached_load_lstm_artifacts(model_path: str, pack_path: str, outputs_path: str):
    tf = _get_tf()
    model = tf.keras.models.load_model(model_path)
    pack = joblib.load(pack_path)
    outputs = np.load(outputs_path, allow_pickle=False)
    return model, pack, outputs


@st.cache_data(show_spinner=False)
def cached_feature_meta(dataset_path: str, file_type: str, sheet: Optional[str], features: Tuple[str, ...]) -> Dict[str, Any]:
    if file_type == "csv":
        df_full = pd.read_csv(dataset_path)
    else:
        df_full = pd.read_excel(dataset_path, sheet_name=sheet)
    return build_feature_meta(df_full, list(features))


# ============================================================
# Data loading helpers
# ============================================================
def load_project_dataset(p: Dict[str, Any]) -> pd.DataFrame:
    ds = p.get("dataset", {})
    path = ds.get("path")
    if not path:
        raise FileNotFoundError("Dataset file not saved. Upload again in Data page.")
    return cached_load_dataset(path, ds.get("file_type"), ds.get("sheet"))


def build_feature_meta(df: pd.DataFrame, features: List[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for f in features:
        if f not in df.columns:
            continue
        s = df[f]
        is_num = pd.api.types.is_numeric_dtype(s)
        meta[f] = {"type": "numeric" if is_num else "categorical"}
        if not is_num:
            vals = s.dropna().astype(str).unique().tolist()
            meta[f]["options"] = vals[:50]
    return meta


# ============================================================
# ✅ Target profile helper (Professional prediction explanation)
# ============================================================
def build_target_profile(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    if target_col not in df.columns:
        return {}

    y = df[target_col].copy()

    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.notna().sum() >= max(5, int(0.2 * len(y))):
        yv = y_num.dropna().astype(float)
        return {
            "type": "regression",
            "count": int(len(yv)),
            "min": float(np.nanmin(yv.values)) if len(yv) else None,
            "max": float(np.nanmax(yv.values)) if len(yv) else None,
            "mean": float(np.nanmean(yv.values)) if len(yv) else None,
            "std": float(np.nanstd(yv.values)) if len(yv) else None,
        }

    ys = y.astype(str).fillna("NA")
    vc = ys.value_counts().head(20)
    return {
        "type": "classification",
        "count": int(len(ys)),
        "classes_top": vc.index.tolist(),
        "counts_top": vc.values.astype(int).tolist(),
    }


# ============================================================
# Pages
# ============================================================
def page_home():
    st.markdown(
        """
        <div class="hero">
          <div class="blur1"></div>
          <div class="blur2"></div>
          <div style="position:relative;">
            <div style="font-weight:600; color: rgba(255,255,255,0.85);">✨ Neural Studio</div>
            <h1>Build, Train, Evaluate & Deploy</h1>
            <div style="max-width: 800px; color: rgba(255,255,255,0.80); font-size: 15px;">
              A guided, mobile-first machine learning workflow UI — from upload to predictions — with a premium modern design.
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
        st.button("➕ New Project", use_container_width=True, on_click=lambda: (new_project(), request_nav("data")))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-row">', unsafe_allow_html=True)
        if st.button("📂 Open Current Project", use_container_width=True):
            if ensure_current_project() is None:
                st.warning("No current project found. Create one first.")
            else:
                request_nav("data")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.subheader("Features")
    feats = [
        ("📤", "Upload Data", "CSV/Excel upload with preview & column selection"),
        ("🧹", "Preprocess", "Split settings are used in training"),
        ("🧠", "Design Model", "ANN + Multivariate LSTM configuration"),
        ("🏋️", "Train", "Real training + saved model artifacts"),
        ("✅", "Evaluate", "Real metrics + labeled confusion matrix (ANN classification)"),
        ("🔮", "Predict", "ANN manual/batch • LSTM future forecast (features held constant if unknown)"),
        ("📊", "Visualize", "Training curves + residuals + actual vs predicted"),
        ("💾", "Save/Load", "Download & restore project JSON"),
        ("⚡", "Performance", "Lazy-loaded TF/sklearn/matplotlib for faster startup"),
    ]
    cols = st.columns(4)
    for i, (ico, title, desc) in enumerate(feats):
        with cols[i % 4]:
            st.markdown(
                f"""
                <div class="ns-card" style="min-height: 140px;">
                  <div style="font-size:28px; margin-bottom:6px;">{ico}</div>
                  <div style="font-weight:800; font-size:16px;">{title}</div>
                  <div class="muted" style="margin-top:4px; font-size:13px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("Recent Projects")
    projects = get_projects()
    if not projects:
        st.info("No saved projects yet.")
    else:
        grid = st.columns(3)
        for i, p in enumerate(projects[:6]):
            with grid[i % 3]:
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.markdown(f"**{p.get('name', 'Untitled')}**")
                st.caption(project_badge(p))
                ds = p.get("dataset", {})
                st.write(f"Dataset: {ds.get('filename', '—')}")
                st.write(f"Rows: {ds.get('rows', 0)} • Cols: {ds.get('cols', 0)} • Missing: {ds.get('missing', 0)}")
                if st.button("Open Project", key=f"open_{p.get('id')}"):
                    set_current_project(p)
                    request_nav("data")
                st.markdown("</div>", unsafe_allow_html=True)


def page_data():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project. Go to Home and create a new project.")
        return

    st.title("Data Upload")
    status_bar(p.get("status", "data_loaded"))

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Upload Dataset (CSV or Excel)")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    df = None
    sheet_name = None
    file_type = None

    if uploaded is not None:
        name = uploaded.name.lower()
        try:
            if name.endswith(".csv"):
                df = pd.read_csv(uploaded)
                file_type = "csv"
            else:
                xls = pd.ExcelFile(uploaded)
                sheet_name = st.selectbox("Select sheet", xls.sheet_names, index=0)
                df = pd.read_excel(uploaded, sheet_name=sheet_name)
                file_type = "excel"
        except Exception as e:
            st.error(f"Could not read file: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            return

    st.markdown("</div>", unsafe_allow_html=True)

    if df is not None:
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
        missing = int(df.isna().sum().sum())

        DATASETS_DIR = DATA_DIR / "datasets"
        DATASETS_DIR.mkdir(exist_ok=True)

        suffix = ".csv" if file_type == "csv" else ".xlsx"
        dataset_path = DATASETS_DIR / f"{p['id']}{suffix}"
        with open(dataset_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Clear only dataset cache (safe)
        try:
            cached_load_dataset.clear()
        except Exception:
            pass

        p["dataset"] = {
            "filename": uploaded.name,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing": missing,
            "path": str(dataset_path),
            "file_type": file_type,
            "sheet": sheet_name,
        }
        p["raw_preview"] = df.head(10).to_dict(orient="records")
        p["raw_columns"] = df.columns.tolist()
        p["status"] = "data_loaded"
        p["evaluation_metrics"] = None
        p["artifacts"] = None
        p["history"] = {"loss": [], "val_loss": []}
        p.setdefault("viz_cache", {})
        p["viz_cache"].pop("ann_regression", None)
        p["target_profile"] = {}

        p["columns"]["target"] = None
        p["columns"]["time"] = None
        p["columns"]["features"] = []
        p["feature_meta"] = {}

        upsert_project(p)

    ds = p.get("dataset", {})
    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Dataset Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filename", ds.get("filename", "—"))
    c2.metric("Rows", ds.get("rows", 0))
    c3.metric("Columns", ds.get("cols", 0))
    mv = ds.get("missing", 0)
    c4.metric("Missing Values", mv)
    if mv and mv > 0:
        st.warning("Missing values detected — handled automatically during training.")
    st.markdown("</div>", unsafe_allow_html=True)

    cols = p.get("raw_columns", [])
    if cols:
        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Column Configuration")

        saved_target = p.get("columns", {}).get("target")
        target = st.selectbox("Target Column (what to predict)", cols, index=_safe_index(cols, saved_target, 0))

        time_options = ["(None)"] + [c for c in cols if c != target]
        saved_time = p.get("columns", {}).get("time")
        time_default = "(None)" if (saved_time is None or saved_time == target or saved_time not in time_options) else saved_time
        time_col = st.selectbox("Time Column (optional, for time series)", time_options, index=_safe_index(time_options, time_default, 0))

        feature_options = [c for c in cols if c != target]
        saved_features = p.get("columns", {}).get("features", []) or []
        safe_defaults = [c for c in saved_features if c in feature_options]
        if not safe_defaults:
            safe_defaults = feature_options[: min(6, len(feature_options))]

        features = st.multiselect("Feature Columns (inputs)", feature_options, default=safe_defaults)

        p["columns"]["target"] = target
        p["columns"]["time"] = None if time_col == "(None)" else time_col
        p["columns"]["features"] = [c for c in features if c != target]

        # Feature meta only when features selected (cached)
        try:
            ds_path = p.get("dataset", {}).get("path")
            ds_type = p.get("dataset", {}).get("file_type")
            ds_sheet = p.get("dataset", {}).get("sheet")
            if ds_path and p["columns"]["features"]:
                p["feature_meta"] = cached_feature_meta(ds_path, ds_type, ds_sheet, tuple(p["columns"]["features"]))
            else:
                p["feature_meta"] = {}
        except Exception:
            p["feature_meta"] = {}

        upsert_project(p)

        st.caption(f"Selected features: {len(p['columns']['features'])}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Data Preview")

        if "data_preview_rows" not in st.session_state:
            st.session_state["data_preview_rows"] = 10

        n_rows = st.number_input(
            "Number of uploaded rows shown",
            min_value=1,
            step=1,
            value=int(st.session_state["data_preview_rows"]),
            key="data_preview_rows",
        )

        # Load only here (not on every page)
        try:
            df_full = load_project_dataset(p)
            st.dataframe(df_full.head(int(n_rows)), use_container_width=True, height=360)
        except Exception:
            preview = p.get("raw_preview", [])
            if preview:
                st.dataframe(pd.DataFrame(preview).head(int(n_rows)), use_container_width=True, height=360)
            else:
                st.info("Upload a dataset to preview rows.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.write("")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="btn-row">', unsafe_allow_html=True)
            st.button("⬅ Back to Home", use_container_width=True, on_click=request_nav, args=("home",))
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
            st.button("Continue to Preprocess ➜", use_container_width=True, on_click=request_nav, args=("preprocess",))
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a CSV/Excel to enable column selection and preview.")


def page_preprocess():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Preprocess")
    status_bar(p.get("status", "data_loaded"))

    ds = p.get("dataset", {})
    cols_cfg = p.get("columns", {})

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Dataset Summary")
    st.write(f"**Project:** {p.get('name')}")
    st.write(f"**Rows:** {ds.get('rows', 0)} • **Features:** {len(cols_cfg.get('features', []))} • **Target:** {cols_cfg.get('target', '—')}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Missing Values")
    p["preprocess"]["missing_strategy"] = st.selectbox(
        "Strategy",
        ["Drop rows", "Fill with mean/median", "Forward fill (time-series)"],
        index=_safe_index(["Drop rows", "Fill with mean/median", "Forward fill (time-series)"], p["preprocess"].get("missing_strategy"), 0),
    )
    st.caption("ANN uses imputers automatically; LSTM cleans & forward/back fills after encoding.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Train/Test Split")
    split = st.slider("Train %", 50, 95, int(float(p["preprocess"].get("split", 0.8)) * 100))
    p["preprocess"]["split"] = split / 100.0
    p["preprocess"]["seed"] = st.number_input("Random Seed", value=int(p["preprocess"].get("seed", 42)), step=1)
    st.caption(f"Train: {split}% • Test: {100 - split}%")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#c4b5fd; background:#faf5ff;">', unsafe_allow_html=True)
    st.write("### LSTM / Time Series Settings")
    p["preprocess"]["lookback"] = st.number_input("Lookback window", value=int(p["preprocess"].get("lookback", 20)), min_value=1, step=1)
    p["preprocess"]["horizon"] = st.number_input("Forecast horizon (steps ahead)", value=int(p["preprocess"].get("horizon", 1)), min_value=1, step=1)
    st.caption("Multivariate LSTM uses past lookback rows of (features + target) to predict target at horizon.")
    st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="btn-row">', unsafe_allow_html=True)
        st.button("⬅ Back to Data", use_container_width=True, on_click=request_nav, args=("data",))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
        st.button(
            "Continue to Model Configuration ➜",
            use_container_width=True,
            on_click=lambda: (p.__setitem__("status", "preprocessed"), upsert_project(p), request_nav("model")),
        )
        st.markdown("</div>", unsafe_allow_html=True)


def page_model():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Model")
    status_bar(p.get("status", "data_loaded"))

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Model Type")
    model_type = st.radio("Choose", ["ann", "lstm"], index=0 if p.get("model_type") == "ann" else 1, horizontal=True)
    p["model_type"] = model_type
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Task Type")
    task_type = st.selectbox(
        "Task",
        ["auto-detect", "classification", "regression"],
        index=_safe_index(["auto-detect", "classification", "regression"], p.get("task_type", "classification"), 1),
    )
    p["task_type"] = task_type
    st.caption("For LSTM, task is always regression (time-series forecasting).")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if model_type == "ann":
        st.markdown('<div class="ns-card" style="background:#faf5ff; border-color:#ddd6fe;">', unsafe_allow_html=True)
        st.write("### ANN Configuration (Applied for real ✅)")
        p.setdefault("ann_config", {})
        p["ann_config"]["hidden_layers"] = st.number_input("Hidden layers", min_value=1, max_value=12, value=int(p["ann_config"].get("hidden_layers", 3)))
        hl = int(p["ann_config"]["hidden_layers"])
        neurons = p["ann_config"].get("neurons", [256, 128, 64])
        if not isinstance(neurons, list):
            neurons = [256, 128, 64]
        neurons = (neurons + [64] * hl)[:hl]
        new_neurons = []
        for i in range(hl):
            new_neurons.append(int(st.number_input(f"Neurons in layer {i + 1}", min_value=1, max_value=2048, value=int(neurons[i]))))
        p["ann_config"]["neurons"] = new_neurons
        p["ann_config"]["activation"] = st.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"], index=_safe_index(["ReLU", "Tanh", "Sigmoid"], p["ann_config"].get("activation", "ReLU"), 0))
        p["ann_config"]["output_activation"] = st.selectbox("Output activation", ["Auto", "Linear", "Sigmoid", "Softmax"], index=_safe_index(["Auto", "Linear", "Sigmoid", "Softmax"], p["ann_config"].get("output_activation", "Auto"), 0))
        st.caption("These settings are used during ANN training.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="ns-card" style="background:#ecfeff; border-color:#a5f3fc;">', unsafe_allow_html=True)
        st.write("### LSTM Configuration (Multivariate ✅)")
        p.setdefault("lstm_config", {})
        p["lstm_config"]["units"] = st.number_input("LSTM units", min_value=1, max_value=2048, value=int(p["lstm_config"].get("units", 64)))
        p["lstm_config"]["layers"] = st.number_input("Number of LSTM layers (stored)", min_value=1, max_value=6, value=int(p["lstm_config"].get("layers", 2)))
        p["lstm_config"]["dropout"] = st.slider("Dropout rate", 0.0, 0.8, float(p["lstm_config"].get("dropout", 0.2)))
        p["lstm_config"]["bidirectional"] = st.checkbox("Bidirectional (stored)", value=bool(p["lstm_config"].get("bidirectional", False)))
        st.caption("This LSTM uses selected FEATURES + past TARGET to predict future TARGET.")
        st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="btn-row">', unsafe_allow_html=True)
        st.button("⬅ Back to Preprocess", use_container_width=True, on_click=request_nav, args=("preprocess",))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
        st.button(
            "Continue to Training ➜",
            use_container_width=True,
            on_click=lambda: (p.__setitem__("status", "configured"), upsert_project(p), request_nav("train")),
        )
        st.markdown("</div>", unsafe_allow_html=True)


def page_train():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Train")
    status_bar(p.get("status", "data_loaded"))

    ds = p.get("dataset", {})
    features = p.get("columns", {}).get("features", [])

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Model Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Architecture", p.get("model_type", "—").upper())
    c2.metric("Task", p.get("task_type", "—"))
    c3.metric("Features", len(features))
    train_samples = int(ds.get("rows", 0) * float(p["preprocess"]["split"])) if ds.get("rows", 0) else 0
    c4.metric("Training Samples", train_samples)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Training Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        p["train_config"]["epochs"] = int(st.number_input("Epochs", min_value=1, max_value=500, value=int(p["train_config"]["epochs"])))
    with c2:
        p["train_config"]["batch_size"] = int(st.number_input("Batch size", min_value=1, max_value=2048, value=int(p["train_config"]["batch_size"])))
    with c3:
        p["train_config"]["lr"] = float(st.number_input("Learning rate (ANN only)", min_value=1e-6, max_value=1.0, value=float(p["train_config"]["lr"]), format="%.6f"))
    with c4:
        p["train_config"]["patience"] = int(st.number_input("Early stop patience", min_value=1, max_value=50, value=int(p["train_config"]["patience"])))
    p["train_config"]["early_stop"] = st.checkbox("Enable early stopping", value=bool(p["train_config"]["early_stop"]))
    st.markdown("</div>", unsafe_allow_html=True)

    upsert_project(p)

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Training Controls")
    st.markdown('<div class="btn-row btn-emerald">', unsafe_allow_html=True)
    run = st.button("▶ Start Training", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        dataset_path = ds.get("path")
        if not dataset_path:
            st.error("No dataset saved. Go to Data page and upload a file.")
            return

        target = p.get("columns", {}).get("target")
        time_col = p.get("columns", {}).get("time")
        feature_cols = p.get("columns", {}).get("features", [])

        if not target:
            st.error("Select target in Data page first.")
            return

        try:
            df = load_project_dataset(p)
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return

        epochs = int(p["train_config"]["epochs"])
        batch_size = int(p["train_config"]["batch_size"])
        lr = float(p["train_config"]["lr"])
        early_stop = bool(p["train_config"]["early_stop"])
        patience = int(p["train_config"]["patience"])
        seed = int(p["preprocess"]["seed"])
        test_size = 1.0 - float(p["preprocess"]["split"])

        MODELS_DIR = DATA_DIR / "models" / p["id"]
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        if p.get("model_type") == "ann":
            if not feature_cols:
                st.error("Select feature columns in Data page first (ANN needs features).")
                return

            ui_task = p.get("task_type", "classification")
            task_choice = "auto" if ui_task == "auto-detect" else ui_task

            try:
                with st.spinner("Training model (real ANN)..."):
                    model, preprocessor, label_encoder, results, history = train_ann_from_df(
                        df=df,
                        target_col=target,
                        feature_cols=feature_cols,
                        task_choice=task_choice,
                        test_size=test_size,
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=lr,
                        early_stop=early_stop,
                        patience=patience,
                        seed=seed,
                        ann_config=p.get("ann_config", {}),
                    )
            except Exception as e:
                st.error(f"ANN training could not start: {e}")
                return

            # Save residual plot data for ANN regression
            try:
                if results.get("task") == "regression":
                    sk = _get_sklearn()
                    train_test_split = sk["train_test_split"]

                    X_full = df[feature_cols].copy()
                    y_full = pd.to_numeric(df[target], errors="coerce").astype(float).values

                    keep = ~np.isnan(y_full)
                    X_full = X_full.loc[keep].reset_index(drop=True)
                    y_full = y_full[keep]

                    X_full = _expand_datetime_features(X_full)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X_full,
                        y_full,
                        test_size=test_size,
                        random_state=seed,
                        shuffle=True,
                    )

                    X_test_np = preprocessor.transform(X_test)
                    y_pred = model.predict(X_test_np, verbose=0).reshape(-1)

                    p.setdefault("viz_cache", {})
                    p["viz_cache"]["ann_regression"] = {
                        "y_test": y_test.tolist(),
                        "y_pred": y_pred.tolist(),
                    }
                else:
                    p.setdefault("viz_cache", {})
                    p["viz_cache"].pop("ann_regression", None)
            except Exception:
                p.setdefault("viz_cache", {})
                p["viz_cache"].pop("ann_regression", None)

            model_path = MODELS_DIR / "model.keras"
            prep_path = MODELS_DIR / "preprocessor.joblib"
            le_path = MODELS_DIR / "label_encoder.joblib"

            model.save(model_path)
            joblib.dump(preprocessor, prep_path)
            joblib.dump(label_encoder, le_path)

            try:
                cached_load_ann_artifacts.clear()
            except Exception:
                pass

            p["artifacts"] = {
                "kind": "ann",
                "model_path": str(model_path),
                "preprocessor_path": str(prep_path),
                "label_encoder_path": str(le_path),
            }

            try:
                p["target_profile"] = build_target_profile(df, target)
            except Exception:
                p["target_profile"] = {}

            p["history"] = history
            p["evaluation_metrics"] = results
            p["status"] = "trained"
            upsert_project(p)

            st.success("Training finished (ANN). You can now Evaluate → Predict → Visualize.")

        else:
            if not feature_cols:
                st.error("For MULTIVARIATE LSTM, please select feature columns in Data page.")
                return

            cfg = LSTMConfig(
                target_col=target,
                feature_cols=feature_cols,
                date_col=time_col,
                lookback=int(p["preprocess"]["lookback"]),
                horizon=int(p["preprocess"]["horizon"]),
                test_size=float(test_size),
                lstm_units=int(p.get("lstm_config", {}).get("units", 64)),
                dropout=float(p.get("lstm_config", {}).get("dropout", 0.2)),
                epochs=int(epochs),
                batch_size=int(batch_size),
                patience=int(patience),
                seed=int(seed),
                val_split=0.1,
            )

            with st.spinner("Training model (real MULTIVARIATE LSTM)..."):
                model, pack, history, metrics, outputs = train_lstm_from_df(df, cfg)

            if (metrics or {}).get("status") != "ok" or model is None or pack is None:
                st.error((metrics or {}).get("message", "LSTM training could not start."))
                warns = (metrics or {}).get("warnings", [])
                if warns:
                    st.warning("\n".join([f"• {w}" for w in warns]))
                return

            model_path = MODELS_DIR / "lstm_model.keras"
            pack_path = MODELS_DIR / "lstm_pack.joblib"
            outputs_path = MODELS_DIR / "lstm_outputs.npz"

            model.save(model_path)
            joblib.dump(pack, pack_path)
            np.savez_compressed(outputs_path, **outputs)

            try:
                cached_load_lstm_artifacts.clear()
            except Exception:
                pass

            used_lookback = int(outputs["lookback"][0]) if "lookback" in outputs else cfg.lookback

            p["artifacts"] = {
                "kind": "lstm",
                "model_path": str(model_path),
                "pack_path": str(pack_path),
                "outputs_path": str(outputs_path),
                "lookback": used_lookback,
            }

            try:
                p["target_profile"] = build_target_profile(df, target)
            except Exception:
                p["target_profile"] = {}

            p["history"] = {"loss": history.get("loss", []), "val_loss": history.get("val_loss", [])}
            p["evaluation_metrics"] = metrics
            p["status"] = "trained"
            upsert_project(p)

            st.success("Training finished (Multivariate LSTM). You can now Evaluate → Predict → Visualize.")
            if metrics.get("warnings"):
                st.warning("\n".join([f"• {w}" for w in metrics["warnings"]]))

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="btn-row">', unsafe_allow_html=True)
        st.button("⬅ Back to Model", use_container_width=True, on_click=request_nav, args=("model",))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
        st.button("Continue to Evaluate ➜", use_container_width=True, on_click=request_nav, args=("evaluate",))
        st.markdown("</div>", unsafe_allow_html=True)


def page_evaluate():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Evaluate")
    status_bar(p.get("status", "data_loaded"))

    if p.get("status") not in ["trained", "evaluated"]:
        st.warning("Please train a model first.")
        return

    metrics = p.get("evaluation_metrics")
    if metrics is None:
        st.info("No evaluation metrics found. Train the model first.")
        return

    task = metrics.get("task", "classification")
    is_cls = (task == "classification")

    if is_cls:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics.get('accuracy', 0.0) * 100:.1f}%")
        c2.metric("Precision", f"{metrics.get('precision', 0.0) * 100:.1f}%")
        c3.metric("Recall", f"{metrics.get('recall', 0.0) * 100:.1f}%")
        c4.metric("F1 Score", f"{metrics.get('f1_score', 0.0) * 100:.1f}%")

        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Confusion Matrix (labeled)")
        cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
        labels = metrics.get("class_labels") or None
        cm_df = pd.DataFrame(cm)
        if labels and len(labels) == cm_df.shape[0] == cm_df.shape[1]:
            cm_df.index = [f"Actual: {x}" for x in labels]
            cm_df.columns = [f"Pred: {x}" for x in labels]
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Rows = actual, columns = predicted.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{metrics.get('mae', 0.0):.4f}")
        c2.metric("RMSE", f"{metrics.get('rmse', 0.0):.4f}")
        r2v = float(metrics.get("r2_score", 0.0))
        c3.metric("R² Score", f"{r2v:.4f}")

        if "channels_used" in metrics:
            st.write("")
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.write("### LSTM Details")
            st.write(f"Channels used: **{metrics.get('channels_used')}** (features + target)")
            st.write(f"Lookback: **{metrics.get('lookback_used')}** • Horizon: **{metrics.get('horizon_used')}**")
            st.write(f"Rows used: **{metrics.get('rows_used')}**")
            st.write(f"Train sequences: **{metrics.get('train_sequences')}** • Test sequences: **{metrics.get('test_sequences')}**")
            st.markdown("</div>", unsafe_allow_html=True)

    p["status"] = "evaluated"
    upsert_project(p)

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="btn-row">', unsafe_allow_html=True)
        st.button("⬅ Back to Train", use_container_width=True, on_click=request_nav, args=("train",))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
        st.button("Continue to Predict ➜", use_container_width=True, on_click=request_nav, args=("predict",))
        st.markdown("</div>", unsafe_allow_html=True)


def page_predict():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Predict")
    status_bar(p.get("status", "data_loaded"))

    artifacts = p.get("artifacts")
    if not artifacts:
        st.warning("No trained model found. Train the model first.")
        return

    kind = artifacts.get("kind", "ann")

    if kind == "ann":
        features = p.get("columns", {}).get("features", [])
        if not features:
            st.warning("No features selected. Go to Data Upload and select feature columns.")
            return

        try:
            model, preprocessor, label_encoder = cached_load_ann_artifacts(
                artifacts["model_path"],
                artifacts["preprocessor_path"],
                artifacts["label_encoder_path"],
            )
        except Exception as e:
            st.error(f"Failed to load trained model artifacts: {e}")
            return

        target_name = (p.get("columns") or {}).get("target") or "target"
        task = (p.get("evaluation_metrics") or {}).get("task", "classification")

        tab1, tab2 = st.tabs(["Manual Input", "Upload File"])

        with tab1:
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.write("### Input Features")

            st.caption(
                f"**What will be predicted?** This model estimates **{target_name}** from the feature values you enter below. "
                + ("**Classification:** predicts a label/class." if task == "classification"
                   else "**Regression:** predicts a numeric value.")
            )

            meta = p.get("feature_meta", {}) or {}
            vals: Dict[str, Any] = {}
            cols2 = st.columns(2)

            for i, f in enumerate(features):
                with cols2[i % 2]:
                    fmeta = meta.get(f, {"type": "numeric"})
                    if fmeta.get("type") == "categorical":
                        options = fmeta.get("options") or []
                        vals[f] = st.selectbox(f, options=options, index=0) if options else st.text_input(f, value="")
                    else:
                        vals[f] = st.number_input(f, value=0.0, format="%.6f")

            st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown('<div class="btn-row btn-emerald">', unsafe_allow_html=True)
            do = st.button("🔮 Predict", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            if do:
                row = pd.DataFrame([vals], columns=features)
                pred, conf = predict_from_df(row, model, preprocessor, task=task, label_encoder=label_encoder)

                st.write("")
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.write("### Prediction Result")

                a, b, c = st.columns([1.2, 1.0, 1.2], gap="large")
                with a:
                    st.caption("Target")
                    st.markdown(f"**{target_name}**")
                with b:
                    st.caption("Output Type")
                    st.markdown(f"**{'Classification' if task == 'classification' else 'Regression'}**")
                with c:
                    st.caption("Model")
                    st.markdown("**ANN**")

                st.write("")

                if task == "classification":
                    pred_label = str(pred[0])
                    conf_pct = f"{float(conf[0]) * 100:.1f}%" if conf is not None else "—"

                    c1, c2 = st.columns(2, gap="large")
                    with c1:
                        st.metric(label=f"Predicted {target_name}", value=pred_label)
                    with c2:
                        st.metric(label="Confidence", value=conf_pct)

                    st.caption("Confidence is the model’s estimated certainty for the predicted label.")
                else:
                    st.metric(label=f"Predicted {target_name}", value=f"{float(pred[0]):.6f}")
                    st.caption("This numeric value is estimated by the trained model using your input features.")

                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            up = st.file_uploader("Upload CSV or Excel for batch predictions", type=["csv", "xlsx", "xls"], key="pred_upload")
            st.caption("File must include the same feature columns.")
            sheet = None
            df_up = None

            if up is not None:
                try:
                    if up.name.lower().endswith(".csv"):
                        df_up = pd.read_csv(up)
                    else:
                        xls = pd.ExcelFile(up)
                        sheet = st.selectbox("Select sheet", xls.sheet_names, index=0, key="pred_sheet")
                        df_up = pd.read_excel(up, sheet_name=sheet)
                except Exception as e:
                    st.error(f"Could not read uploaded prediction file: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

            if df_up is not None:
                missing_cols = [c for c in features if c not in df_up.columns]
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    st.markdown('<div class="btn-row btn-emerald">', unsafe_allow_html=True)
                    gen = st.button("Generate Predictions", use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    if gen:
                        X_feat = df_up[features].copy()
                        preds, conf = predict_from_df(X_feat, model, preprocessor, task=task, label_encoder=label_encoder)

                        out = df_up.copy()
                        out[f"prediction__{target_name}"] = preds
                        if task == "classification" and conf is not None:
                            out["confidence"] = np.round(conf.astype(float), 6)

                        st.success("Predictions generated (REAL).")
                        st.dataframe(out.head(20), use_container_width=True)

                        csv_bytes = out.to_csv(index=False).encode("utf-8")
                        st.download_button("Download Results", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    else:
        try:
            model, pack, outputs = cached_load_lstm_artifacts(
                artifacts["model_path"],
                artifacts["pack_path"],
                artifacts["outputs_path"],
            )
        except Exception as e:
            st.error(f"Failed to load LSTM artifacts: {e}")
            return

        KEY_FC_STEPS = f"fc_steps__{p['id']}"
        KEY_FC_SHOW = f"fc_show__{p['id']}"
        KEY_FC_DF = f"fc_df__{p['id']}"

        st.session_state.setdefault(KEY_FC_STEPS, 1)
        st.session_state.setdefault(KEY_FC_SHOW, 1)

        def _generate_forecast(n_steps: int) -> pd.DataFrame:
            scaled_full = outputs["scaled_full"]
            lookback = int(artifacts.get("lookback", int(outputs["lookback"][0]) if "lookback" in outputs else 10))
            last_feat = outputs["last_feature_scaled"]
            scaler_y = pack["scaler_y"]

            future = forecast_future_multivariate(
                model=model,
                scaler_y=scaler_y,
                scaled_full=scaled_full,
                lookback=lookback,
                n_steps=int(n_steps),
                last_feature_scaled=last_feat,
            )
            return pd.DataFrame(future, columns=["forecast"])

        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### Future Forecast (Multivariate LSTM)")
        st.caption("If future feature values are unknown, the app holds features constant at the last observed values.")

        st.number_input(
            "Number of future steps to forecast",
            min_value=1,
            step=1,
            key=KEY_FC_STEPS,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="btn-row btn-emerald">', unsafe_allow_html=True)
        do = st.button("🔮 Forecast", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if do:
            try:
                n_steps = int(st.session_state[KEY_FC_STEPS])
                df_fc = _generate_forecast(n_steps)
                if df_fc.empty:
                    st.error("Forecast generated no results. Try increasing data or adjusting lookback/horizon.")
                else:
                    st.session_state[KEY_FC_DF] = df_fc
                    st.session_state[KEY_FC_SHOW] = max(1, min(int(st.session_state[KEY_FC_SHOW]), int(len(df_fc))))
            except Exception as e:
                st.error(f"Forecast error: {e}")

        df_fc: Optional[pd.DataFrame] = st.session_state.get(KEY_FC_DF, None)

        if df_fc is not None and not df_fc.empty:
            target_name = (p.get("columns") or {}).get("target") or "target"

            st.write("")
            st.markdown('<div class="ns-card">', unsafe_allow_html=True)
            st.write("### Forecast Result")

            a, b, c = st.columns([1.2, 1.0, 1.2], gap="large")
            with a:
                st.caption("Target")
                st.markdown(f"**{target_name}**")
            with b:
                st.caption("Steps Generated")
                st.markdown(f"**{len(df_fc)}**")
            with c:
                st.caption("Model")
                st.markdown("**LSTM (Time Series)**")

            first_val = float(df_fc["forecast"].iloc[0])
            st.metric(label=f"Forecasted {target_name} (next step)", value=f"{first_val:,.4f}")

            st.write("")
            st.number_input(
                "Number of forecast steps shown",
                min_value=1,
                step=1,
                key=KEY_FC_SHOW,
            )

            want_show = int(st.session_state[KEY_FC_SHOW])

            if want_show > len(df_fc):
                try:
                    df_new = _generate_forecast(want_show)
                    if not df_new.empty:
                        st.session_state[KEY_FC_DF] = df_new
                        df_fc = df_new
                except Exception as e:
                    st.warning(f"Could not auto-expand forecast: {e}")
                    want_show = len(df_fc)

            show_n = max(1, min(want_show, int(len(df_fc))))
            preview_df = df_fc.head(show_n).copy()
            preview_df.insert(0, "step", range(1, len(preview_df) + 1))

            st.dataframe(preview_df, use_container_width=True, height=320)

            csv_bytes = df_fc.to_csv(index=False).encode("utf-8")
            st.download_button("Download Forecast CSV", data=csv_bytes, file_name="lstm_forecast.csv", mime="text/csv")

            st.caption("These values are generated from the trained LSTM model based on your historical data.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="btn-row">', unsafe_allow_html=True)
        st.button("⬅ Back to Evaluate", use_container_width=True, on_click=request_nav, args=("evaluate",))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
        st.button("Continue to Visualize ➜", use_container_width=True, on_click=request_nav, args=("visualize",))
        st.markdown("</div>", unsafe_allow_html=True)


def page_visualize():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Visualize")
    status_bar(p.get("status", "data_loaded"))
    plt = _get_plt()

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Target Preview")

    ds = p.get("dataset", {})
    target = (p.get("columns") or {}).get("target")
    if ds.get("path") and target:
        try:
            df = load_project_dataset(p)
            if target not in df.columns:
                st.info("Target column not found in the dataset.")
            else:
                y = df[target].copy()
                y_num = pd.to_numeric(y, errors="coerce")

                if y_num.notna().sum() >= 5:
                    y_plot = y_num.dropna().head(400)
                    fig0 = plt.figure(figsize=(11, 4))
                    plt.plot(range(len(y_plot)), y_plot.values)
                    plt.title(f"Target Trend (first {len(y_plot)} points)")
                    plt.xlabel("Index")
                    plt.ylabel(target)
                    plt.grid(True, alpha=0.25)
                    st.pyplot(fig0, clear_figure=True)
                    st.caption("Shows the beginning of the target series after numeric conversion.")
                else:
                    vc = y.astype(str).value_counts().head(20)
                    fig0 = plt.figure(figsize=(11, 4))
                    plt.bar(vc.index.astype(str), vc.values)
                    plt.title("Target Category Counts (Top 20)")
                    plt.xlabel("Class")
                    plt.ylabel("Count")
                    plt.xticks(rotation=35, ha="right")
                    plt.grid(True, axis="y", alpha=0.25)
                    st.pyplot(fig0, clear_figure=True)
                    st.caption("Target is not numeric; showing class distribution.")
        except Exception as e:
            st.info(f"Preview plot unavailable: {e}")
    else:
        st.info("Upload data and choose a target column to see the target preview.")
    st.markdown("</div>", unsafe_allow_html=True)

    loss = (p.get("history") or {}).get("loss", []) or []
    val_loss = (p.get("history") or {}).get("val_loss", []) or []

    st.write("")
    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Training Curve (Loss)")
    if loss:
        fig = plt.figure(figsize=(11, 4))
        x = range(1, len(loss) + 1)
        plt.plot(x, loss, label="Train Loss", linewidth=2.0)
        if val_loss:
            xv = range(1, len(val_loss) + 1)
            plt.plot(xv, val_loss, label="Validation Loss", linewidth=2.0)
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (lower is better)")
        plt.grid(True, alpha=0.25)
        plt.legend()
        st.pyplot(fig, clear_figure=True)
        st.caption("Lower loss generally means better fit. Validation loss helps detect overfitting.")
    else:
        st.info("No training history yet. Train the model first.")
    st.markdown("</div>", unsafe_allow_html=True)

    viz = (p.get("viz_cache") or {}).get("ann_regression")
    if viz:
        try:
            y_test = np.array(viz.get("y_test", []), dtype=float)
            y_pred = np.array(viz.get("y_pred", []), dtype=float)
            if len(y_test) and len(y_pred) and len(y_test) == len(y_pred):
                residuals = y_test - y_pred

                st.write("")
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.write("### Residual Plot (ANN Regression)")
                fig_r = plt.figure(figsize=(11, 4))
                plt.scatter(y_pred, residuals, alpha=0.6)
                plt.axhline(0, linestyle="--")
                plt.title("Residuals vs Predicted")
                plt.xlabel("Predicted")
                plt.ylabel("Residual (Actual - Predicted)")
                plt.grid(True, alpha=0.25)
                st.pyplot(fig_r, clear_figure=True)
                st.caption("Good model: residuals spread randomly around 0 without a visible pattern.")
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass

    artifacts = p.get("artifacts") or {}
    if artifacts.get("kind") == "lstm":
        st.write("")
        st.markdown('<div class="ns-card">', unsafe_allow_html=True)
        st.write("### LSTM: Actual vs Predicted")
        try:
            _, _, outputs = cached_load_lstm_artifacts(
                artifacts["model_path"],
                artifacts["pack_path"],
                artifacts["outputs_path"],
            )

            y_train_actual = outputs["y_train_actual"].reshape(-1)
            y_test_actual = outputs["y_test_actual"].reshape(-1)
            train_pred_actual = outputs["train_pred_actual"].reshape(-1)
            test_pred_actual = outputs["test_pred_actual"].reshape(-1)

            lookback = int(artifacts.get("lookback", int(outputs["lookback"][0]) if "lookback" in outputs else 10))

            fig2 = plt.figure(figsize=(11, 4.5))
            train_idx = range(lookback, lookback + len(y_train_actual))
            test_idx = range(lookback + len(y_train_actual), lookback + len(y_train_actual) + len(y_test_actual))

            plt.plot(train_idx, y_train_actual, label="Actual (Train)", alpha=0.65, linewidth=1.8)
            plt.plot(train_idx, train_pred_actual, label="Predicted (Train)", alpha=0.65, linestyle="--", linewidth=1.8)
            plt.plot(test_idx, y_test_actual, label="Actual (Test)", alpha=0.85, linewidth=2.0)
            plt.plot(test_idx, test_pred_actual, label="Predicted (Test)", alpha=0.85, linestyle="--", linewidth=2.0)
            plt.axvline(x=lookback + len(y_train_actual), linestyle=":", label="Train/Test Split")

            plt.title("LSTM Forecast Performance")
            plt.xlabel("Time Index")
            plt.ylabel("Target Value")
            plt.grid(True, alpha=0.25)
            plt.legend()
            st.pyplot(fig2, clear_figure=True)
        except Exception as e:
            st.info(f"LSTM plot not available: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

        try:
            y_test_l = outputs["y_test_actual"].reshape(-1)
            y_pred_l = outputs["test_pred_actual"].reshape(-1)
            if len(y_test_l) and len(y_pred_l) and len(y_test_l) == len(y_pred_l):
                residuals_l = y_test_l - y_pred_l

                st.write("")
                st.markdown('<div class="ns-card">', unsafe_allow_html=True)
                st.write("### Residual Plot (LSTM Test)")
                fig_rl = plt.figure(figsize=(11, 4))
                plt.scatter(y_pred_l, residuals_l, alpha=0.6)
                plt.axhline(0, linestyle="--")
                plt.title("Residuals vs Predicted (Test)")
                plt.xlabel("Predicted")
                plt.ylabel("Residual (Actual - Predicted)")
                plt.grid(True, alpha=0.25)
                st.pyplot(fig_rl, clear_figure=True)
                st.caption("Residuals computed from saved LSTM test predictions.")
                st.markdown("</div>", unsafe_allow_html=True)
        except Exception:
            pass

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#a7f3d0; background:#ecfdf5;">', unsafe_allow_html=True)
    st.write("### Evaluation Metrics")
    m = p.get("evaluation_metrics")
    if not m:
        st.caption("No evaluation metrics yet.")
    else:
        st.json(m)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="btn-row">', unsafe_allow_html=True)
        st.button("⬅ Back to Predict", use_container_width=True, on_click=request_nav, args=("predict",))
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="btn-row btn-grad">', unsafe_allow_html=True)
        st.button("Continue to Save / Load ➜", use_container_width=True, on_click=request_nav, args=("save",))
        st.markdown("</div>", unsafe_allow_html=True)


def page_save_load():
    p = ensure_current_project()
    if p is None:
        st.warning("No current project.")
        return

    st.title("Save / Load")
    status_bar(p.get("status", "data_loaded"))

    st.markdown('<div class="ns-card">', unsafe_allow_html=True)
    st.write("### Current Model Info")
    st.write(f"**Project:** {p.get('name')}")
    st.write(f"**Model:** {p.get('model_type', '—').upper()} • **Task:** {(p.get('evaluation_metrics') or {}).get('task', p.get('task_type', '—'))}")
    st.write(f"**Status:** {p.get('status', '—')}")
    st.write("**Features:** " + ", ".join(p.get("columns", {}).get("features", [])[:12]) + (" ..." if len(p.get("columns", {}).get("features", [])) > 12 else ""))
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#a7f3d0;">', unsafe_allow_html=True)
    st.write("### Save Project Package (JSON)")
    st.caption("Saves UI settings + history + paths. (Model files are stored under .neural_studio/models/)")
    json_bytes = json.dumps(_json_safe(p), indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button("💾 Download JSON", data=json_bytes, file_name="neural_studio_project.json", mime="application/json")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="ns-card" style="border-color:#c7d2fe;">', unsafe_allow_html=True)
    st.write("### Load Project Package (JSON)")
    up = st.file_uploader("Upload saved JSON", type=["json"])
    if up is not None:
        try:
            loaded = json.loads(up.read().decode("utf-8"))
            if "id" not in loaded:
                loaded["id"] = str(uuid.uuid4())
            loaded.setdefault("columns", {"target": None, "time": None, "features": []})
            loaded.setdefault("dataset", {"filename": "—", "rows": 0, "cols": 0, "missing": 0, "path": None, "file_type": None, "sheet": None})
            loaded.setdefault("preprocess", {"split": 0.8, "seed": 42, "lookback": 20, "horizon": 1, "missing_strategy": "Drop rows"})
            loaded.setdefault("train_config", {"epochs": 20, "batch_size": 32, "lr": 0.001, "early_stop": True, "patience": 5})
            loaded.setdefault("ann_config", {"hidden_layers": 3, "neurons": [256, 128, 64], "activation": "ReLU", "output_activation": "Auto"})
            loaded.setdefault("lstm_config", {"units": 64, "layers": 2, "dropout": 0.2, "bidirectional": False})
            loaded.setdefault("viz_cache", {})
            loaded.setdefault("target_profile", {})
            upsert_project(loaded)
            request_nav("data")
        except Exception as e:
            st.error(f"Failed to load JSON: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="btn-row">', unsafe_allow_html=True)
    st.button("⬅ Back to Visualize", use_container_width=True, on_click=request_nav, args=("visualize",))
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Routing (✅ requested order)
# ============================================================
PAGES = {
    "home": ("Home", page_home),
    "data": ("Data Upload", page_data),
    "preprocess": ("Preprocess", page_preprocess),
    "model": ("Model", page_model),
    "train": ("Train", page_train),
    "evaluate": ("Evaluate", page_evaluate),
    "predict": ("Predict", page_predict),
    "visualize": ("Visualize", page_visualize),
    "save": ("Save/Load", page_save_load),
}


def main():
    st.set_page_config(page_title="Neural Studio", layout="wide")
    inject_css()

    apply_pending_nav()

    with st.sidebar:
        st.markdown("## Neural Studio")
        st.caption("ML workflow builder (Python UI)")

        cur = get_current_project()
        if cur:
            st.markdown(f"**Current:** {cur.get('name', 'Untitled')}")
            st.caption(project_badge(cur))
        else:
            st.info("No current project.")

        st.write("")

        page_keys = list(PAGES.keys())
        labels = [PAGES[k][0] for k in page_keys]

        active_key = st.query_params.get("page", "home")
        if active_key not in PAGES:
            active_key = "home"

        idx = page_keys.index(active_key)
        chosen = st.radio("Navigate", labels, index=idx)

        chosen_key = page_keys[labels.index(chosen)]
        if chosen_key != active_key:
            request_nav(chosen_key)

        st.write("---")
        if st.button("➕ New Project", use_container_width=True):
            new_project()
            request_nav("data")

        if st.button("🗑️ Clear Current Project", use_container_width=True):
            if CURRENT_FILE.exists():
                CURRENT_FILE.unlink()
            request_nav("home")

    apply_pending_nav()

    active_key = st.query_params.get("page", "home")
    if active_key not in PAGES:
        active_key = "home"

    bottom_nav(active_key)

    _, renderer = PAGES[active_key]
    renderer()


if __name__ == "__main__":
    main()
