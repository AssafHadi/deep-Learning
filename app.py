
from __future__ import annotations

import os
import sys
from pathlib import Path

ANN_SOURCE = r'''
import os
import io
import re
import json
import math
import time
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------- #
# App metadata / persistence
# ----------------------------- #
APP_TITLE = "Oil & Gas ANN Studio"
APP_SUBTITLE = "MATLAB-style workflow for tabular oil, gas, drilling, reservoir, and production data"
APP_DIR = Path(".oil_gas_ann_studio")
PROJECTS_DIR = APP_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- #
# Lazy imports
# ----------------------------- #
_TF = None
_PLT = None

def get_tf():
    global _TF
    if _TF is None:
        import tensorflow as tf
        _TF = tf
    return _TF

def get_plt():
    global _PLT
    if _PLT is None:
        import matplotlib.pyplot as plt
        plt.rcParams["figure.dpi"] = 125
        plt.rcParams["savefig.dpi"] = 125
        plt.rcParams["axes.titlesize"] = 15
        plt.rcParams["axes.labelsize"] = 12.5
        plt.rcParams["xtick.labelsize"] = 10.5
        plt.rcParams["ytick.labelsize"] = 10.5
        plt.rcParams["legend.fontsize"] = 10.5
        _PLT = plt
    return _PLT

# sklearn imports
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# ----------------------------- #
# Session state
# ----------------------------- #
def default_config() -> Dict[str, Any]:
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

def init_state():
    if "config" not in st.session_state:
        st.session_state.config = default_config()
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "prepared_data" not in st.session_state:
        st.session_state.prepared_data = None
    if "trained_model" not in st.session_state:
        st.session_state.trained_model = None
    if "training_history" not in st.session_state:
        st.session_state.training_history = None
    if "results" not in st.session_state:
        st.session_state.results = {}
    if "loaded_project_name" not in st.session_state:
        st.session_state.loaded_project_name = None

# ----------------------------- #
# Styling
# ----------------------------- #
def inject_css():
    st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
        <style>
        .main {padding-top: 1rem;}
        .block-container {padding-top: 1.3rem; padding-bottom: 2rem; max-width: 1320px;}
        .hero {
            padding: 1.25rem 1.5rem;
            border-radius: 20px;
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 45%, #0ea5e9 100%);
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(2,6,23,0.18);
        }
        .hero h1 {margin: 0; font-size: 2rem; line-height: 1.15;}
        .hero p {margin: .5rem 0 0 0; font-size: 1rem; opacity: 0.95;}
        .soft-card {
            background: #ffffff;
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem;
            box-shadow: 0 6px 22px rgba(15,23,42,0.05);
            height: 100%;
        }
        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 6px 22px rgba(15,23,42,0.04);
        }
        .tiny {font-size: 0.86rem; color: #475569;}
        .status-pill {
            display: inline-block;
            padding: .35rem .65rem;
            border-radius: 999px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #1d4ed8;
            font-size: .85rem;
            margin-right: .35rem;
            margin-bottom: .35rem;
        }
        .section-note {
            padding: .8rem 1rem;
            border-left: 4px solid #1d4ed8;
            background: #eff6ff;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def hero():
    st.markdown(
        f"""
        <div class="hero">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------------------- #
# Utility helpers
# ----------------------------- #
def set_seed(seed: int):
    np.random.seed(seed)
    try:
        tf = get_tf()
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass

def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return name or "project"

def parse_hidden_layers(text: str) -> List[int]:
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

def read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Only CSV, XLSX, and XLS files are supported.")

def is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            return False
        converted = pd.to_datetime(sample, errors="coerce")
        ratio = converted.notna().mean()
        return ratio >= 0.8
    return False

def expand_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    expanded = []
    for col in list(out.columns):
        s = out[col]
        if is_datetime_like(s):
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() == 0:
                continue
            out[f"{col}__year"] = dt.dt.year
            out[f"{col}__month"] = dt.dt.month
            out[f"{col}__day"] = dt.dt.day
            out[f"{col}__dayofweek"] = dt.dt.dayofweek
            out[f"{col}__hour"] = dt.dt.hour
            out[f"{col}__is_month_end"] = dt.dt.is_month_end.astype("float64")
            out[f"{col}__is_month_start"] = dt.dt.is_month_start.astype("float64")
            out.drop(columns=[col], inplace=True)
            expanded.append(col)
    return out, expanded

def infer_task(y: pd.Series) -> str:
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

def mape_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    mape = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.nanmean(mape))

def get_feature_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    schema = {}
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        entry = {
            "dtype": dtype_str,
            "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
            "is_bool": bool(pd.api.types.is_bool_dtype(s)),
            "example": None if s.dropna().empty else str(s.dropna().iloc[0]),
            "categories": None,
        }
        if not entry["is_numeric"]:
            cats = s.dropna().astype(str).value_counts().head(30).index.tolist()
            entry["categories"] = cats
        schema[col] = entry
    return schema

def ensure_feature_frame(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required_columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out[required_columns]
    return out

def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)

# ----------------------------- #
# Data preparation and training
# ----------------------------- #
def build_preprocessor(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols

def prepare_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("No dataset loaded.")

    work_df = df.copy()

    if config["drop_duplicates"]:
        work_df = work_df.drop_duplicates().reset_index(drop=True)

    if config["target_column"] is None or config["target_column"] not in work_df.columns:
        raise ValueError("Choose a valid target column in Preprocess.")

    target_col = config["target_column"]

    # Drop rows with missing target
    work_df = work_df.loc[work_df[target_col].notna()].copy()
    if len(work_df) < 10:
        raise ValueError("Dataset is too small after removing rows with missing target.")

    drop_cols = [c for c in config["drop_columns"] if c in work_df.columns and c != target_col]
    if drop_cols:
        work_df = work_df.drop(columns=drop_cols)

    # Feature selection
    feature_cols = config["feature_columns"]
    if not feature_cols:
        feature_cols = [c for c in work_df.columns if c != target_col]
    feature_cols = [c for c in feature_cols if c in work_df.columns and c != target_col]
    if not feature_cols:
        raise ValueError("No valid feature columns selected.")

    X_raw = work_df[feature_cols].copy()
    y_raw = work_df[target_col].copy()

    datetime_expanded = []
    if config["auto_datetime"]:
        X_raw, datetime_expanded = expand_datetime_columns(X_raw)

    task = infer_task(y_raw) if config["task_mode"] == "Auto Detect" else config["task_mode"].lower()
    if task not in {"classification", "regression"}:
        task = "regression"

    preprocessor, num_cols, cat_cols = build_preprocessor(X_raw)

    if task == "classification":
        y_raw = y_raw.astype(str)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = label_encoder.classes_.tolist()
        stratify = y if len(np.unique(y)) > 1 and min(pd.Series(y).value_counts()) >= 2 else None
    else:
        label_encoder = None
        class_names = None
        y = pd.to_numeric(y_raw, errors="coerce").astype(float).values
        valid = ~np.isnan(y)
        X_raw = X_raw.loc[valid].reset_index(drop=True)
        y = y[valid]
        stratify = None

    if len(X_raw) < 10:
        raise ValueError("Not enough valid rows to continue.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=float(config["test_size"]),
        random_state=int(config["random_seed"]),
        shuffle=bool(config["shuffle_data"]),
        stratify=stratify,
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train.shape[1])]

    class_weights = None
    if task == "classification" and config.get("use_class_weights", True):
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    return {
        "df_used": work_df,
        "task": task,
        "target_column": target_col,
        "feature_columns_original": feature_cols,
        "feature_columns_after_datetime": X_raw.columns.tolist(),
        "datetime_expanded_columns": datetime_expanded,
        "feature_schema": get_feature_schema(work_df[feature_cols]),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "X_train": np.asarray(X_train).astype("float32"),
        "X_test": np.asarray(X_test).astype("float32"),
        "X_train_raw": X_train_raw.reset_index(drop=True),
        "X_test_raw": X_test_raw.reset_index(drop=True),
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "feature_names": feature_names,
        "class_weights": class_weights,
    }

def build_model(input_dim: int, task: str, config: Dict[str, Any], n_classes: int = 1):
    tf = get_tf()
    layers = parse_hidden_layers(config["hidden_layers"])
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

class StreamlitProgressCallback:
    def __init__(self, progress_bar, info_box, total_epochs):
        self.progress_bar = progress_bar
        self.info_box = info_box
        self.total_epochs = max(1, int(total_epochs))

    def as_callback(self):
        tf = get_tf()
        outer = self

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                frac = (epoch + 1) / outer.total_epochs
                outer.progress_bar.progress(min(frac, 1.0))
                txt = [f"Epoch {epoch + 1}/{outer.total_epochs}"]
                for k, v in logs.items():
                    try:
                        txt.append(f"{k}: {float(v):.4f}")
                    except Exception:
                        pass
                outer.info_box.markdown(" | ".join(txt))
        return _CB()

def train_model(prepared: Dict[str, Any], config: Dict[str, Any]):
    set_seed(int(config["random_seed"]))
    tf = get_tf()

    task = prepared["task"]
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_names = prepared["class_names"] or []

    n_classes = len(class_names) if task == "classification" else 1
    model = build_model(X_train.shape[1], task, config, n_classes=n_classes)

    callbacks = []
    progress = st.progress(0)
    info_box = st.empty()
    callbacks.append(StreamlitProgressCallback(progress, info_box, config["epochs"]).as_callback())

    if config["early_stopping"]:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=int(config["patience"]),
                restore_best_weights=True,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_split=float(config["validation_split"]),
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        verbose=0,
        callbacks=callbacks,
        class_weight=prepared["class_weights"],
    )

    # Predictions
    if task == "classification":
        proba = model.predict(X_test, verbose=0)
        if n_classes == 2:
            y_score = proba.reshape(-1)
            threshold = float(config["threshold"])
            y_pred = (y_score >= threshold).astype(int)
            y_prob_full = np.column_stack([1 - y_score, y_score])
        else:
            y_prob_full = proba
            y_pred = np.argmax(proba, axis=1)
            y_score = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
        if n_classes == 2 and y_score is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_score))
            except Exception:
                pass

        results = {
            "task": task,
            "metrics": metrics,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob_full,
            "y_score": y_score,
            "class_names": class_names,
            "classification_report": classification_report(
                y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
            ),
        }
    else:
        y_pred = model.predict(X_test, verbose=0).reshape(-1)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
            "mape": float(mape_safe(y_test, y_pred)),
        }
        results = {
            "task": task,
            "metrics": metrics,
            "y_true": y_test,
            "y_pred": y_pred,
            "residuals": y_test - y_pred,
        }

    progress.progress(1.0)
    info_box.success("Training finished.")
    return model, history.history, results

def predict_with_pipeline(model, prepared: Dict[str, Any], input_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    feature_cols = prepared["feature_columns_original"]
    raw = ensure_feature_frame(input_df, feature_cols)
    if prepared["datetime_expanded_columns"]:
        raw, _ = expand_datetime_columns(raw)

    # Align to training-time columns after datetime expansion
    required_after_dt = prepared["feature_columns_after_datetime"]
    raw = ensure_feature_frame(raw, required_after_dt)

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

# ----------------------------- #
# Serialization
# ----------------------------- #
def create_project_bundle_bytes() -> bytes:
    if st.session_state.trained_model is None or st.session_state.prepared_data is None:
        raise ValueError("Train or load a project first.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = safe_filename(st.session_state.config["project_name"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        model_path = tmp / "model.keras"
        bundle_path = tmp / "bundle.joblib"
        meta_path = tmp / "meta.json"

        st.session_state.trained_model.save(model_path)

        bundle = {
            "raw_df": st.session_state.raw_df,
            "config": st.session_state.config,
            "prepared_data": st.session_state.prepared_data,
            "training_history": st.session_state.training_history,
            "results": st.session_state.results,
        }
        joblib.dump(bundle, bundle_path)

        meta = {
            "project_name": project_name,
            "saved_at": datetime.now().isoformat(),
            "task": st.session_state.results.get("task"),
            "target_column": st.session_state.config.get("target_column"),
        }
        meta_path.write_text(json.dumps(meta, indent=2, default=json_default), encoding="utf-8")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path, arcname="model.keras")
            zf.write(bundle_path, arcname="bundle.joblib")
            zf.write(meta_path, arcname="meta.json")
        return buffer.getvalue()

def save_project_locally():
    data = create_project_bundle_bytes()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = safe_filename(st.session_state.config["project_name"])
    path = PROJECTS_DIR / f"{project_name}_{ts}.zip"
    path.write_bytes(data)
    return path

def load_project_from_zip_bytes(zip_bytes: bytes):
    tf = get_tf()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        zip_path = tmp / "project.zip"
        zip_path.write_bytes(zip_bytes)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)

        model_path = tmp / "model.keras"
        bundle_path = tmp / "bundle.joblib"

        model = tf.keras.models.load_model(model_path)
        bundle = joblib.load(bundle_path)

    st.session_state.raw_df = bundle["raw_df"]
    st.session_state.config = bundle["config"]
    st.session_state.prepared_data = bundle["prepared_data"]
    st.session_state.training_history = bundle["training_history"]
    st.session_state.results = bundle["results"]
    st.session_state.trained_model = model
    st.session_state.loaded_project_name = bundle["config"].get("project_name", "loaded_project")
    return bundle

# ----------------------------- #
# Visualization helpers
# ----------------------------- #
def plot_training_curves(history: Dict[str, List[float]]):
    if not history:
        st.info("No training history available yet.")
        return
    plt = get_plt()
    base_metrics = ["loss"] + sorted({
        (k[4:] if k.startswith("val_") else k)
        for k in history.keys()
        if k != "val_loss" and not k.startswith("val_val_")
    } - {"loss"})
    for metric in base_metrics:
        fig, ax = plt.subplots(figsize=(11.2, 4.8))
        ax.plot(history.get(metric, []), label=metric)
        if metric == "loss" and "val_loss" in history:
            ax.plot(history.get("val_loss", []), label="val_loss")
        elif f"val_{metric}" in history:
            ax.plot(history.get(f"val_{metric}", []), label=f"val_{metric}")
        ax.set_title(f"Training Curve - {metric}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True, use_container_width=True)

def plot_confusion(y_true, y_pred, class_names):
    plt = get_plt()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9.6, 7.4))
    im = ax.imshow(cm, aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    st.pyplot(fig, clear_figure=True, use_container_width=True)

def plot_binary_curves(y_true, y_score):
    if y_score is None:
        return
    plt = get_plt()

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig1, ax1 = plt.subplots(figsize=(9.2, 4.9))
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, clear_figure=True, use_container_width=True)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig2, ax2 = plt.subplots(figsize=(9.2, 5.0))
    ax2.plot(recall, precision)
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True, use_container_width=True)

def plot_regression_results(y_true, y_pred):
    plt = get_plt()
    residuals = y_true - y_pred

    fig1, ax1 = plt.subplots(figsize=(9.4, 6.2))
    ax1.scatter(y_true, y_pred, alpha=0.65)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax1.plot([mn, mx], [mn, mx], linestyle="--")
    ax1.set_title("Actual vs Predicted")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, clear_figure=True, use_container_width=True)

    fig2, ax2 = plt.subplots(figsize=(9.2, 5.0))
    ax2.scatter(y_pred, residuals, alpha=0.65)
    ax2.axhline(0.0, linestyle="--")
    std = np.std(residuals)
    ax2.axhline(2 * std, linestyle=":")
    ax2.axhline(-2 * std, linestyle=":")
    ax2.set_title("Residuals vs Predicted")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(9.2, 4.9))
    ax3.hist(residuals, bins=30)
    ax3.set_title("Residual Histogram")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, clear_figure=True, use_container_width=True)

    n = min(len(y_true), 300)
    fig4, ax4 = plt.subplots(figsize=(11.2, 4.8))
    ax4.plot(np.arange(n), y_true[:n], label="Actual")
    ax4.plot(np.arange(n), y_pred[:n], label="Predicted")
    ax4.set_title("Actual vs Predicted Trend (first 300 test samples)")
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Target")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4, clear_figure=True, use_container_width=True)

def plot_data_profile(df: pd.DataFrame, target_col: Optional[str]):
    plt = get_plt()
    if df is None or df.empty:
        st.info("Upload a dataset first.")
        return

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True, height=320)

    st.subheader("Column Summary")
    info = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing": [int(df[c].isna().sum()) for c in df.columns],
        "nunique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    st.dataframe(info, use_container_width=True, height=320)

    if target_col and target_col in df.columns:
        st.subheader("Target Preview")
        plot_col, _ = st.columns(2)
        with plot_col:
            if infer_task(df[target_col]) == "classification":
                counts = df[target_col].astype(str).value_counts().head(25)
                fig, ax = plt.subplots(figsize=(7.0, 3.8))
                ax.bar(counts.index.astype(str), counts.values)
                ax.set_title("Target Class Distribution")
                ax.set_xlabel("Class")
                ax.set_ylabel("Count")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig, clear_figure=True, use_container_width=True)
            else:
                fig, ax = plt.subplots(figsize=(7.0, 3.8))
                clean = pd.to_numeric(df[target_col], errors="coerce").dropna()
                ax.hist(clean, bins=35)
                ax.set_title("Target Histogram")
                ax.set_xlabel(target_col)
                ax.set_ylabel("Frequency")
                st.pyplot(fig, clear_figure=True, use_container_width=True)

def make_training_curve_fig(history: Dict[str, List[float]], metric: str):
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(history.get(metric, []), label=metric)
    if metric == "loss" and "val_loss" in history:
        ax.plot(history.get("val_loss", []), label="val_loss")
    elif f"val_{metric}" in history:
        ax.plot(history.get(f"val_{metric}", []), label=f"val_{metric}")
    ax.set_title(f"Training Curve - {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def make_confusion_fig(y_true, y_pred, class_names):
    plt = get_plt()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    im = ax.imshow(cm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)
    return fig


def make_binary_curve_figs(y_true, y_score):
    if y_score is None:
        return None, None
    plt = get_plt()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig1, ax1 = plt.subplots(figsize=(7.0, 3.8))
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], linestyle="--")
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.grid(True, alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig2, ax2 = plt.subplots(figsize=(7.0, 3.8))
    ax2.plot(recall, precision)
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.grid(True, alpha=0.3)
    return fig1, fig2


def make_predicted_distribution_fig(y_pred):
    plt = get_plt()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    pd.Series(y_pred).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Predicted Class Distribution")
    ax.set_xlabel("Encoded Class")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2, axis="y")
    return fig


def make_regression_figures(y_true, y_pred):
    plt = get_plt()
    residuals = y_true - y_pred

    fig1, ax1 = plt.subplots(figsize=(7.0, 4.1))
    ax1.scatter(y_true, y_pred, alpha=0.65)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax1.plot([mn, mx], [mn, mx], linestyle="--")
    ax1.set_title("Actual vs Predicted")
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.grid(True, alpha=0.3)

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.1))
    ax2.scatter(y_pred, residuals, alpha=0.65)
    ax2.axhline(0.0, linestyle="--")
    std = np.std(residuals)
    ax2.axhline(2 * std, linestyle=":")
    ax2.axhline(-2 * std, linestyle=":")
    ax2.set_title("Residuals vs Predicted")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Residual")
    ax2.grid(True, alpha=0.3)

    fig3, ax3 = plt.subplots(figsize=(7.0, 3.8))
    ax3.hist(residuals, bins=30)
    ax3.set_title("Residual Histogram")
    ax3.set_xlabel("Residual")
    ax3.set_ylabel("Frequency")
    ax3.grid(True, alpha=0.3)

    n = min(len(y_true), 300)
    fig4, ax4 = plt.subplots(figsize=(7.0, 3.8))
    ax4.plot(np.arange(n), y_true[:n], label="Actual")
    ax4.plot(np.arange(n), y_pred[:n], label="Predicted")
    ax4.set_title("Actual vs Predicted Trend")
    ax4.set_xlabel("Sample Index")
    ax4.set_ylabel("Target")
    ax4.legend(loc="upper right")
    ax4.grid(True, alpha=0.3)
    return fig1, fig2, fig3, fig4

# ----------------------------- #
# Pages
# ----------------------------- #
def page_home():
    hero()
    cfg = st.session_state.config
    df = st.session_state.raw_df
    results = st.session_state.results

    st.markdown(
        """
        <div class="section-note">
        This app is built for tabular oil & gas problems such as production forecasting, pressure prediction,
        water-cut estimation, drilling parameter optimization, artificial-lift performance, and equipment status classification.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", 0 if df is None else len(df))
    c2.metric("Columns", 0 if df is None else len(df.columns))
    c3.metric("Target", cfg["target_column"] or "Not set")
    c4.metric("Task", results.get("task", "Not trained"))

    cols = st.columns(3)
    steps = [
        ("Data Upload", "Load CSV/XLSX oil & gas data and preview quality."),
        ("Model", "Configure ANN architecture, training, optimizer, and stopping."),
        ("Preprocess", "Choose target/features, auto-handle dates, missing data, scaling, and encoding."),
        ("Train", "Train a feedforward neural network on the prepared dataset."),
        ("Evaluate", "Inspect holdout metrics, error tables, and class/regression performance."),
        ("Predict", "Run single-row or batch inference on new field data."),
        ("Visualize", "Explore MATLAB-style plots: curves, confusion, residuals, ROC/PR, and trends."),
        ("Save/Load", "Export the full project and reload later without retraining."),
    ]
    for idx, (title, desc) in enumerate(steps):
        with cols[idx % 3]:
            st.markdown(f'<div class="soft-card"><h4>{title}</h4><p class="tiny">{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("### Recommended oil & gas examples")
    st.markdown(
        """
        - **Regression:** oil rate, gas rate, bottom-hole pressure, wellhead pressure, water cut, permeability proxy.
        - **Classification:** equipment state, well status, formation class, artificial-lift fault class.
        - **Typical inputs:** tubing pressure, choke size, temperature, gas-lift rate, depth, porosity, permeability, flowline pressure, date/time, well ID.
        """
    )

def page_data_upload():
    st.title("Data Upload")
    st.caption("Load a CSV/XLSX file containing oil & gas tabular data.")
    uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = read_uploaded_table(uploaded)
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

    plot_data_profile(df, st.session_state.config.get("target_column"))

def page_model():
    st.title("Model")
    cfg = st.session_state.config

    with st.form("model_form"):
        left, right = st.columns(2)
        with left:
            cfg["project_name"] = st.text_input("Project Name", value=cfg["project_name"])
            cfg["task_mode"] = st.selectbox("Task", ["Auto Detect", "Regression", "Classification"],
                                            index=["Auto Detect", "Regression", "Classification"].index(cfg["task_mode"]))
            cfg["hidden_layers"] = st.text_input("Hidden Layers (comma separated)", value=cfg["hidden_layers"])
            cfg["activation"] = st.selectbox("Activation", ["relu", "tanh", "elu", "selu"],
                                             index=["relu", "tanh", "elu", "selu"].index(cfg["activation"]))
            cfg["dropout"] = st.slider("Dropout", 0.0, 0.7, float(cfg["dropout"]), 0.05)
            cfg["batch_norm"] = st.checkbox("Use Batch Normalization", value=bool(cfg["batch_norm"]))
            cfg["l2_reg"] = st.number_input("L2 Regularization", min_value=0.0, max_value=1.0,
                                            value=float(cfg["l2_reg"]), step=0.0001, format="%.4f")
        with right:
            cfg["optimizer"] = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"],
                                            index=["adam", "sgd", "rmsprop"].index(cfg["optimizer"]))
            cfg["learning_rate"] = st.number_input("Learning Rate", min_value=1e-6, max_value=1.0,
                                                   value=float(cfg["learning_rate"]), step=0.0001, format="%.6f")
            cfg["epochs"] = st.number_input("Epochs", min_value=1, max_value=5000, value=int(cfg["epochs"]), step=1)
            cfg["batch_size"] = st.number_input("Batch Size", min_value=1, max_value=2048, value=int(cfg["batch_size"]), step=1)
            cfg["validation_split"] = st.slider("Validation Split", 0.05, 0.40, float(cfg["validation_split"]), 0.05)
            cfg["early_stopping"] = st.checkbox("Early Stopping", value=bool(cfg["early_stopping"]))
            cfg["patience"] = st.number_input("Patience", min_value=1, max_value=200, value=int(cfg["patience"]), step=1)
            cfg["use_class_weights"] = st.checkbox("Use Class Weights (classification)", value=bool(cfg["use_class_weights"]))
            cfg["threshold"] = st.slider("Binary Threshold", 0.05, 0.95, float(cfg["threshold"]), 0.01)
        submitted = st.form_submit_button("Save Model Settings", use_container_width=True)

    if submitted:
        st.session_state.config = cfg
        st.success("Model settings updated.")

    st.markdown("### Current architecture preview")
    layers = parse_hidden_layers(cfg["hidden_layers"])
    st.code(f"Input -> " + " -> ".join([f"Dense({x})" for x in layers]) + " -> Output", language="text")

def page_preprocess():
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
            prepared = prepare_dataset(df, cfg)
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

def page_train():
    st.title("Train")
    prepared = st.session_state.prepared_data
    if prepared is None:
        st.info("Prepare the dataset first.")
        return

    st.subheader("Training Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Task", prepared["task"].title())
    c2.metric("Target", prepared["target_column"])
    c3.metric("Train Samples", prepared["X_train"].shape[0])
    c4.metric("Test Samples", prepared["X_test"].shape[0])

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Input Features", prepared["X_train"].shape[1])
    c6.metric("Original Features", len(prepared["feature_columns_original"]))
    c7.metric("Numeric Columns", len(prepared["num_cols"]))
    c8.metric("Categorical Columns", len(prepared["cat_cols"]))

    with st.expander("Training configuration", expanded=False):
        config_rows = [
            {"setting": "Optimizer", "value": st.session_state.config["optimizer"]},
            {"setting": "Learning Rate", "value": st.session_state.config["learning_rate"]},
            {"setting": "Epochs", "value": st.session_state.config["epochs"]},
            {"setting": "Batch Size", "value": st.session_state.config["batch_size"]},
            {"setting": "Validation Split", "value": st.session_state.config["validation_split"]},
            {"setting": "Early Stopping", "value": st.session_state.config["early_stopping"]},
            {"setting": "Patience", "value": st.session_state.config["patience"]},
            {"setting": "Class Weights", "value": st.session_state.config["use_class_weights"]},
        ]
        st.dataframe(pd.DataFrame(config_rows), use_container_width=True, hide_index=True)

    if st.button("Start Training", type="primary", use_container_width=True):
        try:
            model, history, results = train_model(prepared, st.session_state.config)
            st.session_state.trained_model = model
            st.session_state.training_history = history
            st.session_state.results = results
            st.success("Model trained successfully.")
        except Exception as e:
            st.error(f"Training failed: {e}")

    if st.session_state.training_history:
        st.subheader("Latest training history")
        st.dataframe(pd.DataFrame(st.session_state.training_history), use_container_width=True)

def page_evaluate():
    st.title("Evaluate")
    results = st.session_state.results
    if not results:
        st.info("Train or load a model first.")
        return

    task = results["task"]
    metrics = results["metrics"]
    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        cols[i].metric(k.upper(), f"{v:.4f}")

    if task == "classification":
        class_names = results["class_names"]
        y_true = np.asarray(results["y_true"])
        y_pred = np.asarray(results["y_pred"])

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(make_confusion_fig(y_true, y_pred, class_names), clear_figure=True, use_container_width=True)
        with c2:
            st.pyplot(make_predicted_distribution_fig(y_pred), clear_figure=True, use_container_width=True)

        report_df = pd.DataFrame(results["classification_report"]).T
        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)

        if len(class_names) == 2:
            fig1, fig2 = make_binary_curve_figs(y_true, results.get("y_score"))
            c3, c4 = st.columns(2)
            with c3:
                st.pyplot(fig1, clear_figure=True, use_container_width=True)
            with c4:
                st.pyplot(fig2, clear_figure=True, use_container_width=True)
    else:
        y_true = np.asarray(results["y_true"])
        y_pred = np.asarray(results["y_pred"])
        error_df = pd.DataFrame({
            "actual": y_true,
            "predicted": y_pred,
            "residual": y_true - y_pred,
            "abs_error": np.abs(y_true - y_pred),
        })
        st.subheader("Prediction Errors")
        st.dataframe(error_df.head(300), use_container_width=True)

        fig1, fig2, fig3, fig4 = make_regression_figures(y_true, y_pred)
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

def page_predict():
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
        out = predict_with_pipeline(model, prepared, pred_df, threshold=float(st.session_state.config["threshold"]))
        st.dataframe(out, use_container_width=True)

    st.divider()
    st.subheader("Batch Prediction")
    uploaded = st.file_uploader("Upload prediction file", type=["csv", "xlsx", "xls"], key="pred_file")
    if uploaded is not None:
        try:
            pred_df = read_uploaded_table(uploaded)
            out = predict_with_pipeline(model, prepared, pred_df, threshold=float(st.session_state.config["threshold"]))
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

def page_visualize():
    st.title("Visualize")
    df = st.session_state.raw_df
    if df is not None:
        plot_data_profile(df, st.session_state.config.get("target_column"))

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
                    st.pyplot(make_training_curve_fig(history, metric), clear_figure=True, use_container_width=True)
    else:
        st.info("No training history available yet.")

    results = st.session_state.results
    if not results:
        return

    st.subheader("Model Performance Visuals")
    if results["task"] == "classification":
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(make_confusion_fig(results["y_true"], results["y_pred"], results["class_names"]), clear_figure=True, use_container_width=True)
        with c2:
            st.pyplot(make_predicted_distribution_fig(results["y_pred"]), clear_figure=True, use_container_width=True)

        if len(results["class_names"]) == 2:
            fig1, fig2 = make_binary_curve_figs(results["y_true"], results.get("y_score"))
            c3, c4 = st.columns(2)
            with c3:
                st.pyplot(fig1, clear_figure=True, use_container_width=True)
            with c4:
                st.pyplot(fig2, clear_figure=True, use_container_width=True)
    else:
        fig1, fig2, fig3, fig4 = make_regression_figures(np.asarray(results["y_true"]), np.asarray(results["y_pred"]))
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

def page_save_load():
    st.title("Save / Load")
    st.caption("Export the whole ANN project or reload a previous bundle.")

    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.trained_model is not None and st.session_state.prepared_data is not None:
            try:
                bundle_bytes = create_project_bundle_bytes()
                st.download_button(
                    "Download Project Bundle (.zip)",
                    data=bundle_bytes,
                    file_name=f"{safe_filename(st.session_state.config['project_name'])}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Failed to package project: {e}")
        else:
            st.info("Train or load a project to enable export.")

        if st.button("Save Project Locally", use_container_width=True):
            try:
                path = save_project_locally()
                st.success(f"Saved to {path}")
            except Exception as e:
                st.error(f"Local save failed: {e}")

    with c2:
        uploaded = st.file_uploader("Load bundle (.zip)", type=["zip"], key="load_bundle")
        if uploaded is not None:
            try:
                load_project_from_zip_bytes(uploaded.read())
                st.success("Project loaded successfully.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.subheader("Local Saved Bundles")
    local_files = sorted(PROJECTS_DIR.glob("*.zip"), reverse=True)
    if local_files:
        df = pd.DataFrame({
            "file": [p.name for p in local_files],
            "size_kb": [round(p.stat().st_size / 1024, 2) for p in local_files],
            "modified": [datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ", timespec="seconds") for p in local_files],
        })
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No local bundles found yet.")

# ----------------------------- #
# Workflow / project helpers
# ----------------------------- #
ANN_PAGES = [
    "Home",
    "Data Upload",
    "Model",
    "Preprocess",
    "Train",
    "Evaluate",
    "Predict",
    "Visualize",
    "Save/Load",
]


def _ann_go(page_name: str):
    st.session_state["page"] = page_name


def _ann_reset_state(target_page: str):
    preserved = {k: st.session_state[k] for k in list(st.session_state.keys()) if str(k).startswith("__unified_")}
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    for k, v in preserved.items():
        st.session_state[k] = v
    init_state()
    st.session_state["page"] = target_page
    st.rerun()


def _ann_bottom_nav(page_name: str):
    idx = ANN_PAGES.index(page_name) if page_name in ANN_PAGES else 0
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if idx == 0:
            st.button("⬅ Back to Home", use_container_width=True, disabled=True, key=f"ann_back_{page_name}")
        else:
            prev_page = ANN_PAGES[idx - 1]
            st.button(f"⬅ Back to {prev_page}", use_container_width=True, key=f"ann_back_{page_name}", on_click=_ann_go, args=(prev_page,))
    with c2:
        if idx >= len(ANN_PAGES) - 1:
            st.button("Continue ➜", use_container_width=True, disabled=True, key=f"ann_next_{page_name}")
        else:
            next_page = ANN_PAGES[idx + 1]
            st.button(f"Continue to {next_page} ➜", use_container_width=True, key=f"ann_next_{page_name}", on_click=_ann_go, args=(next_page,))


# ----------------------------- #
# Main
# ----------------------------- #
def main():
    inject_css()
    init_state()

    if "page" not in st.session_state or st.session_state["page"] not in ANN_PAGES:
        st.session_state["page"] = "Home"

    st.sidebar.title("Navigation")
    selected = st.sidebar.radio(
        "Go to",
        ANN_PAGES,
        index=ANN_PAGES.index(st.session_state["page"]),
    )
    st.session_state["page"] = selected

    st.sidebar.markdown("### Status")
    st.sidebar.markdown(f'<span class="status-pill">Dataset: {"Ready" if st.session_state.raw_df is not None else "Missing"}</span>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<span class="status-pill">Prepared: {"Yes" if st.session_state.prepared_data is not None else "No"}</span>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<span class="status-pill">Model: {"Trained/Loaded" if st.session_state.trained_model is not None else "None"}</span>', unsafe_allow_html=True)

    st.sidebar.write("---")
    if st.sidebar.button("➕ New Project", use_container_width=True):
        _ann_reset_state("Data Upload")
    if st.sidebar.button("🗑️ Clear Current Project", use_container_width=True):
        _ann_reset_state("Home")

    page = st.session_state["page"]
    if page == "Home":
        page_home()
    elif page == "Data Upload":
        page_data_upload()
    elif page == "Model":
        page_model()
    elif page == "Preprocess":
        page_preprocess()
    elif page == "Train":
        page_train()
    elif page == "Evaluate":
        page_evaluate()
    elif page == "Predict":
        page_predict()
    elif page == "Visualize":
        page_visualize()
    elif page == "Save/Load":
        page_save_load()

    _ann_bottom_nav(page)

if __name__ == "__main__":
    main()
'''

CNN_SOURCE = r'''
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

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

import streamlit as st
import matplotlib.pyplot as plt

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["legend.fontsize"] = 8

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    log_loss,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight


# -----------------------------
# Paths / constants
# -----------------------------
APP_TITLE = "Oil & Gas CNN Studio"
APP_DIR = Path(".oil_gas_cnn_studio")
DATA_DIR = APP_DIR / "datasets"
MODEL_DIR = APP_DIR / "saved_projects"
CACHE_DIR = APP_DIR / "cache"

for p in [APP_DIR, DATA_DIR, MODEL_DIR, CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Lazy TensorFlow import
# -----------------------------
_TF = None


def get_tf():
    global _TF
    if _TF is None:
        import tensorflow as tf
        _TF = tf
    return _TF


# -----------------------------
# Streamlit config / CSS
# -----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
        .main .block-container {padding-top: 1.0rem; padding-bottom: 1.2rem;}
        div[data-testid="stHorizontalBlock"] {gap: 0.55rem;}
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
            padding: 1.3rem 1.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 10px 30px rgba(2,6,23,0.16);
        }
        .card {
            background: #ffffff;
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(15,23,42,0.05);
            margin-bottom: 0.9rem;
        }
        .small-note {
            color: #475569;
            font-size: 0.92rem;
        }
        .status-chip {
            display:inline-block;
            padding:0.28rem 0.6rem;
            border-radius:999px;
            background:#eff6ff;
            border:1px solid #bfdbfe;
            color:#1d4ed8;
            font-size:0.85rem;
            margin-right:0.35rem;
            margin-bottom:0.35rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utility helpers
# -----------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf = get_tf()
    tf.random.set_seed(seed)


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


def fig_show(fig):
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def bytes_from_pil(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def init_state():
    defaults = {
        "page": "Home",
        "project_name": "oil_gas_cnn_project",
        "dataset_root": None,
        "dataset_df": None,
        "class_names": [],
        "data_summary": {},
        "model_config": {
            "seed": 42,
            "val_ratio": 0.2,
            "batch_size": 16,
            "image_size": 224,
            "color_mode": "RGB",
            "backbone": "MobileNetV2",
            "weights": "imagenet",
            "dense_units": 128,
            "dropout": 0.30,
            "label_smoothing": 0.0,
            "learning_rate": 1e-3,
            "epochs_stage1": 8,
            "epochs_stage2": 4,
            "fine_tune": True,
            "unfreeze_layers": 30,
            "fine_tune_lr": 1e-5,
            "optimizer": "Adam",
            "augmentation": {
                "flip": True,
                "rotation": 0.08,
                "zoom": 0.10,
                "contrast": 0.10,
                "brightness": 0.10,
            },
            "use_class_weights": True,
            "shuffle_buffer": 1024,
        },
        "trained_model": None,
        "history": None,
        "eval_artifacts": None,
        "feature_model_layer": None,
        "loaded_project_path": None,
        "training_complete": False,
        "last_uploaded_images": [],
        "invalid_dataset_files": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def status_bar():
    df = st.session_state.dataset_df
    classes = len(st.session_state.class_names) if st.session_state.class_names else 0
    trained = st.session_state.training_complete and st.session_state.trained_model is not None
    chips = [
        f"<span class='status-chip'>Project: {st.session_state.project_name}</span>",
        f"<span class='status-chip'>Samples: {0 if df is None else len(df)}</span>",
        f"<span class='status-chip'>Classes: {classes}</span>",
        f"<span class='status-chip'>Model Ready: {'Yes' if trained else 'No'}</span>",
    ]
    st.markdown("".join(chips), unsafe_allow_html=True)


def hero():
    st.markdown(
        f"""
        <div class="hero">
            <h2 style="margin:0 0 0.35rem 0;">{APP_TITLE}</h2>
            <div style="font-size:1rem; line-height:1.55;">
                Professional CNN workflow for oil & gas image classification:
                seismic slices, core photos, rock thin sections, corrosion images,
                facility inspections, and other 2D visual datasets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_rgb(img: Image.Image, mode: str = "RGB") -> Image.Image:
    if mode == "RGB":
        return img.convert("RGB")
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    arr3 = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr3)


def save_uploaded_zip(uploaded_file, project_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = DATA_DIR / f"{project_name}_{ts}"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / uploaded_file.name
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extract_dir = target_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif", ".gif", ".heic", ".heif"}


def open_uploaded_image(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        img = Image.open(uploaded_file)
        img.load()
        return img, None
    except UnidentifiedImageError:
        name = getattr(uploaded_file, "name", "file")
        suffix = Path(name).suffix.lower()
        if suffix in {".heic", ".heif"}:
            return None, (
                f"{name}: HEIC/HEIF was selected, but Pillow usually cannot decode it by default. "
                "Convert it to JPG/PNG first, or install pillow-heif in your environment."
            )
        return None, f"{name}: This file is not a readable image for Pillow. Use JPG, PNG, BMP, TIFF, WEBP, or GIF."
    except Exception as e:
        return None, f"{getattr(uploaded_file, 'name', 'file')}: {e}"


def validate_image_file(filepath):
    try:
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            img.convert("RGB")
        return True, None
    except UnidentifiedImageError:
        suffix = Path(filepath).suffix.lower()
        if suffix in {".heic", ".heif"}:
            return False, "HEIC/HEIF is not supported by default in this environment. Convert to JPG or PNG first."
        return False, "Unreadable image file."
    except Exception as e:
        return False, str(e)


def filter_valid_images(df: pd.DataFrame):
    if df is None or df.empty:
        return df, []
    keep_rows = []
    dropped = []
    for _, row in df.iterrows():
        ok, err = validate_image_file(row["filepath"])
        if ok:
            keep_rows.append(row.to_dict())
        else:
            dropped.append({
                "filepath": row["filepath"],
                "label": row["label"],
                "reason": err,
            })
    out_df = pd.DataFrame(keep_rows)
    if out_df.empty:
        return out_df, dropped
    out_df = out_df.reset_index(drop=True)
    return out_df, dropped


def list_image_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def infer_dataset_structure(extract_dir: Path) -> pd.DataFrame:
    files = list_image_files(extract_dir)
    if not files:
        raise ValueError("No image files found in the uploaded ZIP.")

    rows = []
    for fp in files:
        parts = [x for x in fp.relative_to(extract_dir).parts]
        lower_parts = [x.lower() for x in parts]

        split = None
        label = None

        if len(parts) >= 2 and lower_parts[0] in {"train", "training", "val", "valid", "validation", "test"}:
            split_map = {
                "train": "train", "training": "train",
                "val": "val", "valid": "val", "validation": "val",
                "test": "test",
            }
            split = split_map[lower_parts[0]]
            label = parts[1]
        else:
            # assume class folder is the first folder name
            if len(parts) < 2:
                continue
            label = parts[0]

        rows.append({"filepath": str(fp), "label": label, "split": split})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("ZIP structure is invalid. Expected class folders containing images.")

    df = df[df["label"].notna()].copy()
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= 2].index.tolist()
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    if df.empty:
        raise ValueError("Each class must contain at least 2 images.")

    return df


def finalize_splits(df: pd.DataFrame, val_ratio: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    if df["split"].notna().any():
        # keep explicit train/val/test splits; if only train exists, create val from train
        present = set(df["split"].dropna().unique().tolist())
        if "val" not in present and "train" in present:
            train_df = df[df["split"] == "train"].copy()
            if train_df["label"].value_counts().min() >= 2 and len(train_df) >= len(train_df["label"].unique()) * 2:
                tr_idx, va_idx = train_test_split(
                    train_df.index,
                    test_size=val_ratio,
                    random_state=seed,
                    stratify=train_df["label"],
                )
                df.loc[tr_idx, "split"] = "train"
                df.loc[va_idx, "split"] = "val"
        if "train" not in present:
            non_test = df[df["split"] != "test"].copy()
            tr_idx, va_idx = train_test_split(
                non_test.index,
                test_size=val_ratio,
                random_state=seed,
                stratify=non_test["label"],
            )
            df.loc[tr_idx, "split"] = "train"
            df.loc[va_idx, "split"] = "val"
    else:
        tr_idx, va_idx = train_test_split(
            df.index,
            test_size=val_ratio,
            random_state=seed,
            stratify=df["label"],
        )
        df.loc[tr_idx, "split"] = "train"
        df.loc[va_idx, "split"] = "val"

    return df.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict:
    dims = []
    broken = 0
    for fp in df["filepath"].sample(min(len(df), 100), random_state=42).tolist():
        try:
            with Image.open(fp) as img:
                dims.append(img.size)
        except Exception:
            broken += 1
    width_stats = [d[0] for d in dims] if dims else [0]
    height_stats = [d[1] for d in dims] if dims else [0]

    return {
        "total_images": int(len(df)),
        "classes": int(df["label"].nunique()),
        "splits": df["split"].value_counts().to_dict(),
        "class_counts": df["label"].value_counts().to_dict(),
        "width_min": int(np.min(width_stats)),
        "width_max": int(np.max(width_stats)),
        "height_min": int(np.min(height_stats)),
        "height_max": int(np.max(height_stats)),
        "sampled_broken_files": int(broken),
    }


def make_augmentation_layers(cfg):
    tf = get_tf()
    aug_cfg = cfg["augmentation"]
    layers = [
        tf.keras.layers.RandomFlip("horizontal") if aug_cfg["flip"] else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.RandomRotation(aug_cfg["rotation"]) if aug_cfg["rotation"] > 0 else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.RandomZoom(aug_cfg["zoom"]) if aug_cfg["zoom"] > 0 else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.RandomContrast(aug_cfg["contrast"]) if aug_cfg["contrast"] > 0 else tf.keras.layers.Lambda(lambda x: x),
    ]
    return tf.keras.Sequential(layers, name="augmentation")


def preprocess_input_layer(backbone: str):
    tf = get_tf()
    mapping = {
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
        "ResNet50": tf.keras.applications.resnet.preprocess_input,
    }
    fn = mapping[backbone]
    return tf.keras.layers.Lambda(fn, name="preprocess_input")


def build_backbone(backbone_name: str, input_shape, weights="imagenet"):
    tf = get_tf()
    if backbone_name == "MobileNetV2":
        return tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    if backbone_name == "EfficientNetB0":
        return tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    if backbone_name == "ResNet50":
        return tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    raise ValueError(f"Unsupported backbone: {backbone_name}")

def make_classification_loss(label_smoothing: float = 0.0):
    tf = get_tf()
    # Keep sparse integer labels for compatibility.
    # Many TensorFlow/Keras builds do not accept label_smoothing here.
    # So we intentionally ignore it instead of crashing the app.
    _ = label_smoothing
    return tf.keras.losses.SparseCategoricalCrossentropy()

def build_model(class_names, cfg):
    tf = get_tf()
    img_size = int(cfg["image_size"])
    input_shape = (img_size, img_size, 3)

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = make_augmentation_layers(cfg)(inputs)
    x = preprocess_input_layer(cfg["backbone"])(x)

    base_model = build_backbone(cfg["backbone"], input_shape, cfg["weights"])
    if cfg["weights"] == "imagenet":
        base_model.trainable = False
    else:
        base_model.trainable = True

    x = base_model(x, training=False)
    x = tf.keras.layers.Lambda(lambda t: t, name="backbone_feature_maps")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dense(cfg["dense_units"], activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(cfg["dropout"], name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="oil_gas_cnn")
    opt = make_optimizer(cfg["optimizer"], cfg["learning_rate"])
    metrics = [
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, len(class_names)), name="top_k_acc"),
    ]
    model.compile(
        optimizer=opt,
        loss=make_classification_loss(cfg.get("label_smoothing", 0.0)),
        metrics=metrics,
    )
    return model, base_model


def make_optimizer(name: str, lr: float):
    tf = get_tf()
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    if name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def _load_image_with_pil(path_bytes, image_size: int, color_mode: str):
    path = path_bytes.decode("utf-8")
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if color_mode == "Grayscale → 3-channel":
            img = ImageOps.grayscale(img).convert("RGB")
        else:
            img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return arr


def read_image_tf(path, label, image_size: int, color_mode: str):
    tf = get_tf()
    img = tf.numpy_function(
        func=lambda p: _load_image_with_pil(p, image_size, color_mode),
        inp=[path],
        Tout=tf.float32,
    )
    img.set_shape([image_size, image_size, 3])
    return img, label


def make_tf_dataset(paths, labels, cfg, training: bool):
    tf = get_tf()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), cfg["shuffle_buffer"]), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: read_image_tf(x, y, cfg["image_size"], cfg["color_mode"]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds


class EpochHistoryCallback:
    def __init__(self):
        self.rows = []

    def as_keras_callback(self):
        tf = get_tf()
        outer = self

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                row = {"epoch": int(epoch + 1), "lr": lr}
                for k, v in logs.items():
                    try:
                        row[k] = float(v)
                    except Exception:
                        pass
                outer.rows.append(row)

        return _CB()


def combine_histories(h1: dict, h2: dict):
    if not h1:
        return h2
    if not h2:
        return h1
    out = {}
    keys = sorted(set(h1.keys()) | set(h2.keys()))
    for k in keys:
        out[k] = list(h1.get(k, [])) + list(h2.get(k, []))
    return out


def compute_class_weights_from_labels(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def evaluate_model(model, val_df: pd.DataFrame, class_names, cfg):
    tf = get_tf()
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true = val_df["label"].map(label_to_idx).astype(int).to_numpy()
    paths = val_df["filepath"].tolist()
    ds = make_tf_dataset(paths, y_true, cfg, training=False)
    probs = model.predict(ds, verbose=0)
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision_macro": float(precision_score(y_true, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, preds, average="macro", zero_division=0)),
    }
    try:
        metrics["log_loss"] = float(log_loss(y_true, probs, labels=np.arange(len(class_names))))
    except Exception:
        metrics["log_loss"] = None

    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    roc_info = {}
    pr_info = {}

    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
        precision, recall, _ = precision_recall_curve(y_true, probs[:, 1])
        roc_info["binary"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
        pr_info["binary"] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "ap": float(average_precision_score(y_true, probs[:, 1])),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            metrics["roc_auc"] = None
    else:
        try:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_bin, probs, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc_ovr_macro"] = None
        try:
            metrics["ap_macro"] = float(average_precision_score(y_bin, probs, average="macro"))
        except Exception:
            metrics["ap_macro"] = None

        for i, cname in enumerate(class_names):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
                precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
                roc_info[cname] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc(fpr, tpr))}
                pr_info[cname] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "ap": float(average_precision_score(y_bin[:, i], probs[:, i])),
                }
            except Exception:
                pass

    cm = confusion_matrix(y_true, preds)
    report = classification_report(
        y_true,
        preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": preds,
        "y_prob": probs,
        "class_names": class_names,
        "confusion_matrix": cm,
        "classification_report": report,
        "val_paths": paths,
        "roc_info": roc_info,
        "pr_info": pr_info,
    }


def find_last_conv_layer_name(model):
    # We expose the backbone feature map explicitly to make Grad-CAM robust
    # even when the backbone itself is a nested sub-model.
    try:
        model.get_layer("backbone_feature_maps")
        return "backbone_feature_maps"
    except Exception:
        return None


def prepare_single_image(img: Image.Image, cfg):
    tf = get_tf()
    img = ensure_rgb(img, "RGB" if cfg["color_mode"] == "RGB" else "GRAY")
    img = img.resize((cfg["image_size"], cfg["image_size"]))
    arr = np.array(img).astype("float32")
    if cfg["color_mode"] == "Grayscale → 3-channel":
        gray = np.mean(arr, axis=-1, keepdims=True)
        arr = np.concatenate([gray, gray, gray], axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def render_uploaded_images_section(
    uploader_label="Upload image(s)",
    uploader_key="shared_image_uploader",
    show_title=True,
    show_gradcam=True,
):
    cfg = st.session_state.model_config
    model = st.session_state.trained_model
    class_names = st.session_state.class_names

    uploaded_imgs = st.file_uploader(
        uploader_label,
        accept_multiple_files=True,
        key=uploader_key,
        help="JPG, JPEG, PNG, BMP, TIFF, WEBP, and GIF work best. HEIC/HEIF usually require conversion unless extra codecs are installed.",
    )

    if not uploaded_imgs:
        return

    st.session_state.last_uploaded_images = [up.name for up in uploaded_imgs]

    if show_title:
        st.write("### Uploaded Images")

    if model is None or not st.session_state.training_complete or not class_names:
        st.info("Images uploaded successfully. Prediction is locked until you train or load a model.")
        preview_cols = st.columns(5, gap="small")
        for i, up in enumerate(uploaded_imgs):
            try:
                img, err = open_uploaded_image(up)
                if err:
                    preview_cols[i % len(preview_cols)].error(err)
                else:
                    preview_cols[i % len(preview_cols)].image(img, caption=up.name, use_container_width=True)
            except Exception as e:
                preview_cols[i % len(preview_cols)].error(f"{up.name}: {e}")
        return

    rows = []
    for up in uploaded_imgs:
        try:
            img, err = open_uploaded_image(up)
            if err:
                raise ValueError(err)
            clean_img = ensure_rgb(img, "RGB" if cfg["color_mode"] == "RGB" else "GRAY")
            arr = prepare_single_image(clean_img, cfg)
            probs = model.predict(arr, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            top_indices = np.argsort(probs)[::-1][:min(5, len(class_names))]
            heatmap = gradcam_heatmap(model, arr) if show_gradcam else None
            overlay = None
            if heatmap is not None:
                overlay = overlay_heatmap_on_image(
                    clean_img.resize((cfg["image_size"], cfg["image_size"])),
                    heatmap,
                )
            rows.append({
                "filename": up.name,
                "predicted_class": pred_name,
                "confidence": float(probs[pred_idx]),
                "top_k": {class_names[i]: float(probs[i]) for i in top_indices},
                "image": clean_img,
                "overlay": overlay,
            })
        except Exception as e:
            rows.append({
                "filename": up.name,
                "error": str(e),
            })

    valid_rows = [row for row in rows if "error" not in row]
    if valid_rows:
        summary_df = pd.DataFrame({
            "filename": [r["filename"] for r in valid_rows],
            "predicted_class": [r["predicted_class"] for r in valid_rows],
            "confidence": [r["confidence"] for r in valid_rows],
        }).sort_values("confidence", ascending=False)
        st.write("### Prediction Summary")
        st.dataframe(summary_df, use_container_width=True)

    for row in rows:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if "error" in row:
            st.error(f"{row['filename']}: {row['error']}")
            st.markdown("</div>", unsafe_allow_html=True)
            continue

        c1, c2 = st.columns([0.9, 1.1], gap="small")
        with c1:
            st.image(row["image"], caption=row["filename"], use_container_width=True)
        with c2:
            st.write(f"**Predicted Class:** {row['predicted_class']}")
            st.write(f"**Confidence:** {row['confidence']:.4f}")
            prob_df = pd.DataFrame({
                "class": list(row["top_k"].keys()),
                "probability": list(row["top_k"].values())
            })
            st.dataframe(prob_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(3.8, 2.0))
            ax.bar(prob_df["class"], prob_df["probability"])
            ax.set_title("Top Class Probabilities")
            ax.set_ylim(0, 1.0)
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig_show(fig)

            if row.get("overlay") is not None:
                st.write("**Grad-CAM**")
                st.image(row["overlay"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def gradcam_heatmap(model, img_array, layer_name=None):
    tf = get_tf()
    if layer_name is None:
        layer_name = find_last_conv_layer_name(model)
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


def overlay_heatmap_on_image(base_img: Image.Image, heatmap, alpha=0.40):
    if heatmap is None:
        return None
    heatmap = np.uint8(255 * heatmap)
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[:, :, :3]
    colored = Image.fromarray(np.uint8(colored * 255)).resize(base_img.size)
    base = base_img.convert("RGB")
    return Image.blend(base, colored, alpha=alpha)


def extract_features(model, paths, cfg, batch_limit=256):
    tf = get_tf()
    feature_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("gap").output)
    labels_dummy = np.zeros(len(paths), dtype=np.int32)
    ds = make_tf_dataset(paths[:batch_limit], labels_dummy[:batch_limit], cfg, training=False)
    feats = feature_model.predict(ds, verbose=0)
    return feats


def save_project(project_name: str):
    if st.session_state.trained_model is None:
        raise ValueError("No trained model available to save.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = MODEL_DIR / f"{project_name}_{ts}"
    project_dir.mkdir(parents=True, exist_ok=True)

    model_path = project_dir / "model.keras"
    meta_path = project_dir / "metadata.joblib"

    st.session_state.trained_model.save(model_path)

    meta = {
        "saved_at": now_str(),
        "project_name": project_name,
        "class_names": st.session_state.class_names,
        "model_config": st.session_state.model_config,
        "history": st.session_state.history,
        "data_summary": st.session_state.data_summary,
        "eval_artifacts": st.session_state.eval_artifacts,
    }
    joblib.dump(meta, meta_path)
    return project_dir


def load_project(project_dir: Path):
    tf = get_tf()
    model_path = project_dir / "model.keras"
    meta_path = project_dir / "metadata.joblib"
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Selected project is missing model.keras or metadata.joblib.")

    model = tf.keras.models.load_model(model_path)
    meta = joblib.load(meta_path)

    st.session_state.trained_model = model
    st.session_state.project_name = meta.get("project_name", project_dir.name)
    st.session_state.class_names = meta.get("class_names", [])
    st.session_state.model_config = meta.get("model_config", st.session_state.model_config)
    st.session_state.history = meta.get("history", None)
    st.session_state.data_summary = meta.get("data_summary", {})
    st.session_state.eval_artifacts = meta.get("eval_artifacts", None)
    st.session_state.training_complete = True
    st.session_state.loaded_project_path = str(project_dir)


def available_projects():
    dirs = [p for p in MODEL_DIR.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)


# -----------------------------
# Plotting
# -----------------------------
def plot_class_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_ylabel("Images")
    ax.set_xlabel("Class")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig_show(fig)


def plot_image_dimensions(df: pd.DataFrame):
    sample = df.sample(min(len(df), 150), random_state=42)
    widths, heights = [], []
    for fp in sample["filepath"]:
        try:
            with Image.open(fp) as img:
                widths.append(img.size[0])
                heights.append(img.size[1])
        except Exception:
            pass

    if not widths:
        st.warning("Could not inspect image dimensions.")
        return

    fig, ax = plt.subplots(figsize=(4.2, 2.5))
    ax.scatter(widths, heights, alpha=0.7)
    ax.set_title("Image Width vs Height")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.tight_layout()
    fig_show(fig)


def show_sample_gallery(df: pd.DataFrame, n_per_class=5):
    classes = df["label"].unique().tolist()
    for cname in classes:
        sub = df[df["label"] == cname].sample(min(n_per_class, (df["label"] == cname).sum()), random_state=42)
        st.markdown(f"**{cname}**")
        cols = st.columns(n_per_class, gap="small")
        for i, (_, row) in enumerate(sub.iterrows()):
            with Image.open(row["filepath"]) as img:
                cols[min(i, n_per_class - 1)].image(img, use_container_width=True, caption=Path(row["filepath"]).name)


def plot_training_curves(history: dict):
    if not history:
        st.warning("No training history available.")
        return

    df = pd.DataFrame(history)
    if df.empty:
        st.warning("Training history is empty.")
        return

    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        if "loss" in df:
            ax.plot(df.index + 1, df["loss"], label="Train Loss")
        if "val_loss" in df:
            ax.plot(df.index + 1, df["val_loss"], label="Val Loss")
            best_idx = int(df["val_loss"].idxmin()) + 1
            ax.axvline(best_idx, linestyle="--", alpha=0.7, label=f"Best Val Epoch {best_idx}")
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig_show(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4.0, 2.4))
        if "accuracy" in df:
            ax.plot(df.index + 1, df["accuracy"], label="Train Accuracy")
        if "val_accuracy" in df:
            ax.plot(df.index + 1, df["val_accuracy"], label="Val Accuracy")
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        fig_show(fig)

    if "lr" in df:
        fig, ax = plt.subplots(figsize=(4.2, 2.0))
        ax.plot(df.index + 1, df["lr"])
        ax.set_title("Learning Rate by Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning Rate")
        fig.tight_layout()
        fig_show(fig)


def plot_confusion_matrices(cm, class_names):
    cm = np.asarray(cm)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title("Confusion Matrix")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig_show(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(3.6, 3.2))
        im = ax.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title("Normalized Confusion Matrix")
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_yticklabels(class_names)
        thresh = cm_norm.max() / 2 if cm_norm.size else 0
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig_show(fig)


def plot_per_class_metrics(report_dict, class_names):
    rows = []
    for cname in class_names:
        if cname in report_dict:
            rows.append({
                "class": cname,
                "precision": report_dict[cname]["precision"],
                "recall": report_dict[cname]["recall"],
                "f1-score": report_dict[cname]["f1-score"],
                "support": report_dict[cname]["support"],
            })
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No per-class metrics available.")
        return

    st.dataframe(df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(4.8, 2.5))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w, df["precision"], width=w, label="Precision")
    ax.bar(x, df["recall"], width=w, label="Recall")
    ax.bar(x + w, df["f1-score"], width=w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class Metrics")
    ax.legend()
    fig.tight_layout()
    fig_show(fig)


def plot_roc_pr(eval_artifacts):
    roc_info = eval_artifacts["roc_info"]
    pr_info = eval_artifacts["pr_info"]
    class_names = eval_artifacts["class_names"]

    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        if "binary" in roc_info:
            d = roc_info["binary"]
            ax.plot(d["fpr"], d["tpr"], label=f"AUC = {d['auc']:.3f}")
        else:
            for cname in class_names:
                if cname in roc_info:
                    d = roc_info[cname]
                    ax.plot(d["fpr"], d["tpr"], label=f"{cname} ({d['auc']:.2f})")
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.6)
        ax.set_title("ROC Curve")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_show(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4.0, 2.5))
        if "binary" in pr_info:
            d = pr_info["binary"]
            ax.plot(d["recall"], d["precision"], label=f"AP = {d['ap']:.3f}")
        else:
            for cname in class_names:
                if cname in pr_info:
                    d = pr_info[cname]
                    ax.plot(d["recall"], d["precision"], label=f"{cname} ({d['ap']:.2f})")
        ax.set_title("Precision-Recall Curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig_show(fig)


def show_misclassified(eval_artifacts, max_items=12):
    y_true = np.asarray(eval_artifacts["y_true"])
    y_pred = np.asarray(eval_artifacts["y_pred"])
    probs = np.asarray(eval_artifacts["y_prob"])
    paths = eval_artifacts["val_paths"]
    class_names = eval_artifacts["class_names"]

    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        st.success("No misclassified validation samples found.")
        return

    selected = wrong[:max_items]
    cols = st.columns(6, gap="small")
    for i, idx in enumerate(selected):
        with Image.open(paths[idx]) as img:
            prob = probs[idx, y_pred[idx]]
            cols[i % len(cols)].image(
                img,
                caption=f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}\nConf: {prob:.3f}",
                use_container_width=True
            )


def plot_embedding_map(model, eval_artifacts, cfg):
    paths = eval_artifacts["val_paths"]
    y_true = np.asarray(eval_artifacts["y_true"])
    class_names = eval_artifacts["class_names"]
    feats = extract_features(model, paths, cfg, batch_limit=min(len(paths), 256))
    if feats.shape[0] < 3:
        st.warning("Not enough samples to compute PCA embedding.")
        return
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(4.3, 3.0))
    for i, cname in enumerate(class_names):
        mask = y_true[:len(emb)] == i
        ax.scatter(emb[mask, 0], emb[mask, 1], label=cname, alpha=0.75)
    ax.set_title("Validation Feature Embedding (PCA)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_show(fig)


# -----------------------------
# Pages
# -----------------------------
def page_home():
    hero()
    status_bar()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What this app is for")
    st.write(
        """
        Use this for oil & gas image classification tasks such as:
        - seismic facies or fault-image classes
        - core / thin-section rock imagery
        - corrosion / coating / crack / anomaly inspection
        - equipment condition images
        - refinery / field visual inspection categories
        """
    )
    st.write(
        """
        Do **not** use this for plain tabular well-log spreadsheets. CNNs are for images or structured 2D grids.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Required Input", "ZIP image dataset")
    c2.metric("Task Type", "Image classification")
    c3.metric("Deployment Output", "Saved .keras model")

    with st.expander("Expected ZIP structure", expanded=True):
        st.code(
            """Option A (explicit split)
dataset.zip
├── train
│   ├── class_1
│   │   ├── img001.jpg
│   │   └── ...
│   └── class_2
├── val
│   ├── class_1
│   └── class_2

Option B (single folder, app creates validation split)
dataset.zip
├── class_1
│   ├── img001.jpg
│   └── ...
└── class_2
    ├── img101.jpg
    └── ...
""",
            language="text"
        )


def page_data_upload():
    hero()
    status_bar()
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
                seed_everything(cfg["seed"])
                extract_dir = save_uploaded_zip(uploaded_zip, st.session_state.project_name)
                df = infer_dataset_structure(extract_dir)
                df, dropped_files = filter_valid_images(df)
                if df.empty:
                    raise ValueError("No valid readable images were found after validation. Remove corrupted, unsupported, or HEIC/HEIF files and try again.")
                counts = df["label"].value_counts()
                valid_labels = counts[counts >= 2].index.tolist()
                df = df[df["label"].isin(valid_labels)].reset_index(drop=True)
                if df.empty:
                    raise ValueError("After removing invalid images, each class must still contain at least 2 readable images.")
                df = finalize_splits(df, cfg["val_ratio"], cfg["seed"])
                st.session_state.dataset_root = str(extract_dir)
                st.session_state.dataset_df = df
                st.session_state.class_names = sorted(df["label"].unique().tolist())
                st.session_state.data_summary = dataset_summary(df)
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

    st.divider()
    st.write("### Upload Single or Multiple Images")
    st.caption("This uploader is separate from the ZIP dataset uploader. Use it for one picture or many pictures. If a trained model exists, prediction runs immediately. Otherwise the app just previews the images.")
    render_uploaded_images_section(
        uploader_label="Upload image files",
        uploader_key="data_page_image_uploader",
        show_title=False,
        show_gradcam=True,
    )

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
            plot_class_distribution(df)
        with col2:
            plot_image_dimensions(df)

        st.write("### Sample Gallery")
        show_sample_gallery(df, n_per_class=5)

def page_model():
    hero()
    status_bar()
    st.subheader("Model")

    cfg = st.session_state.model_config
    c1, c2, c3 = st.columns(3)
    with c1:
        cfg["backbone"] = st.selectbox("Backbone", ["MobileNetV2", "EfficientNetB0", "ResNet50"],
                                       index=["MobileNetV2", "EfficientNetB0", "ResNet50"].index(cfg["backbone"]))
        cfg["weights"] = st.selectbox("Initial Weights", ["imagenet", None], index=0 if cfg["weights"] == "imagenet" else 1)
        cfg["optimizer"] = st.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"],
                                        index=["Adam", "RMSprop", "SGD"].index(cfg["optimizer"]))
    with c2:
        cfg["dense_units"] = st.select_slider("Dense Units", options=[64, 128, 256, 512], value=cfg["dense_units"])
        cfg["dropout"] = st.slider("Dropout", 0.0, 0.7, float(cfg["dropout"]), 0.05)
        cfg["label_smoothing"] = 0.0
        st.caption("Label smoothing is disabled to keep sparse-label training compatible with your TensorFlow/Keras build.")
    with c3:
        cfg["learning_rate"] = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=cfg["learning_rate"])
        cfg["fine_tune"] = st.checkbox("Enable Fine-Tuning", value=cfg["fine_tune"])
        cfg["use_class_weights"] = st.checkbox("Use Class Weights", value=cfg["use_class_weights"])

    if cfg["fine_tune"]:
        c4, c5 = st.columns(2)
        with c4:
            cfg["unfreeze_layers"] = st.slider("Unfreeze Last N Layers", 5, 120, int(cfg["unfreeze_layers"]), 5)
        with c5:
            cfg["fine_tune_lr"] = st.select_slider("Fine-Tune LR", options=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4], value=cfg["fine_tune_lr"])

    st.info("Recommended starting point for small oil & gas image datasets: MobileNetV2 + ImageNet weights + fine-tuning.")


def page_preprocess():
    hero()
    status_bar()
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


def page_train():
    hero()
    status_bar()
    st.subheader("Train")

    df = st.session_state.dataset_df
    cfg = st.session_state.model_config
    if df is None:
        st.warning("Upload a dataset first.")
        return

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    if train_df.empty or val_df.empty:
        st.error("Training or validation split is empty.")
        return

    c1, c2 = st.columns(2)
    with c1:
        cfg["epochs_stage1"] = st.slider("Stage 1 Epochs (Frozen Backbone)", 1, 40, int(cfg["epochs_stage1"]))
    with c2:
        cfg["epochs_stage2"] = st.slider("Stage 2 Epochs (Fine-Tuning)", 0, 30, int(cfg["epochs_stage2"]))

    if st.button("Start Training", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            seed_everything(cfg["seed"])
            class_names = sorted(st.session_state.class_names)
            label_to_idx = {c: i for i, c in enumerate(class_names)}

            y_train = train_df["label"].map(label_to_idx).astype(int).to_numpy()
            y_val = val_df["label"].map(label_to_idx).astype(int).to_numpy()

            train_ds = make_tf_dataset(train_df["filepath"].tolist(), y_train, cfg, training=True)
            val_ds = make_tf_dataset(val_df["filepath"].tolist(), y_val, cfg, training=False)

            model, base_model = build_model(class_names, cfg)

            tf = get_tf()
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
            ]
            hist_cb_1 = EpochHistoryCallback()
            callbacks_all_1 = callbacks + [hist_cb_1.as_keras_callback()]

            class_weights = compute_class_weights_from_labels(y_train) if cfg["use_class_weights"] else None

            hist1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=cfg["epochs_stage1"],
                callbacks=callbacks_all_1,
                verbose=1,
                class_weight=class_weights,
            )

            merged_history = hist1.history.copy()
            epoch_rows = hist_cb_1.rows.copy()

            if cfg["fine_tune"] and cfg["epochs_stage2"] > 0 and cfg["weights"] == "imagenet":
                base_model.trainable = True
                for layer in base_model.layers[:-cfg["unfreeze_layers"]]:
                    layer.trainable = False

                model.compile(
                    optimizer=make_optimizer(cfg["optimizer"], cfg["fine_tune_lr"]),
                    loss=make_classification_loss(cfg.get("label_smoothing", 0.0)),
                    metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, len(class_names)), name="top_k_acc")],
                )

                hist_cb_2 = EpochHistoryCallback()
                callbacks_all_2 = callbacks + [hist_cb_2.as_keras_callback()]

                hist2 = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=cfg["epochs_stage1"] + cfg["epochs_stage2"],
                    initial_epoch=cfg["epochs_stage1"],
                    callbacks=callbacks_all_2,
                    verbose=1,
                    class_weight=class_weights,
                )
                merged_history = combine_histories(hist1.history, hist2.history)
                epoch_rows.extend(hist_cb_2.rows)

            if epoch_rows:
                hist_df = pd.DataFrame(epoch_rows).sort_values("epoch")
                merged_history["lr"] = hist_df["lr"].tolist()

            st.session_state.trained_model = model
            st.session_state.history = merged_history
            st.session_state.eval_artifacts = evaluate_model(model, val_df, class_names, cfg)
            st.session_state.training_complete = True

        st.success("Training complete.")

    if st.session_state.history is not None:
        plot_training_curves(st.session_state.history)


def page_evaluate():
    hero()
    status_bar()
    st.subheader("Evaluate")

    if not st.session_state.training_complete or st.session_state.eval_artifacts is None:
        st.warning("Train or load a model first.")
        return

    ev = st.session_state.eval_artifacts
    metrics = ev["metrics"]

    cols = st.columns(6)
    items = [
        ("Accuracy", metrics.get("accuracy")),
        ("Balanced Acc", metrics.get("balanced_accuracy")),
        ("Precision Macro", metrics.get("precision_macro")),
        ("Recall Macro", metrics.get("recall_macro")),
        ("F1 Macro", metrics.get("f1_macro")),
        ("Log Loss", metrics.get("log_loss")),
    ]
    for col, (name, val) in zip(cols, items):
        col.metric(name, "-" if val is None else f"{val:.4f}")

    plot_confusion_matrices(ev["confusion_matrix"], ev["class_names"])
    plot_per_class_metrics(ev["classification_report"], ev["class_names"])

    extra = []
    if metrics.get("roc_auc") is not None:
        extra.append(("ROC AUC", metrics["roc_auc"]))
    if metrics.get("roc_auc_ovr_macro") is not None:
        extra.append(("ROC AUC OVR Macro", metrics["roc_auc_ovr_macro"]))
    if metrics.get("ap_macro") is not None:
        extra.append(("AP Macro", metrics["ap_macro"]))

    if extra:
        cols = st.columns(len(extra))
        for c, (n, v) in zip(cols, extra):
            c.metric(n, f"{v:.4f}")

    plot_roc_pr(ev)
    st.write("### Misclassified Validation Samples")
    show_misclassified(ev, max_items=12)


def page_predict():
    hero()
    status_bar()
    st.subheader("Predict")

    if st.session_state.trained_model is None or not st.session_state.training_complete:
        st.warning("No trained model is loaded yet. You can still upload one or many images now, but prediction will start only after training or loading a model.")

    render_uploaded_images_section(
        uploader_label="Upload one image or many images for prediction",
        uploader_key="predict_page_image_uploader",
        show_title=False,
        show_gradcam=True,
    )


def page_visualize():
    hero()
    status_bar()
    st.subheader("Visualize")

    df = st.session_state.dataset_df
    if df is None:
        st.warning("Upload a dataset first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Data", "Training", "Validation", "Explainability"])

    with tab1:
        plot_class_distribution(df)
        plot_image_dimensions(df)
        show_sample_gallery(df, n_per_class=5)

    with tab2:
        plot_training_curves(st.session_state.history)

    with tab3:
        if st.session_state.eval_artifacts is None:
            st.info("Train the model to unlock validation plots.")
        else:
            ev = st.session_state.eval_artifacts
            plot_confusion_matrices(ev["confusion_matrix"], ev["class_names"])
            plot_roc_pr(ev)
            plot_per_class_metrics(ev["classification_report"], ev["class_names"])
            plot_embedding_map(st.session_state.trained_model, ev, st.session_state.model_config)
            show_misclassified(ev, max_items=12)

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
                    arr = prepare_single_image(img, cfg)
                    probs = model.predict(arr, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    st.write(f"**Predicted:** {st.session_state.class_names[pred_idx]} ({probs[pred_idx]:.4f})")
                    c1, c2 = st.columns(2, gap="small")
                    c1.image(img, caption="Original", use_container_width=True)
                    heatmap = gradcam_heatmap(model, arr)
                    overlay = overlay_heatmap_on_image(img.resize((cfg["image_size"], cfg["image_size"])), heatmap)
                    if overlay is not None:
                        c2.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            else:
                up = st.file_uploader("Upload image", key="viz_custom_img", help="JPG/PNG/BMP/TIFF/WEBP/GIF work best. HEIC usually needs conversion.")
                if up is not None:
                    img, err = open_uploaded_image(up)
                    if err:
                        st.error(err)
                        return
                    arr = prepare_single_image(img, cfg)
                    probs = model.predict(arr, verbose=0)[0]
                    pred_idx = int(np.argmax(probs))
                    st.write(f"**Predicted:** {st.session_state.class_names[pred_idx]} ({probs[pred_idx]:.4f})")
                    c1, c2 = st.columns(2, gap="small")
                    c1.image(img, caption="Original", use_container_width=True)
                    heatmap = gradcam_heatmap(model, arr)
                    overlay = overlay_heatmap_on_image(img.resize((cfg["image_size"], cfg["image_size"])), heatmap)
                    if overlay is not None:
                        c2.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)


def page_save_load():
    hero()
    status_bar()
    st.subheader("Save / Load")

    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.write("### Save Current Project")
        st.session_state.project_name = st.text_input("Save Name", value=st.session_state.project_name, key="save_name_input")
        if st.button("Save Project", type="primary", use_container_width=True):
            try:
                project_dir = save_project(st.session_state.project_name)
                st.success(f"Project saved to: {project_dir}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with col2:
        st.write("### Load Existing Project")
        projects = available_projects()
        if not projects:
            st.info("No saved projects found yet.")
        else:
            choice = st.selectbox("Saved Projects", projects, format_func=lambda p: p.name)
            if st.button("Load Selected Project", use_container_width=True):
                try:
                    load_project(choice)
                    st.success(f"Loaded: {choice.name}")
                except Exception as e:
                    st.error(f"Load failed: {e}")

    st.write("### Saved Project Inventory")
    rows = []
    for p in available_projects():
        rows.append({
            "project_dir": str(p),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "has_model": (p / "model.keras").exists(),
            "has_metadata": (p / "metadata.joblib").exists(),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# -----------------------------
# Workflow / project helpers
# -----------------------------
CNN_PAGES = [
    "Home",
    "Data Upload",
    "Model",
    "Preprocess",
    "Train",
    "Evaluate",
    "Predict",
    "Visualize",
    "Save/Load",
]


def _cnn_go(page_name: str):
    st.session_state.page = page_name


def _cnn_reset_state(target_page: str):
    preserved = {k: st.session_state[k] for k in list(st.session_state.keys()) if str(k).startswith("__unified_")}
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    for k, v in preserved.items():
        st.session_state[k] = v
    init_state()
    st.session_state.page = target_page
    st.rerun()


def _cnn_bottom_nav(page_name: str):
    idx = CNN_PAGES.index(page_name) if page_name in CNN_PAGES else 0
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if idx == 0:
            st.button("⬅ Back to Home", use_container_width=True, disabled=True, key=f"cnn_back_{page_name}")
        else:
            prev_page = CNN_PAGES[idx - 1]
            st.button(f"⬅ Back to {prev_page}", use_container_width=True, key=f"cnn_back_{page_name}", on_click=_cnn_go, args=(prev_page,))
    with c2:
        if idx >= len(CNN_PAGES) - 1:
            st.button("Continue ➜", use_container_width=True, disabled=True, key=f"cnn_next_{page_name}")
        else:
            next_page = CNN_PAGES[idx + 1]
            st.button(f"Continue to {next_page} ➜", use_container_width=True, key=f"cnn_next_{page_name}", on_click=_cnn_go, args=(next_page,))


# -----------------------------
# Main app
# -----------------------------
def main():
    init_state()

    pages = CNN_PAGES
    if st.session_state.page not in pages:
        st.session_state.page = "Home"

    st.sidebar.title(APP_TITLE)
    selected = st.sidebar.radio("Navigation", pages, index=pages.index(st.session_state.page) if st.session_state.page in pages else 0)
    st.session_state.page = selected

    st.sidebar.write("---")
    if st.sidebar.button("➕ New Project", use_container_width=True):
        _cnn_reset_state("Data Upload")
    if st.sidebar.button("🗑️ Clear Current Project", use_container_width=True):
        _cnn_reset_state("Home")

    page_map = {
        "Home": page_home,
        "Data Upload": page_data_upload,
        "Model": page_model,
        "Preprocess": page_preprocess,
        "Train": page_train,
        "Evaluate": page_evaluate,
        "Predict": page_predict,
        "Visualize": page_visualize,
        "Save/Load": page_save_load,
    }

    page_map[selected]()
    _cnn_bottom_nav(selected)


if __name__ == "__main__":
    main()
'''

LSTM_SOURCE = r'''
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# -----------------------------
# Lazy TensorFlow loader
# -----------------------------
_TF_CACHE: Dict[str, object] = {}


def get_tf():
    if "tf" not in _TF_CACHE:
        import tensorflow as tf

        _TF_CACHE["tf"] = tf
    return _TF_CACHE["tf"]


# -----------------------------
# App constants
# -----------------------------
APP_TITLE = "Oil & Gas LSTM Studio"
APP_SUBTITLE = "MATLAB-style workflow for multivariate energy forecasting with LSTM"
PAGES = [
    "Home",
    "Data Upload",
    "Model",
    "Preprocess",
    "Train",
    "Evaluate",
    "Predict",
    "Visualize",
    "Save/Load",
]

BASE_DIR = Path(".oil_gas_lstm_studio")
PROJECTS_DIR = BASE_DIR / "projects"
TEMP_IMPORT_DIR = BASE_DIR / "imports"
for p in [BASE_DIR, PROJECTS_DIR, TEMP_IMPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Styling
# -----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1350px;}
        .hero {
            padding: 1.2rem 1.4rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1e293b 100%);
            color: white;
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.35);
            margin-bottom: 1rem;
        }
        .hero h1 {margin: 0 0 0.35rem 0; font-size: 2rem;}
        .hero p {margin: 0; color: rgba(255,255,255,0.84);}
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.94));
            border: 1px solid rgba(15,23,42,0.08);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(15,23,42,0.07);
            margin-bottom: 1rem;
        }
        .pill {
            display: inline-block;
            padding: 0.28rem 0.65rem;
            border-radius: 999px;
            background: #e2e8f0;
            color: #0f172a;
            font-size: 0.82rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .section-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }
        .small-muted {color: #475569; font-size: 0.9rem;}
        .good {color: #0f766e; font-weight: 700;}
        .bad {color: #b91c1c; font-weight: 700;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# State helpers
# -----------------------------
def default_config() -> Dict:
    return {
        "date_col": None,
        "feature_cols": [],
        "target_cols": [],
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


def init_state() -> None:
    defaults = {
        "page": "Home",
        "raw_df": None,
        "processed": None,
        "training": None,
        "prediction": None,
        "config": default_config(),
        "loaded_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_after_data_change() -> None:
    st.session_state["processed"] = None
    st.session_state["training"] = None
    st.session_state["prediction"] = None


# -----------------------------
# Utility functions
# -----------------------------
def read_uploaded_data(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")


def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    name_hits = [c for c in df.columns if any(x in str(c).lower() for x in ["date", "time", "timestamp"])]
    for col in name_hits + list(df.columns):
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.70:
                return col
        except Exception:
            continue
    return None


def get_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or []
    numeric = []
    for col in df.columns:
        if col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() >= 0.70:
            numeric.append(col)
    return numeric


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def frequency_from_dates(dates: pd.Series) -> str:
    if dates is None or len(dates) < 3:
        return "Unknown"
    diffs = dates.sort_values().diff().dropna().dt.total_seconds() / 86400.0
    if len(diffs) == 0:
        return "Unknown"
    med = float(np.median(diffs))
    if med <= 2:
        return "Daily"
    if med <= 10:
        return "Weekly"
    if med <= 40:
        return "Monthly"
    return "Irregular"


def annualization_factor(freq_name: str) -> int:
    if freq_name == "Daily":
        return 252
    if freq_name == "Weekly":
        return 52
    if freq_name == "Monthly":
        return 12
    return 252


def make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError("Unknown scaler")


def apply_missing(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    if method == "drop":
        return out.dropna()
    if method == "ffill":
        return out.ffill()
    if method == "bfill":
        return out.bfill()
    if method == "ffill_bfill":
        return out.ffill().bfill()
    if method == "interpolate":
        return out.interpolate(method="linear").ffill().bfill()
    if method == "median_impute":
        imp = SimpleImputer(strategy="median")
        out[out.columns] = imp.fit_transform(out)
        return out
    return out


def apply_transform(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "raw":
        return df.copy()
    if mode == "pct_change":
        return df.pct_change()
    if mode == "log_return":
        safe = df.replace(0, np.nan)
        return np.log(safe / safe.shift(1))
    raise ValueError("Unknown transform mode")


def clip_quantiles(df: pd.DataFrame, low_q: float, high_q: float) -> pd.DataFrame:
    low = df.quantile(low_q)
    high = df.quantile(high_q)
    return df.clip(low, high, axis=1)


def create_sequences(
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


def inverse_3d(flat_pred: np.ndarray, scaler, horizon: int, num_targets: int) -> np.ndarray:
    arr = flat_pred.reshape(-1, horizon, num_targets)
    inv = np.empty_like(arr, dtype=np.float64)
    for step in range(horizon):
        inv[:, step, :] = scaler.inverse_transform(arr[:, step, :])
    return inv


def flatten_y(y_3d: np.ndarray) -> np.ndarray:
    return y_3d.reshape(y_3d.shape[0], -1)


def model_summary_text(model) -> str:
    lines: List[str] = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)


def preprocess_dataset(df: pd.DataFrame, cfg: Dict) -> Dict:
    if df is None or df.empty:
        raise ValueError("No dataset loaded.")
    date_col = cfg["date_col"]
    feature_cols = cfg["feature_cols"]
    target_cols = cfg["target_cols"]
    lookback = int(cfg["lookback"])
    horizon = int(cfg["horizon"])

    if not feature_cols:
        raise ValueError("Select at least one feature column.")
    if not target_cols:
        raise ValueError("Select at least one target column.")
    if lookback < 2:
        raise ValueError("Lookback must be at least 2.")
    if horizon < 1:
        raise ValueError("Horizon must be at least 1.")

    work = df.copy()
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        dates = work[date_col].copy()
    else:
        dates = pd.Series(pd.date_range("2000-01-01", periods=len(work), freq="D"))

    keep_cols = list(dict.fromkeys(feature_cols + target_cols))
    numeric_df = coerce_numeric(work[keep_cols], keep_cols)

    if cfg["resample_rule"] != "None" and date_col:
        tmp = pd.concat([dates, numeric_df], axis=1).set_index(date_col)
        numeric_df = tmp.resample(cfg["resample_rule"]).mean()
        dates = pd.Series(numeric_df.index)
        numeric_df = numeric_df.reset_index(drop=True)

    numeric_df = apply_missing(numeric_df, cfg["missing_method"])
    if cfg["clip_outliers"]:
        numeric_df = clip_quantiles(numeric_df, cfg["clip_low_q"], cfg["clip_high_q"])

    transformed = apply_transform(numeric_df, cfg["transform_mode"])
    transformed = apply_missing(transformed, cfg["missing_method"])
    valid_mask = transformed.notna().all(axis=1)
    transformed = transformed.loc[valid_mask].reset_index(drop=True)
    numeric_df = numeric_df.loc[valid_mask].reset_index(drop=True)
    dates = dates.loc[valid_mask].reset_index(drop=True)

    feature_frame = transformed[feature_cols].copy()
    target_frame = transformed[target_cols].copy()

    n = len(transformed)
    if n <= lookback + horizon + 10:
        raise ValueError("Dataset is too small after preprocessing for the chosen lookback and horizon.")

    train_end_idx = int(n * cfg["train_frac"])
    val_end_idx = int(n * (cfg["train_frac"] + cfg["val_frac"]))
    train_end_idx = max(train_end_idx, lookback + horizon)
    val_end_idx = max(val_end_idx, train_end_idx + 1)
    val_end_idx = min(val_end_idx, n - 1)

    feature_scaler = make_scaler(cfg["scale_method"])
    target_scaler = make_scaler(cfg["scale_method"])
    feature_scaler.fit(feature_frame.iloc[:train_end_idx])
    target_scaler.fit(target_frame.iloc[:train_end_idx])

    X_scaled = feature_scaler.transform(feature_frame)
    y_scaled = target_scaler.transform(target_frame)

    seq = create_sequences(
        X_scaled,
        y_scaled,
        dates.to_numpy(),
        lookback,
        horizon,
        train_end_idx=train_end_idx,
        val_end_idx=val_end_idx,
    )

    X_seq = seq["X_seq"]
    y_seq = seq["y_seq"]
    tags = seq["split_tags"]

    X_train = X_seq[tags == "train"]
    X_val = X_seq[tags == "val"]
    X_test = X_seq[tags == "test"]
    y_train = y_seq[tags == "train"]
    y_val = y_seq[tags == "val"]
    y_test = y_seq[tags == "test"]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Split produced an empty train/validation/test segment. Adjust split fractions or reduce lookback.")

    return {
        "dates": dates,
        "original_numeric": numeric_df,
        "transformed": transformed,
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "X_seq": X_seq,
        "y_seq": y_seq,
        "target_times": seq["target_times"],
        "split_tags": tags,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "y_train_flat": flatten_y(y_train),
        "y_val_flat": flatten_y(y_val),
        "y_test_flat": flatten_y(y_test),
        "train_end_idx": train_end_idx,
        "val_end_idx": val_end_idx,
        "freq_name": frequency_from_dates(pd.to_datetime(dates, errors="coerce")),
        "config_snapshot": dict(cfg),
    }


def build_model(input_shape: Tuple[int, int], num_targets: int, horizon: int, cfg: Dict):
    tf = get_tf()
    tf.keras.backend.clear_session()
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(int(cfg["seed"]))
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
    out = tf.keras.layers.Dense(horizon * num_targets, name="forecast")(x)
    model = tf.keras.Model(inp, out)

    loss_name = cfg["loss"]
    if loss_name == "huber":
        loss_fn = tf.keras.losses.Huber()
    else:
        loss_fn = loss_name

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(cfg["learning_rate"])),
        loss=loss_fn,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"), tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


class StreamlitTrainProgressCallback:
    def __init__(self, progress_bar, info_box, total_epochs: int):
        self.progress_bar = progress_bar
        self.info_box = info_box
        self.total_epochs = max(1, int(total_epochs))

    def as_callback(self):
        tf = get_tf()
        outer = self

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                frac = (epoch + 1) / outer.total_epochs
                outer.progress_bar.progress(min(frac, 1.0))
                parts = [f"Epoch {epoch + 1}/{outer.total_epochs}"]
                for key in ["loss", "mae", "rmse", "val_loss", "val_mae", "val_rmse"]:
                    if key in logs:
                        try:
                            parts.append(f"{key}: {float(logs[key]):.4f}")
                        except Exception:
                            pass
                outer.info_box.markdown(" | ".join(parts))

        return _CB()


def train_model(processed: Dict, cfg: Dict) -> Dict:
    tf = get_tf()
    num_targets = len(processed["target_cols"])
    horizon = int(cfg["horizon"])
    model = build_model(processed["X_train"].shape[1:], num_targets, horizon, cfg)

    ckpt_dir = tempfile.mkdtemp(prefix="oilgas_lstm_ckpt_")
    ckpt_path = str(Path(ckpt_dir) / "best_model.keras")

    progress_bar = st.progress(0)
    info_box = st.empty()

    callbacks = [
        StreamlitTrainProgressCallback(progress_bar, info_box, int(cfg["epochs"])).as_callback(),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(cfg["patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, int(cfg["patience"]) // 2),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        processed["X_train"],
        processed["y_train_flat"],
        validation_data=(processed["X_val"], processed["y_val_flat"]),
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        verbose=0,
        callbacks=callbacks,
        shuffle=True,
    )

    progress_bar.progress(1.0)
    info_box.success("Training finished.")

    val_pred_flat = model.predict(processed["X_val"], verbose=0)
    test_pred_flat = model.predict(processed["X_test"], verbose=0)
    train_pred_flat = model.predict(processed["X_train"], verbose=0)

    preds = {
        "train": inverse_3d(train_pred_flat, processed["target_scaler"], horizon, num_targets),
        "val": inverse_3d(val_pred_flat, processed["target_scaler"], horizon, num_targets),
        "test": inverse_3d(test_pred_flat, processed["target_scaler"], horizon, num_targets),
    }
    actuals = {
        "train": inverse_3d(processed["y_train_flat"], processed["target_scaler"], horizon, num_targets),
        "val": inverse_3d(processed["y_val_flat"], processed["target_scaler"], horizon, num_targets),
        "test": inverse_3d(processed["y_test_flat"], processed["target_scaler"], horizon, num_targets),
    }

    split_tags = processed["split_tags"]
    split_times = {
        "train": processed["target_times"][split_tags == "train"],
        "val": processed["target_times"][split_tags == "val"],
        "test": processed["target_times"][split_tags == "test"],
    }

    metrics_tables = {
        split: compute_metrics_table(actuals[split], preds[split], processed["target_cols"], horizon)
        for split in ["train", "val", "test"]
    }

    return {
        "model": model,
        "history": history.history,
        "predictions": preds,
        "actuals": actuals,
        "times": split_times,
        "metrics": metrics_tables,
        "model_summary": model_summary_text(model),
    }


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    out = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(out) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean() * 100.0)


def compute_metrics_table(actual_3d: np.ndarray, pred_3d: np.ndarray, target_cols: List[str], horizon: int) -> pd.DataFrame:
    rows = []
    for h in range(horizon):
        for i, col in enumerate(target_cols):
            yt = actual_3d[:, h, i]
            yp = pred_3d[:, h, i]
            rows.append(
                {
                    "horizon_step": h + 1,
                    "target": col,
                    "MAE": mean_absolute_error(yt, yp),
                    "RMSE": math.sqrt(mean_squared_error(yt, yp)),
                    "MAPE_%": safe_mape(yt, yp),
                    "R2": r2_score(yt, yp),
                    "Directional_Accuracy_%": directional_accuracy(yt, yp),
                }
            )
    return pd.DataFrame(rows)


def backtest_strategies(actual_returns: np.ndarray, predicted_returns: np.ndarray, dates: np.ndarray, target_cols: List[str], freq_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    actual = np.asarray(actual_returns, dtype=float)
    pred = np.asarray(predicted_returns, dtype=float)
    if actual.ndim != 2 or pred.ndim != 2:
        raise ValueError("Backtest inputs must be 2D arrays [samples, targets].")

    n_samples, n_assets = actual.shape
    eq = np.full(n_assets, 1.0 / n_assets)

    strat_returns = {
        "EqualWeight": np.full(n_samples, np.nan),
        "BestBet": np.full(n_samples, np.nan),
        "LongOnly": np.full(n_samples, np.nan),
        "LongShort": np.full(n_samples, np.nan),
    }

    for t in range(n_samples):
        p = pred[t]
        a = actual[t]
        strat_returns["EqualWeight"][t] = float(np.dot(eq, a))

        best_idx = int(np.argmax(p))
        w_best = np.zeros(n_assets)
        w_best[best_idx] = 1.0
        strat_returns["BestBet"][t] = float(np.dot(w_best, a))

        positive = np.where(p > 0, p, 0)
        if positive.sum() > 0:
            w_long = positive / positive.sum()
        else:
            w_long = np.zeros(n_assets)
        strat_returns["LongOnly"][t] = float(np.dot(w_long, a))

        abs_sum = np.abs(p).sum()
        if abs_sum > 0:
            w_ls = p / abs_sum
        else:
            w_ls = np.zeros(n_assets)
        strat_returns["LongShort"][t] = float(np.dot(w_ls, a))

    returns_df = pd.DataFrame(strat_returns, index=pd.to_datetime(dates))
    equity_df = (1.0 + returns_df.fillna(0)).cumprod()

    ann = annualization_factor(freq_name)
    summary_rows = []
    for col in returns_df.columns:
        r = returns_df[col].dropna()
        if len(r) == 0:
            continue
        equity = (1.0 + r).cumprod()
        total_return = float(equity.iloc[-1] - 1.0)
        years = max(len(r) / ann, 1e-8)
        cagr = float(equity.iloc[-1] ** (1 / years) - 1.0)
        vol = float(r.std(ddof=1) * np.sqrt(ann)) if len(r) > 1 else np.nan
        sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(ann)) if len(r) > 1 and r.std(ddof=1) > 0 else np.nan
        drawdown = equity / equity.cummax() - 1.0
        max_dd = float(drawdown.min())
        hit = float((r > 0).mean() * 100.0)
        summary_rows.append(
            {
                "strategy": col,
                "Total_Return_%": total_return * 100.0,
                "CAGR_%": cagr * 100.0,
                "Volatility_%": vol * 100.0 if pd.notna(vol) else np.nan,
                "Sharpe": sharpe,
                "Max_Drawdown_%": max_dd * 100.0,
                "Positive_Periods_%": hit,
            }
        )

    return equity_df, pd.DataFrame(summary_rows)


def returns_for_backtest(transform_mode: str, values: np.ndarray) -> np.ndarray:
    if transform_mode == "pct_change":
        return values
    if transform_mode == "log_return":
        return np.expm1(values)
    raise ValueError("Backtest is only valid for pct_change or log_return mode.")


def make_bundle_bytes() -> bytes:
    processed = st.session_state.get("processed")
    training = st.session_state.get("training")
    cfg = st.session_state.get("config")
    if processed is None or training is None:
        raise ValueError("Nothing to save. Preprocess and train first.")

    with tempfile.TemporaryDirectory(prefix="oilgas_bundle_") as tmpdir:
        tmp = Path(tmpdir)
        model_path = tmp / "model.keras"
        state_path = tmp / "state.joblib"
        cfg_path = tmp / "config.json"
        metrics_path = tmp / "test_metrics.csv"

        training["model"].save(model_path)
        payload = {
            "processed": {k: v for k, v in processed.items() if k not in {"feature_scaler", "target_scaler"}},
            "feature_scaler": processed["feature_scaler"],
            "target_scaler": processed["target_scaler"],
            "training": {k: v for k, v in training.items() if k != "model"},
            "raw_df": st.session_state.get("raw_df"),
            "loaded_filename": st.session_state.get("loaded_filename"),
        }
        joblib.dump(payload, state_path)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, default=str)
        training["metrics"]["test"].to_csv(metrics_path, index=False)

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path, arcname="model.keras")
            zf.write(state_path, arcname="state.joblib")
            zf.write(cfg_path, arcname="config.json")
            zf.write(metrics_path, arcname="test_metrics.csv")
        return mem.getvalue()


def load_bundle(uploaded_zip) -> None:
    data = uploaded_zip.getvalue()
    import_dir = Path(tempfile.mkdtemp(prefix="oilgas_import_", dir=str(TEMP_IMPORT_DIR)))
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(import_dir)

    tf = get_tf()
    model = tf.keras.models.load_model(import_dir / "model.keras")
    payload = joblib.load(import_dir / "state.joblib")
    with open(import_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    processed = payload["processed"]
    processed["feature_scaler"] = payload["feature_scaler"]
    processed["target_scaler"] = payload["target_scaler"]
    training = payload["training"]
    training["model"] = model

    st.session_state["config"] = cfg
    st.session_state["processed"] = processed
    st.session_state["training"] = training
    st.session_state["raw_df"] = payload["raw_df"]
    st.session_state["loaded_filename"] = payload["loaded_filename"]
    st.session_state["prediction"] = None


# -----------------------------
# Plot helpers
# -----------------------------
PLOT_W = 5.6
PLOT_H = 2.7

def fig_line(df: pd.DataFrame, title: str, ylabel: str = "Value"):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col), linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncols=min(4, len(df.columns)), fontsize=8)
    fig.tight_layout()
    return fig


def fig_training_history(history: Dict):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    if "loss" in history:
        ax.plot(history["loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Validation Loss", linewidth=2)
        best_epoch = int(np.argmin(history["val_loss"]))
        ax.axvline(best_epoch, linestyle="--", linewidth=1)
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_actual_vs_predicted(actual: np.ndarray, pred: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.plot(actual, label="Actual", linewidth=1.8)
    ax.plot(pred, label="Predicted", linewidth=1.5)
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_distribution(actual: np.ndarray, pred: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    bins = 40
    ax.hist(actual, bins=bins, alpha=0.6, label="Actual")
    ax.hist(pred, bins=bins, alpha=0.6, label="Predicted")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_scatter(actual: np.ndarray, pred: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.scatter(actual, pred, alpha=0.55)
    low = min(np.min(actual), np.min(pred))
    high = max(np.max(actual), np.max(pred))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_residuals_vs_pred(pred: np.ndarray, residuals: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.scatter(pred, residuals, alpha=0.55)
    ax.axhline(0, linewidth=1.2)
    std = np.std(residuals)
    ax.axhline(2 * std, linestyle="--", linewidth=1)
    ax.axhline(-2 * std, linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_residual_hist(residuals: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.hist(residuals, bins=40, alpha=0.75)
    ax.axvline(np.mean(residuals), linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_qq(residuals: np.ndarray, title: str):
    fig = plt.figure(figsize=(PLOT_W, PLOT_H))
    ax = fig.add_subplot(111)
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_rolling_rmse(actual: np.ndarray, pred: np.ndarray, title: str, window: int = 30):
    errors = (actual - pred) ** 2
    rolling = pd.Series(errors).rolling(window=window, min_periods=max(5, window // 3)).mean().pow(0.5)
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.plot(rolling.values, linewidth=1.7)
    ax.set_title(title)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Rolling RMSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def fig_correlation_heatmap(df: pd.DataFrame, title: str):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def fig_missing_bars(df: pd.DataFrame, title: str):
    miss = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.bar(miss.index.astype(str), miss.values)
    ax.set_title(title)
    ax.set_ylabel("Missing Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25, axis="y")
    return fig


def fig_split_overview(dates: pd.Series, train_end_idx: int, val_end_idx: int, title: str):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    ax.plot(dates, np.ones(len(dates)), alpha=0)
    ax.axvspan(dates.iloc[0], dates.iloc[train_end_idx - 1], alpha=0.25)
    ax.axvspan(dates.iloc[train_end_idx], dates.iloc[val_end_idx - 1], alpha=0.25)
    ax.axvspan(dates.iloc[val_end_idx], dates.iloc[-1], alpha=0.25)
    ax.set_yticks([])
    ax.set_title(title)
    return fig


def fig_equity_curves(equity_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    for col in equity_df.columns:
        ax.plot(equity_df.index, equity_df[col], label=col, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel("Equity (Start = 1.0)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_drawdowns(equity_df: pd.DataFrame, title: str):
    dd = equity_df / equity_df.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(PLOT_W, PLOT_H))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], label=col, linewidth=1.4)
    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# -----------------------------
# UI components
# -----------------------------
def hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">{title}</div>
            <div class="small-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_status_bar():
    raw_df = st.session_state.get("raw_df")
    processed = st.session_state.get("processed")
    training = st.session_state.get("training")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Loaded", 0 if raw_df is None else len(raw_df))
    c2.metric("Features Selected", len(st.session_state["config"].get("feature_cols", [])))
    c3.metric("Targets Selected", len(st.session_state["config"].get("target_cols", [])))
    c4.metric("Model Status", "Ready" if training is not None else ("Preprocessed" if processed is not None else "Not Trained"))


# -----------------------------
# Pages
# -----------------------------
def page_home():
    hero(APP_TITLE, APP_SUBTITLE)
    top_status_bar()
    st.markdown(
        """
        <div class="card">
            <div class="section-title">What this app actually does</div>
            <div class="small-muted">
                It takes multivariate oil and gas time-series data, cleans it, converts it to raw values or returns,
                builds rolling LSTM sequences, trains a stacked LSTM forecaster, evaluates forecast quality, produces
                deep diagnostics, and optionally simulates signal-driven strategy curves when you model returns.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        info_card("Workflow", "Home → Data Upload → Model → Preprocess → Train → Evaluate → Predict → Visualize → Save/Load")
    with c2:
        info_card("Best Use Case", "Daily or weekly oil, gas, refined product, or energy-market datasets with a date column and multiple numeric series")
    with c3:
        info_card("Built For", "WTI, Brent, Natural Gas, Heating Oil, Diesel, LPG, or any structured commodity time series with enough history")

    st.subheader("Expected input structure")
    st.code(
        "date,WTI,Brent,NaturalGas,HeatingOil,Diesel\n"
        "2020-01-01,61.17,66.25,2.12,1.98,2.01\n"
        "2020-01-02,62.02,67.10,2.18,2.01,2.03\n"
        "...",
        language="text",
    )

    st.subheader("Hard truth before you start")
    st.write(
        "If your data is short, dirty, inconsistent, or mostly noise, the model will not magically become good because it is an LSTM. "
        "Most bad forecasting projects are bad data projects wearing a neural-network costume."
    )


def page_data_upload():
    hero("Data Upload", "Load oil and gas market data and inspect it before you poison the model with garbage.")
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"], key="main_uploader")

    if uploaded is not None:
        try:
            df = read_uploaded_data(uploaded)
            st.session_state["raw_df"] = df
            st.session_state["loaded_filename"] = uploaded.name
            reset_after_data_change()
            detected_date = detect_date_column(df)
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

    detected_date = detect_date_column(df)
    numeric_cols = get_numeric_columns(df, exclude=[detected_date] if detected_date else [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Columns", df.shape[1])
    c3.metric("Detected Date Column", detected_date or "None")
    c4.metric("Numeric-like Columns", len(numeric_cols))

    left, right = st.columns([1.2, 1])
    with left:
        st.pyplot(fig_missing_bars(df, "Missing Values by Column"), use_container_width=True)
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


def page_model():
    hero("Model", "Set forecasting targets, sequence design, and network architecture.")
    df = st.session_state.get("raw_df")
    if df is None:
        st.info("Load data first.")
        return

    cfg = st.session_state["config"]
    guessed_date = detect_date_column(df)
    numeric_cols = get_numeric_columns(df, exclude=[guessed_date] if guessed_date else [])
    if guessed_date and (cfg["date_col"] is None or cfg["date_col"] not in df.columns):
        cfg["date_col"] = guessed_date
    if not cfg["feature_cols"]:
        cfg["feature_cols"] = numeric_cols[:]
    if not cfg["target_cols"]:
        cfg["target_cols"] = numeric_cols[: min(4, len(numeric_cols))]

    with st.form("model_form"):
        c1, c2 = st.columns(2)
        with c1:
            cfg["project_name"] = st.text_input("Project name", value=cfg["project_name"])
            cfg["date_col"] = st.selectbox("Date column", options=[None] + list(df.columns), index=([None] + list(df.columns)).index(cfg["date_col"]) if cfg["date_col"] in df.columns else 0)
            cfg["feature_cols"] = st.multiselect("Feature columns", options=numeric_cols, default=[c for c in cfg["feature_cols"] if c in numeric_cols])
            cfg["target_cols"] = st.multiselect("Target columns", options=numeric_cols, default=[c for c in cfg["target_cols"] if c in numeric_cols])
            cfg["lookback"] = st.number_input("Lookback window", min_value=2, max_value=365, value=int(cfg["lookback"]), step=1)
            cfg["horizon"] = st.number_input("Forecast horizon", min_value=1, max_value=30, value=int(cfg["horizon"]), step=1)
        with c2:
            cfg["lstm_units_1"] = st.number_input("LSTM units 1", min_value=8, max_value=512, value=int(cfg["lstm_units_1"]), step=8)
            cfg["lstm_units_2"] = st.number_input("LSTM units 2", min_value=8, max_value=512, value=int(cfg["lstm_units_2"]), step=8)
            cfg["lstm_units_3"] = st.number_input("LSTM units 3", min_value=8, max_value=512, value=int(cfg["lstm_units_3"]), step=8)
            cfg["dense_units"] = st.number_input("Dense units", min_value=8, max_value=512, value=int(cfg["dense_units"]), step=8)
            cfg["dropout"] = st.slider("Dropout", min_value=0.0, max_value=0.8, value=float(cfg["dropout"]), step=0.05)
            cfg["loss"] = st.selectbox("Loss", ["mse", "mae", "huber"], index=["mse", "mae", "huber"].index(cfg["loss"]))
            cfg["seed"] = st.number_input("Random seed", min_value=0, max_value=999999, value=int(cfg["seed"]), step=1)
        submitted = st.form_submit_button("Save Model Settings")

    if submitted:
        reset_after_data_change()
        st.success("Model configuration saved.")

    st.markdown("### Model Design Summary")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Features", len(cfg["feature_cols"]))
    s2.metric("Targets", len(cfg["target_cols"]))
    s3.metric("Lookback", int(cfg["lookback"]))
    s4.metric("Horizon", int(cfg["horizon"]))

    a, b = st.columns(2)
    with a:
        st.markdown("**Selected Inputs**")
        st.dataframe(pd.DataFrame({"Feature Columns": cfg["feature_cols"]}), use_container_width=True, hide_index=True, height=260)
    with b:
        st.markdown("**Prediction Targets**")
        st.dataframe(pd.DataFrame({"Target Columns": cfg["target_cols"]}), use_container_width=True, hide_index=True, height=260)

    with st.expander("Full configuration", expanded=False):
        st.code(json.dumps(cfg, indent=2, default=str), language="json")


def page_preprocess():
    hero("Preprocess", "Clean the series, transform it, split it sequentially, and build LSTM windows.")
    df = st.session_state.get("raw_df")
    if df is None:
        st.info("Load data first.")
        return

    cfg = st.session_state["config"]
    with st.form("preprocess_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            cfg["transform_mode"] = st.selectbox(
                "Target/data transform",
                ["raw", "pct_change", "log_return"],
                index=["raw", "pct_change", "log_return"].index(cfg["transform_mode"]),
            )
            cfg["missing_method"] = st.selectbox(
                "Missing value handling",
                ["ffill_bfill", "interpolate", "drop", "ffill", "bfill", "median_impute"],
                index=["ffill_bfill", "interpolate", "drop", "ffill", "bfill", "median_impute"].index(cfg["missing_method"]),
            )
            cfg["resample_rule"] = st.selectbox("Resample rule", ["None", "D", "W", "M"], index=["None", "D", "W", "M"].index(cfg["resample_rule"]))
        with c2:
            cfg["scale_method"] = st.selectbox("Scaling", ["standard", "minmax", "robust"], index=["standard", "minmax", "robust"].index(cfg["scale_method"]))
            cfg["train_frac"] = st.slider("Train fraction", min_value=0.50, max_value=0.85, value=float(cfg["train_frac"]), step=0.05)
            cfg["val_frac"] = st.slider("Validation fraction", min_value=0.05, max_value=0.30, value=float(cfg["val_frac"]), step=0.05)
        with c3:
            cfg["clip_outliers"] = st.checkbox("Clip outliers by quantile", value=bool(cfg["clip_outliers"]))
            cfg["clip_low_q"] = st.number_input("Lower quantile", min_value=0.0, max_value=0.20, value=float(cfg["clip_low_q"]), step=0.005)
            cfg["clip_high_q"] = st.number_input("Upper quantile", min_value=0.80, max_value=1.0, value=float(cfg["clip_high_q"]), step=0.005)
        submitted = st.form_submit_button("Run Preprocessing")

    if submitted:
        try:
            if cfg["train_frac"] + cfg["val_frac"] >= 0.95:
                raise ValueError("Train fraction + validation fraction must leave room for a test set.")
            processed = preprocess_dataset(df, cfg)
            st.session_state["processed"] = processed
            st.session_state["training"] = None
            st.session_state["prediction"] = None
            st.success("Preprocessing complete.")
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")

    processed = st.session_state.get("processed")
    if processed is None:
        st.info("Configure preprocessing and click Run Preprocessing.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed Rows", len(processed["transformed"]))
    c2.metric("Train Sequences", len(processed["X_train"]))
    c3.metric("Validation Sequences", len(processed["X_val"]))
    c4.metric("Test Sequences", len(processed["X_test"]))

    st.markdown("**Sequential Data Split Overview**")
    split_left, split_center, split_right = st.columns([1, 2.4, 1])
    with split_center:
        st.pyplot(
            fig_split_overview(processed["dates"], processed["train_end_idx"], processed["val_end_idx"], "Sequential Data Split Overview"),
            use_container_width=True,
        )

    st.write("**Transformed data preview**")
    preview = processed["transformed"].copy()
    preview.insert(0, "date", processed["dates"])
    st.dataframe(preview.head(20), use_container_width=True)


def page_train():
    hero("Train", "Train the LSTM and keep the best checkpoint instead of praying at a random epoch.")
    processed = st.session_state.get("processed")
    if processed is None:
        st.info("Preprocess data first.")
        return

    cfg = st.session_state["config"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cfg["epochs"] = st.number_input("Epochs", min_value=1, max_value=1000, value=int(cfg["epochs"]), step=1)
    with c2:
        cfg["batch_size"] = st.number_input("Batch size", min_value=1, max_value=1024, value=int(cfg["batch_size"]), step=1)
    with c3:
        cfg["learning_rate"] = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=float(cfg["learning_rate"]), format="%.6f")
    with c4:
        cfg["patience"] = st.number_input("Early stopping patience", min_value=1, max_value=200, value=int(cfg["patience"]), step=1)

    train_now = st.button("Start Training", type="primary")
    if train_now:
        try:
            with st.spinner("Training model..."):
                training = train_model(processed, cfg)
            st.session_state["training"] = training
            st.success("Training complete.")
        except Exception as e:
            st.error(f"Training failed: {e}")

    training = st.session_state.get("training")
    if training is None:
        st.info("Press Start Training.")
        return

    st.markdown("### Training Summary")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Train Sequences", len(processed["X_train"]))
    t2.metric("Validation Sequences", len(processed["X_val"]))
    t3.metric("Test Sequences", len(processed["X_test"]))
    t4.metric("Input Shape", f"{processed['X_train'].shape[1]} × {processed['X_train'].shape[2]}")

    t5, t6, t7, t8 = st.columns(4)
    t5.metric("Epochs", int(cfg["epochs"]))
    t6.metric("Batch Size", int(cfg["batch_size"]))
    t7.metric("Learning Rate", f"{float(cfg['learning_rate']):.6f}")
    t8.metric("Patience", int(cfg["patience"]))

    hist_left, hist_center, hist_right = st.columns([1, 2.4, 1])
    with hist_center:
        st.pyplot(fig_training_history(training["history"]), use_container_width=True)

    with st.container(border=True):
        st.markdown("**Model summary**")
        st.code(training["model_summary"], language="text")

    st.write("**Validation metrics**")
    st.dataframe(training["metrics"]["val"], use_container_width=True, hide_index=True)


def page_evaluate():
    hero("Evaluate", "Check whether the model learned signal or just learned how to disappoint you more elegantly.")
    training = st.session_state.get("training")
    processed = st.session_state.get("processed")
    if training is None or processed is None:
        st.info("Train the model first.")
        return

    cfg = st.session_state["config"]
    split = st.selectbox("Evaluation split", ["train", "val", "test"], index=2)
    target = st.selectbox("Target", processed["target_cols"], index=0)
    horizon_step = st.selectbox("Horizon step", list(range(1, int(cfg["horizon"]) + 1)), index=0)

    idx = processed["target_cols"].index(target)
    h = horizon_step - 1
    actual = training["actuals"][split][:, h, idx]
    pred = training["predictions"][split][:, h, idx]
    residuals = actual - pred

    row = training["metrics"][split]
    row = row[(row["target"] == target) & (row["horizon_step"] == horizon_step)].iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"{row['MAE']:.6f}")
    c2.metric("RMSE", f"{row['RMSE']:.6f}")
    c3.metric("MAPE %", f"{row['MAPE_%']:.2f}")
    c4.metric("R²", f"{row['R2']:.4f}")
    c5.metric("Directional Accuracy %", f"{row['Directional_Accuracy_%']:.2f}")

    st.dataframe(training["metrics"][split], use_container_width=True, hide_index=True)

    a, b = st.columns(2)
    with a:
        st.pyplot(fig_actual_vs_predicted(actual, pred, f"{split.title()} | {target} | Step {horizon_step}: Actual vs Predicted"), use_container_width=True)
    with b:
        st.pyplot(fig_distribution(actual, pred, f"{split.title()} | {target} | Actual vs Predicted Distribution"), use_container_width=True)

    c, d = st.columns(2)
    with c:
        st.pyplot(fig_scatter(actual, pred, f"{split.title()} | {target} | Scatter"), use_container_width=True)
    with d:
        st.pyplot(fig_residual_hist(residuals, f"{split.title()} | {target} | Residual Histogram"), use_container_width=True)

    e, f = st.columns(2)
    with e:
        st.pyplot(fig_residuals_vs_pred(pred, residuals, f"{split.title()} | {target} | Residuals vs Predicted"), use_container_width=True)
    with f:
        st.pyplot(fig_qq(residuals, f"{split.title()} | {target} | Residual QQ Plot"), use_container_width=True)


def prepare_prediction_input(df_new: pd.DataFrame, processed: Dict, cfg: Dict) -> Tuple[np.ndarray, pd.DataFrame]:
    feature_cols = processed["feature_cols"]
    target_cols = processed["target_cols"]
    date_col = cfg["date_col"]
    work = df_new.copy()

    if date_col and date_col in work.columns:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    needed = list(dict.fromkeys(feature_cols + target_cols))
    missing = [c for c in needed if c not in work.columns]
    if missing:
        raise ValueError(f"New data is missing required columns: {missing}")

    numeric = coerce_numeric(work[needed], needed)
    if cfg["resample_rule"] != "None" and date_col and date_col in work.columns:
        tmp = pd.concat([work[[date_col]], numeric], axis=1).set_index(date_col)
        numeric = tmp.resample(cfg["resample_rule"]).mean().reset_index(drop=True)

    numeric = apply_missing(numeric, cfg["missing_method"])
    if cfg["clip_outliers"]:
        numeric = clip_quantiles(numeric, cfg["clip_low_q"], cfg["clip_high_q"])
    transformed = apply_transform(numeric, cfg["transform_mode"])
    transformed = apply_missing(transformed, cfg["missing_method"]).dropna().reset_index(drop=True)

    if len(transformed) < int(cfg["lookback"]):
        raise ValueError("New data does not have enough rows for the configured lookback window.")

    X = transformed[feature_cols].copy()
    X_scaled = processed["feature_scaler"].transform(X)
    last_window = X_scaled[-int(cfg["lookback"]) :]
    return last_window[np.newaxis, :, :].astype(np.float32), numeric


def implied_prices_from_returns(last_prices: pd.Series, pred_values: np.ndarray, transform_mode: str, target_cols: List[str]) -> pd.DataFrame:
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


def page_predict():
    hero("Predict", "Generate forecasts from the latest window or from a new dataset that matches the training schema.")
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
                pred_df = read_uploaded_data(new_file)
            except Exception as e:
                st.error(f"Prediction file load failed: {e}")
                return
    else:
        raw_df = st.session_state.get("raw_df")
        pred_df = raw_df.copy() if raw_df is not None else None

    if pred_df is None:
        st.info("Provide a source dataset for prediction.")
        return

    if st.button("Run Forecast", type="primary"):
        try:
            X_input, numeric = prepare_prediction_input(pred_df, processed, cfg)
            pred_flat = training["model"].predict(X_input, verbose=0)
            pred_inv = inverse_3d(pred_flat, processed["target_scaler"], int(cfg["horizon"]), len(processed["target_cols"]))[0]
            pred_table = pd.DataFrame(pred_inv, columns=processed["target_cols"], index=[f"step_{i+1}" for i in range(int(cfg["horizon"]))])

            implied_prices = None
            if cfg["transform_mode"] in {"pct_change", "log_return", "raw"}:
                last_prices = numeric.iloc[-1]
                implied_prices = implied_prices_from_returns(last_prices, pred_inv, cfg["transform_mode"], processed["target_cols"])

            st.session_state["prediction"] = {
                "forecast_table": pred_table,
                "implied_prices": implied_prices,
            }
            st.success("Forecast complete.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    pred_state = st.session_state.get("prediction")
    if pred_state is None:
        return

    st.write("**Forecast in modeled units**")
    st.dataframe(pred_state["forecast_table"], use_container_width=True)
    if pred_state["implied_prices"] is not None:
        st.write("**Implied future prices**")
        st.dataframe(pred_state["implied_prices"], use_container_width=True)


def page_visualize():
    hero("Visualize", "All the plots you actually need, not decorative nonsense.")
    training = st.session_state.get("training")
    processed = st.session_state.get("processed")
    if training is None or processed is None:
        st.info("Train the model first.")
        return

    cfg = st.session_state["config"]

    st.subheader("Data and training overview")
    ov1, ov2 = st.columns(2)
    if isinstance(processed["dates"].iloc[0], pd.Timestamp):
        frame = processed["transformed"][processed["target_cols"]].copy()
        frame.index = processed["dates"]
        with ov1:
            st.pyplot(fig_line(frame, "Target Series After Transformation", ylabel="Modeled Value"), use_container_width=True)
    else:
        with ov1:
            st.info("Date-based overview is only shown when a valid date column exists.")
    with ov2:
        st.pyplot(fig_training_history(training["history"]), use_container_width=True)

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
        st.pyplot(fig_actual_vs_predicted(actual, pred, f"{split.title()} | {target} | Overlay"), use_container_width=True)
    with b:
        st.pyplot(fig_scatter(actual, pred, f"{split.title()} | {target} | Scatter"), use_container_width=True)

    c, d = st.columns(2)
    with c:
        st.pyplot(fig_residuals_vs_pred(pred, residuals, f"{split.title()} | {target} | Residuals vs Predicted"), use_container_width=True)
    with d:
        st.pyplot(fig_rolling_rmse(actual, pred, f"{split.title()} | {target} | Rolling RMSE", window=max(10, int(cfg['lookback']))), use_container_width=True)

    e, f = st.columns(2)
    with e:
        st.pyplot(fig_distribution(actual, pred, f"{split.title()} | {target} | Distribution"), use_container_width=True)
    with f:
        st.pyplot(fig_qq(residuals, f"{split.title()} | {target} | QQ Plot"), use_container_width=True)

    if cfg["transform_mode"] in {"pct_change", "log_return"} and int(cfg["horizon"]) >= 1:
        st.subheader("Signal-driven strategy simulation (MATLAB-style idea, simplified)")
        actual_step1 = training["actuals"]["test"][:, 0, :]
        pred_step1 = training["predictions"]["test"][:, 0, :]
        actual_ret = returns_for_backtest(cfg["transform_mode"], actual_step1)
        pred_ret = returns_for_backtest(cfg["transform_mode"], pred_step1)
        test_dates = np.array([pd.to_datetime(x[0]) for x in training["times"]["test"]])
        equity_df, summary_df = backtest_strategies(actual_ret, pred_ret, test_dates, processed["target_cols"], processed["freq_name"])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.pyplot(fig_equity_curves(equity_df, "Strategy Equity Curves"), use_container_width=True)
        st.pyplot(fig_drawdowns(equity_df, "Strategy Drawdowns"), use_container_width=True)
    else:
        st.info("Backtest-style strategy plots are only shown when the modeled targets are returns or log returns.")


def page_save_load():
    hero("Save / Load", "Persist the full project so you do not have to rebuild everything from scratch every time.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Save current project")
        if st.session_state.get("training") is None:
            st.info("Train a model before saving.")
        else:
            try:
                bundle_bytes = make_bundle_bytes()
                file_name = f"{st.session_state['config']['project_name']}.zip"
                st.download_button(
                    "Download project bundle",
                    data=bundle_bytes,
                    file_name=file_name,
                    mime="application/zip",
                )
                if st.button("Write a local copy", key="write_local_bundle"):
                    local_path = PROJECTS_DIR / file_name
                    with open(local_path, "wb") as f:
                        f.write(bundle_bytes)
                    st.success(f"Local copy saved to: {local_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with c2:
        st.subheader("Load existing project")
        uploaded_zip = st.file_uploader("Upload saved project zip", type=["zip"], key="bundle_loader")
        if uploaded_zip is not None:
            try:
                load_bundle(uploaded_zip)
                st.success("Project loaded.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    existing = sorted(PROJECTS_DIR.glob("*.zip"))
    if existing:
        st.subheader("Local saved bundles")
        st.dataframe(pd.DataFrame({"file": [p.name for p in existing], "path": [str(p) for p in existing]}), use_container_width=True, hide_index=True)


# -----------------------------
# Workflow / project helpers
# -----------------------------
def _lstm_go(page_name: str):
    st.session_state["page"] = page_name


def _lstm_reset_state(target_page: str):
    preserved = {k: st.session_state[k] for k in list(st.session_state.keys()) if str(k).startswith("__unified_")}
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    for k, v in preserved.items():
        st.session_state[k] = v
    init_state()
    st.session_state["page"] = target_page
    st.rerun()


def _lstm_bottom_nav(page_name: str):
    idx = PAGES.index(page_name) if page_name in PAGES else 0
    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if idx == 0:
            st.button("⬅ Back to Home", use_container_width=True, disabled=True, key=f"lstm_back_{page_name}")
        else:
            prev_page = PAGES[idx - 1]
            st.button(f"⬅ Back to {prev_page}", use_container_width=True, key=f"lstm_back_{page_name}", on_click=_lstm_go, args=(prev_page,))
    with c2:
        if idx >= len(PAGES) - 1:
            st.button("Continue ➜", use_container_width=True, disabled=True, key=f"lstm_next_{page_name}")
        else:
            next_page = PAGES[idx + 1]
            st.button(f"Continue to {next_page} ➜", use_container_width=True, key=f"lstm_next_{page_name}", on_click=_lstm_go, args=(next_page,))


# -----------------------------
# Main app
# -----------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="⛽", layout="wide")
    inject_css()
    init_state()

    with st.sidebar:
        st.markdown(f"### {APP_TITLE}")
        selected = st.radio("Navigation", PAGES, index=PAGES.index(st.session_state["page"]))
        st.session_state["page"] = selected
        st.caption("Single-file professional LSTM workflow for oil and gas forecasting.")

        cfg = st.session_state["config"]
        st.markdown("---")
        st.write("**Current configuration**")
        st.write(f"Lookback: {cfg['lookback']}")
        st.write(f"Horizon: {cfg['horizon']}")
        st.write(f"Targets: {len(cfg['target_cols'])}")
        st.write(f"Transform: {cfg['transform_mode']}")

        st.markdown("---")
        if st.button("➕ New Project", use_container_width=True):
            _lstm_reset_state("Data Upload")
        if st.button("🗑️ Clear Current Project", use_container_width=True):
            _lstm_reset_state("Home")

    page = st.session_state["page"]
    if page == "Home":
        page_home()
    elif page == "Data Upload":
        page_data_upload()
    elif page == "Model":
        page_model()
    elif page == "Preprocess":
        page_preprocess()
    elif page == "Train":
        page_train()
    elif page == "Evaluate":
        page_evaluate()
    elif page == "Predict":
        page_predict()
    elif page == "Visualize":
        page_visualize()
    elif page == "Save/Load":
        page_save_load()

    _lstm_bottom_nav(page)


if __name__ == "__main__":
    main()
'''



import streamlit as st


def get_sources():
    return {
        'ANN': ANN_SOURCE,
        'CNN': CNN_SOURCE,
        'LSTM': LSTM_SOURCE,
    }


def write_sources(output_dir: str | os.PathLike = 'extracted_sources_from_unified') -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / 'ANN.py').write_text(ANN_SOURCE, encoding='utf-8')
    (target / 'CNN.py').write_text(CNN_SOURCE, encoding='utf-8')
    (target / 'LSTM.py').write_text(LSTM_SOURCE, encoding='utf-8')
    return target


def _patched_set_page_config(*args, **kwargs):
    return None


def _ensure_streamlit_shell() -> str:
    st.set_page_config(page_title='Oil & Gas Neural Studio', layout='wide', initial_sidebar_state='expanded')

    st.markdown(
        """
        <style>
            .main .block-container {padding-top: 1.0rem; padding-bottom: 1.2rem;}
            div[data-testid="stHorizontalBlock"] {gap: 0.55rem;}
            .hero {
                background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
                padding: 1.3rem 1.5rem;
                border-radius: 20px;
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(2,6,23,0.16);
            }
            .shell-card {
                background: #ffffff;
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 24px rgba(15,23,42,0.05);
                margin-bottom: 0.9rem;
            }
            .shell-chip {
                display:inline-block;
                padding:0.28rem 0.6rem;
                border-radius:999px;
                background:#eff6ff;
                border:1px solid #bfdbfe;
                color:#1d4ed8;
                font-size:0.85rem;
                margin-right:0.35rem;
                margin-bottom:0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Oil & Gas Neural Studio")
        st.caption("One Streamlit website for ANN, CNN, and LSTM.")
        choice = st.selectbox(
            "Active Model",
            options=["ANN", "CNN", "LSTM"],
            key="__unified_model_choice",
        )

    previous = st.session_state.get("__unified_active_model")
    if previous != choice:
        preserved = {
            "__unified_model_choice": choice,
            "__unified_active_model": choice,
        }
        for k in list(st.session_state.keys()):
            if k not in preserved:
                del st.session_state[k]
        st.session_state["__unified_active_model"] = choice
        st.rerun()

    st.markdown(
        f"""
        <div class="hero">
            <h2 style="margin:0 0 0.35rem 0;">Oil &amp; Gas Neural Studio</h2>
            <div style="font-size:1rem; line-height:1.55;">
                One Streamlit website for ANN, LSTM, and CNN oil &amp; gas workflows.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chips = [
        f"<span class='shell-chip'>Mode: {choice}</span>",
        "<span class='shell-chip'>Unified File: Yes</span>",
        "<span class='shell-chip'>Source Integrity: Embedded</span>",
    ]
    st.markdown("".join(chips), unsafe_allow_html=True)

    return choice


def launch_selected_source() -> None:
    model_name = _ensure_streamlit_shell()
    source = get_sources()[model_name]

    # Prevent duplicate page-config crashes from the embedded original apps.
    st.set_page_config = _patched_set_page_config

    namespace = {
        '__name__': '__main__',
        '__file__': str(Path(__file__).resolve()),
        '__package__': None,
        '__cached__': None,
        '__doc__': None,
    }
    exec(compile(source, f'<{model_name}_SOURCE>', 'exec'), namespace, namespace)


def main() -> None:
    if '--extract' in sys.argv:
        target = write_sources()
        print(f'Extracted original sources to: {target.resolve()}')
        return
    launch_selected_source()


if __name__ == '__main__':
    main()
