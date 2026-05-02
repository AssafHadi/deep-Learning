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
from services.training import *
from services.evaluation import *
from services.prediction import *

def ann_safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip())
    return name or "project"


def ann_json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return str(obj)


def ann_create_project_bundle_bytes() -> bytes:
    if st.session_state.trained_model is None or st.session_state.prepared_data is None:
        raise ValueError("Train or load a project first.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = ann_safe_filename(st.session_state.config["project_name"])

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
        meta_path.write_text(json.dumps(meta, indent=2, default=ann_json_default), encoding="utf-8")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path, arcname="model.keras")
            zf.write(bundle_path, arcname="bundle.joblib")
            zf.write(meta_path, arcname="meta.json")
        return buffer.getvalue()


def ann_save_project_locally():
    data = ann_create_project_bundle_bytes()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = ann_safe_filename(st.session_state.config["project_name"])
    path = ANN_PROJECTS_DIR / f"{project_name}_{ts}.zip"
    path.write_bytes(data)
    return path


def ann_load_project_from_zip_bytes(zip_bytes: bytes):
    tf = ann_get_tf()
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


def cnn_now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cnn_save_project(project_name: str):
    if st.session_state.trained_model is None:
        raise ValueError("No trained model available to save.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = CNN_MODEL_DIR / f"{project_name}_{ts}"
    project_dir.mkdir(parents=True, exist_ok=True)

    model_path = project_dir / "model.keras"
    meta_path = project_dir / "metadata.joblib"

    st.session_state.trained_model.save(model_path)

    meta = {
        "saved_at": cnn_now_str(),
        "project_name": project_name,
        "class_names": st.session_state.class_names,
        "model_config": st.session_state.model_config,
        "history": st.session_state.history,
        "data_summary": st.session_state.data_summary,
        "eval_artifacts": st.session_state.eval_artifacts,
    }
    joblib.dump(meta, meta_path)
    return project_dir


def cnn_load_project(project_dir: Path):
    tf = cnn_get_tf()
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


def cnn_available_projects():
    dirs = [p for p in CNN_MODEL_DIR.iterdir() if p.is_dir()]
    return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def lstm_make_bundle_bytes() -> bytes:
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


def lstm_load_bundle(uploaded_zip) -> None:
    data = uploaded_zip.getvalue()
    import_dir = Path(tempfile.mkdtemp(prefix="oilgas_import_", dir=str(LSTM_TEMP_IMPORT_DIR)))
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        zf.extractall(import_dir)

    tf = lstm_get_tf()
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


# Public persistence API
def save_project(*args, **kwargs):
    model_name = kwargs.pop("model_name", None)
    if model_name == "CNN":
        return cnn_save_project(*args, **kwargs)
    if model_name == "LSTM":
        return lstm_make_bundle_bytes(*args, **kwargs)
    return ann_save_project_locally(*args, **kwargs)

def load_project(*args, **kwargs):
    model_name = kwargs.pop("model_name", None)
    if model_name == "CNN":
        return cnn_load_project(*args, **kwargs)
    if model_name == "LSTM":
        return lstm_load_bundle(*args, **kwargs)
    return ann_load_project_from_zip_bytes(*args, **kwargs)

def save_model_artifacts(*args, **kwargs):
    return save_project(*args, **kwargs)

def load_model_artifacts(*args, **kwargs):
    return load_project(*args, **kwargs)

def save_metadata(path, metadata):
    Path(path).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

def load_metadata(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
