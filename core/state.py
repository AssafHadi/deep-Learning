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

from contextlib import contextmanager

from core.config import WORKFLOW_PAGES
from models.ann import *
from models.cnn import *
from models.lstm import *

def ann_init_state():
    if "config" not in st.session_state:
        st.session_state.config = ann_default_config()
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


def cnn_init_state():
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


def lstm_init_state() -> None:
    defaults = {
        "page": "Home",
        "raw_df": None,
        "processed": None,
        "training": None,
        "prediction": None,
        "config": lstm_default_config(),
        "loaded_filename": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def lstm_reset_after_data_change() -> None:
    st.session_state["processed"] = None
    st.session_state["training"] = None
    st.session_state["prediction"] = None


_MODEL_INIT = {"ANN": ann_init_state, "CNN": cnn_init_state, "LSTM": lstm_init_state}
_STATE_PREFIX = "__ns_model_state_"
_PROTECTED_PREFIXES = ("__ns_",)


def _snapshot_public_state() -> dict:
    return {k: st.session_state[k] for k in list(st.session_state.keys()) if not str(k).startswith(_PROTECTED_PREFIXES)}


def _restore_public_state(snapshot: dict) -> None:
    for key in list(st.session_state.keys()):
        if not str(key).startswith(_PROTECTED_PREFIXES):
            del st.session_state[key]
    for key, value in snapshot.items():
        st.session_state[key] = value


def _current_model() -> str:
    return st.session_state.get("__ns_selected_model", "ANN")


def switch_model(model_name: str) -> None:
    previous = st.session_state.get("__ns_active_model")
    if previous == model_name:
        _MODEL_INIT[model_name]()
        return
    if previous in _MODEL_INIT:
        st.session_state[_STATE_PREFIX + previous] = _snapshot_public_state()
    snapshot = st.session_state.get(_STATE_PREFIX + model_name, {})
    _restore_public_state(snapshot)
    st.session_state["__ns_active_model"] = model_name
    st.session_state.setdefault("__ns_selected_page", "Home")
    _MODEL_INIT[model_name]()


@contextmanager
def model_state_context(model_name: str):
    previous = st.session_state.get("__ns_active_model")
    if previous != model_name:
        switch_model(model_name)
    else:
        _MODEL_INIT[model_name]()
    try:
        yield
    finally:
        st.session_state[_STATE_PREFIX + model_name] = _snapshot_public_state()


def get_selected_page() -> str:
    page = st.session_state.get("__ns_selected_page", "Home")
    return page if page in WORKFLOW_PAGES else "Home"


def set_selected_page(page_name: str) -> None:
    st.session_state["__ns_selected_page"] = page_name if page_name in WORKFLOW_PAGES else "Home"


def reset_model_project(model_name: str, target_page: str = "Home") -> None:
    for key in list(st.session_state.keys()):
        if not str(key).startswith(_PROTECTED_PREFIXES):
            del st.session_state[key]
    _MODEL_INIT[model_name]()
    set_selected_page(target_page)
    st.rerun()
