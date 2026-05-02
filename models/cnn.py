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

MODEL_NAME = "CNN"

# Model-specific constants and algorithms extracted from the original implementation.
CNN_MODEL_NAME = "CNN"


plt.rcParams["figure.dpi"] = 110


plt.rcParams["savefig.dpi"] = 110


plt.rcParams["axes.titlesize"] = 11


plt.rcParams["axes.labelsize"] = 9


plt.rcParams["xtick.labelsize"] = 8


plt.rcParams["ytick.labelsize"] = 8


plt.rcParams["legend.fontsize"] = 8


CNN_APP_TITLE = "Oil & Gas CNN Studio"


CNN_APP_DIR = Path(".oil_gas_cnn_studio")


CNN_DATA_DIR = CNN_APP_DIR / "datasets"


CNN_MODEL_DIR = CNN_APP_DIR / "saved_projects"


CNN_CACHE_DIR = CNN_APP_DIR / "cache"


for p in [CNN_APP_DIR, CNN_DATA_DIR, CNN_MODEL_DIR, CNN_CACHE_DIR]:
    p.mkdir(parents=True, exist_ok=True)


_CNN_TF = None


CNN_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif", ".gif", ".heic", ".heif"}


def cnn_get_tf():
    global _CNN_TF
    if _CNN_TF is None:
        import tensorflow as tf
        _CNN_TF = tf
    return _CNN_TF


def cnn_seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf = cnn_get_tf()
    tf.random.set_seed(seed)


def cnn_safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


def cnn_safe_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


def cnn_bytes_from_pil(img: Image.Image, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def cnn_ensure_rgb(img: Image.Image, mode: str = "RGB") -> Image.Image:
    if mode == "RGB":
        return img.convert("RGB")
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    arr3 = np.stack([arr, arr, arr], axis=-1)
    return Image.fromarray(arr3)


def cnn_make_augmentation_layers(cfg):
    tf = cnn_get_tf()
    aug_cfg = cfg["augmentation"]
    layers = [
        tf.keras.layers.RandomFlip("horizontal") if aug_cfg["flip"] else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.RandomRotation(aug_cfg["rotation"]) if aug_cfg["rotation"] > 0 else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.RandomZoom(aug_cfg["zoom"]) if aug_cfg["zoom"] > 0 else tf.keras.layers.Lambda(lambda x: x),
        tf.keras.layers.RandomContrast(aug_cfg["contrast"]) if aug_cfg["contrast"] > 0 else tf.keras.layers.Lambda(lambda x: x),
    ]
    return tf.keras.Sequential(layers, name="augmentation")


def cnn_preprocess_input_layer(backbone: str):
    tf = cnn_get_tf()
    mapping = {
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "EfficientNetB0": tf.keras.applications.efficientnet.preprocess_input,
        "ResNet50": tf.keras.applications.resnet.preprocess_input,
    }
    fn = mapping[backbone]
    return tf.keras.layers.Lambda(fn, name="preprocess_input")


def cnn_build_backbone(backbone_name: str, input_shape, weights="imagenet"):
    tf = cnn_get_tf()
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


def cnn_make_classification_loss(label_smoothing: float = 0.0):
    tf = cnn_get_tf()
    # Keep sparse integer labels for compatibility.
    # Many TensorFlow/Keras builds do not accept label_smoothing here.
    # So we intentionally ignore it instead of crashing the app.
    _ = label_smoothing
    return tf.keras.losses.SparseCategoricalCrossentropy()


def cnn_build_model(class_names, cfg):
    tf = cnn_get_tf()
    img_size = int(cfg["image_size"])
    input_shape = (img_size, img_size, 3)

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = cnn_make_augmentation_layers(cfg)(inputs)
    x = cnn_preprocess_input_layer(cfg["backbone"])(x)

    base_model = cnn_build_backbone(cfg["backbone"], input_shape, cfg["weights"])
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
    opt = cnn_make_optimizer(cfg["optimizer"], cfg["learning_rate"])
    metrics = [
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, len(class_names)), name="top_k_acc"),
    ]
    model.compile(
        optimizer=opt,
        loss=cnn_make_classification_loss(cfg.get("label_smoothing", 0.0)),
        metrics=metrics,
    )
    return model, base_model


def cnn_make_optimizer(name: str, lr: float):
    tf = cnn_get_tf()
    if name == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    if name == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {name}")


def cnn_find_last_conv_layer_name(model):
    # We expose the backbone feature map explicitly to make Grad-CAM robust
    # even when the backbone itself is a nested sub-model.
    try:
        model.get_layer("backbone_feature_maps")
        return "backbone_feature_maps"
    except Exception:
        return None


# Public architecture aliases requested by the layered design.
build_cnn_model = cnn_build_model
def get_cnn_default_config() -> Dict[str, Any]:
    return {
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
    }
