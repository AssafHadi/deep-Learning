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

class ann_StreamlitProgressCallback:
    def __init__(self, progress_bar, info_box, total_epochs):
        self.progress_bar = progress_bar
        self.info_box = info_box
        self.total_epochs = max(1, int(total_epochs))

    def as_callback(self):
        tf = ann_get_tf()
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


def ann_train_model(prepared: Dict[str, Any], config: Dict[str, Any]):
    ann_set_seed(int(config["random_seed"]))
    tf = ann_get_tf()

    task = prepared["task"]
    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    class_names = prepared["class_names"] or []

    n_classes = len(class_names) if task == "classification" else 1
    model = ann_build_model(X_train.shape[1], task, config, n_classes=n_classes)

    callbacks = []
    progress = st.progress(0)
    info_box = st.empty()
    callbacks.append(ann_StreamlitProgressCallback(progress, info_box, config["epochs"]).as_callback())

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
            "mape": float(ann_mape_safe(y_test, y_pred)),
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


class cnn_EpochHistoryCallback:
    def __init__(self):
        self.rows = []

    def as_keras_callback(self):
        tf = cnn_get_tf()
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


def cnn_combine_histories(h1: dict, h2: dict):
    if not h1:
        return h2
    if not h2:
        return h1
    out = {}
    keys = sorted(set(h1.keys()) | set(h2.keys()))
    for k in keys:
        out[k] = list(h1.get(k, [])) + list(h2.get(k, []))
    return out


def cnn_compute_class_weights_from_labels(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


class lstm_StreamlitTrainProgressCallback:
    def __init__(self, progress_bar, info_box, total_epochs: int):
        self.progress_bar = progress_bar
        self.info_box = info_box
        self.total_epochs = max(1, int(total_epochs))

    def as_callback(self):
        tf = lstm_get_tf()
        outer = self

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                frac = (epoch + 1) / outer.total_epochs
                outer.progress_bar.progress(min(frac, 1.0))
                parts = [f"Epoch {epoch + 1}/{outer.total_epochs}"]
                for key in ["loss", "mae", "rmse", "accuracy", "val_loss", "val_mae", "val_rmse", "val_accuracy"]:
                    if key in logs:
                        try:
                            parts.append(f"{key}: {float(logs[key]):.4f}")
                        except Exception:
                            pass
                outer.info_box.markdown(" | ".join(parts))

        return _CB()


def lstm_train_model(processed: Dict, cfg: Dict) -> Dict:
    tf = lstm_get_tf()
    task = processed.get("task", "regression")
    num_targets = len(processed["target_cols"])
    horizon = int(cfg["horizon"])

    n_classes = len(processed.get("class_names") or []) if task == "classification" else None
    model = lstm_build_model(processed["X_train"].shape[1:], num_targets, horizon, cfg, n_classes=n_classes)

    ckpt_dir = tempfile.mkdtemp(prefix="oilgas_lstm_ckpt_")
    ckpt_path = str(Path(ckpt_dir) / "best_model.keras")

    progress_bar = st.progress(0)
    info_box = st.empty()

    callbacks = [
        lstm_StreamlitTrainProgressCallback(progress_bar, info_box, int(cfg["epochs"])).as_callback(),
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

    split_tags = processed["split_tags"]
    split_times = {
        "train": processed["target_times"][split_tags == "train"],
        "val": processed["target_times"][split_tags == "val"],
        "test": processed["target_times"][split_tags == "test"],
    }

    if task == "classification":
        raw_probs = {
            "train": model.predict(processed["X_train"], verbose=0),
            "val": model.predict(processed["X_val"], verbose=0),
            "test": model.predict(processed["X_test"], verbose=0),
        }

        predictions, probabilities = {}, {}
        for split, probs in raw_probs.items():
            if n_classes == 2:
                score = np.asarray(probs).reshape(-1)
                pred = (score >= 0.5).astype(int)
                full_prob = np.column_stack([1.0 - score, score])
            else:
                full_prob = np.asarray(probs)
                pred = np.argmax(full_prob, axis=1)
            predictions[split] = pred
            probabilities[split] = full_prob

        actuals = {
            "train": processed["y_train_flat"],
            "val": processed["y_val_flat"],
            "test": processed["y_test_flat"],
        }

        metrics_tables = {
            split: lstm_compute_classification_metrics_table(actuals[split], predictions[split], probabilities[split], processed["class_names"])
            for split in ["train", "val", "test"]
        }

        return {
            "task": "classification",
            "model": model,
            "history": history.history,
            "predictions": predictions,
            "probabilities": probabilities,
            "actuals": actuals,
            "times": split_times,
            "metrics": metrics_tables,
            "model_summary": lstm_model_summary_text(model),
        }

    train_pred_flat = model.predict(processed["X_train"], verbose=0)
    val_pred_flat = model.predict(processed["X_val"], verbose=0)
    test_pred_flat = model.predict(processed["X_test"], verbose=0)

    preds = {
        "train": lstm_inverse_3d(train_pred_flat, processed["target_scaler"], horizon, num_targets),
        "val": lstm_inverse_3d(val_pred_flat, processed["target_scaler"], horizon, num_targets),
        "test": lstm_inverse_3d(test_pred_flat, processed["target_scaler"], horizon, num_targets),
    }
    actuals = {
        "train": lstm_inverse_3d(processed["y_train_flat"], processed["target_scaler"], horizon, num_targets),
        "val": lstm_inverse_3d(processed["y_val_flat"], processed["target_scaler"], horizon, num_targets),
        "test": lstm_inverse_3d(processed["y_test_flat"], processed["target_scaler"], horizon, num_targets),
    }

    metrics_tables = {
        split: lstm_compute_metrics_table(actuals[split], preds[split], processed["target_cols"], horizon)
        for split in ["train", "val", "test"]
    }

    return {
        "task": "regression",
        "model": model,
        "history": history.history,
        "predictions": preds,
        "actuals": actuals,
        "times": split_times,
        "metrics": metrics_tables,
        "model_summary": lstm_model_summary_text(model),
    }


# Public workflow API
def train_ann(*args, **kwargs):
    return ann_train_model(*args, **kwargs)

def train_cnn(train_df: pd.DataFrame, val_df: pd.DataFrame, class_names: list[str], cfg: dict):
    """Run the original CNN training workflow without owning any page UI."""
    cnn_seed_everything(cfg["seed"])
    class_names = sorted(class_names)
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    y_train = train_df["label"].map(label_to_idx).astype(int).to_numpy()
    y_val = val_df["label"].map(label_to_idx).astype(int).to_numpy()

    train_ds = cnn_make_tf_dataset(train_df["filepath"].tolist(), y_train, cfg, training=True)
    val_ds = cnn_make_tf_dataset(val_df["filepath"].tolist(), y_val, cfg, training=False)

    model, base_model = cnn_build_model(class_names, cfg)
    tf = cnn_get_tf()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]

    hist_cb_1 = cnn_EpochHistoryCallback()
    callbacks_all_1 = callbacks + [hist_cb_1.as_keras_callback()]
    class_weights = cnn_compute_class_weights_from_labels(y_train) if cfg["use_class_weights"] else None

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
            optimizer=cnn_make_optimizer(cfg["optimizer"], cfg["fine_tune_lr"]),
            loss=cnn_make_classification_loss(cfg.get("label_smoothing", 0.0)),
            metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=min(3, len(class_names)), name="top_k_acc")],
        )

        hist_cb_2 = cnn_EpochHistoryCallback()
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
        merged_history = cnn_combine_histories(hist1.history, hist2.history)
        epoch_rows.extend(hist_cb_2.rows)

    if epoch_rows:
        hist_df = pd.DataFrame(epoch_rows).sort_values("epoch")
        merged_history["lr"] = hist_df["lr"].tolist()

    eval_artifacts = cnn_evaluate_model(model, val_df, class_names, cfg)
    return {"model": model, "history": merged_history, "eval_artifacts": eval_artifacts}

def train_lstm(*args, **kwargs):
    return lstm_train_model(*args, **kwargs)
