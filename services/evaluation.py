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

def cnn_evaluate_model(model, val_df: pd.DataFrame, class_names, cfg):
    tf = cnn_get_tf()
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_true = val_df["label"].map(label_to_idx).astype(int).to_numpy()
    paths = val_df["filepath"].tolist()
    ds = cnn_make_tf_dataset(paths, y_true, cfg, training=False)
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


def lstm_safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    out = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(out) * 100.0)


def lstm_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean() * 100.0)


def lstm_compute_metrics_table(actual_3d: np.ndarray, pred_3d: np.ndarray, target_cols: List[str], horizon: int) -> pd.DataFrame:
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
                    "MAPE_%": lstm_safe_mape(yt, yp),
                    "R2": r2_score(yt, yp),
                    "Directional_Accuracy_%": lstm_directional_accuracy(yt, yp),
                }
            )
    return pd.DataFrame(rows)


def lstm_compute_classification_metrics_table(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    rows = [
        {
            "metric": "Accuracy",
            "value": accuracy_score(y_true, y_pred),
        },
        {
            "metric": "Precision Weighted",
            "value": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        },
        {
            "metric": "Recall Weighted",
            "value": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        },
        {
            "metric": "F1 Weighted",
            "value": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        },
    ]
    if len(class_names) == 2 and y_prob is not None and len(np.asarray(y_prob).shape) == 2:
        try:
            rows.append({"metric": "ROC AUC", "value": roc_auc_score(y_true, np.asarray(y_prob)[:, 1])})
        except Exception:
            pass
    return pd.DataFrame(rows)


def lstm_backtest_strategies(actual_returns: np.ndarray, predicted_returns: np.ndarray, dates: np.ndarray, target_cols: List[str], freq_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    ann = lstm_annualization_factor(freq_name)
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


def lstm_returns_for_backtest(transform_mode: str, actual_values: np.ndarray, pred_values: np.ndarray) -> tuple:
    if transform_mode == "pct_change":
        return actual_values, pred_values

    if transform_mode == "log_return":
        return np.expm1(actual_values), np.expm1(pred_values)

    raise ValueError("Backtest is only valid for pct_change or log_return mode.")


# Public workflow API
def evaluate_ann(results: dict | None = None):
    """Return ANN evaluation artifacts already produced by ann_train_model()."""
    return results if results is not None else st.session_state.get("results", {})

def evaluate_cnn(*args, **kwargs):
    return cnn_evaluate_model(*args, **kwargs)

def evaluate_lstm(training: dict | None = None):
    """Return LSTM evaluation artifacts already produced by lstm_train_model()."""
    return training if training is not None else st.session_state.get("training", {})
