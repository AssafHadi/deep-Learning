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

def ann_plot_training_curves(history: Dict[str, List[float]]):
    if not history:
        st.info("No training history available yet.")
        return
    plt = ann_get_plt()
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


def ann_plot_confusion(y_true, y_pred, class_names):
    plt = ann_get_plt()
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


def ann_plot_binary_curves(y_true, y_score):
    if y_score is None:
        return
    plt = ann_get_plt()

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


def ann_plot_regression_results(y_true, y_pred):
    plt = ann_get_plt()
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


def ann_plot_data_profile(df: pd.DataFrame, target_col: Optional[str]):
    plt = ann_get_plt()
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
            if ann_infer_task(df[target_col]) == "classification":
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

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        st.subheader("Numeric Correlation Heatmap")
        heatmap_col, _ = st.columns(2)
        with heatmap_col:
            corr = df[numeric_cols].corr(numeric_only=True)
            top = corr.columns[: min(20, len(corr.columns))]
            corr = corr.loc[top, top]
            fig, ax = plt.subplots(figsize=(7.0, 5.2))
            im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(corr.index)))
            ax.set_yticklabels(corr.index)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig, clear_figure=True, use_container_width=True)


def ann_make_training_curve_fig(history: Dict[str, List[float]], metric: str):
    plt = ann_get_plt()
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


def ann_make_confusion_fig(y_true, y_pred, class_names):
    plt = ann_get_plt()
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


def ann_make_binary_curve_figs(y_true, y_score):
    if y_score is None:
        return None, None
    plt = ann_get_plt()
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


def ann_make_predicted_distribution_fig(y_pred):
    plt = ann_get_plt()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    pd.Series(y_pred).value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Predicted Class Distribution")
    ax.set_xlabel("Encoded Class")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2, axis="y")
    return fig


def ann_make_regression_figures(y_true, y_pred):
    plt = ann_get_plt()
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


def cnn_fig_show(fig):
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def _cnn_cnn_style_axis(ax, title=None, xlabel=None, ylabel=None, grid_axis="both"):
    if title:
        ax.set_title(title, fontsize=10.5, fontweight="semibold", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.8)
    if str(grid_axis).lower() not in {"none", "off", "false"}:
        ax.grid(True, alpha=0.22, axis=grid_axis, linewidth=0.7)
    else:
        ax.grid(False)
    ax.tick_params(axis="both", labelsize=7.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _cnn_short_label(value, max_len=18):
    text = str(value)
    return text if len(text) <= max_len else text[:max_len - 1] + "…"


def cnn_plot_class_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().sort_values(ascending=True)
    if counts.empty:
        st.warning("No class labels found.")
        return

    fig_h = min(3.6, max(2.2, 0.34 * len(counts) + 1.0))
    fig, ax = plt.subplots(figsize=(5.0, fig_h))
    labels = [_cnn_short_label(x, 24) for x in counts.index]
    ax.barh(labels, counts.values)
    _cnn_cnn_style_axis(ax, "Class Distribution", "Images", "Class", grid_axis="x")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=7.5)
    fig.tight_layout()
    cnn_fig_show(fig)


def cnn_plot_image_dimensions(df: pd.DataFrame):
    sample = df.sample(min(len(df), 180), random_state=42)
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

    col1, col2 = st.columns(2, gap="small")
    with col1:
        fig, ax = plt.subplots(figsize=(4.2, 2.55))
        ax.scatter(widths, heights, alpha=0.68, s=18)
        _cnn_cnn_style_axis(ax, "Image Width vs Height", "Width (px)", "Height (px)")
        fig.tight_layout()
        cnn_fig_show(fig)

    with col2:
        ratios = np.asarray(widths, dtype=float) / np.maximum(np.asarray(heights, dtype=float), 1.0)
        fig, ax = plt.subplots(figsize=(4.2, 2.55))
        ax.hist(ratios, bins=min(24, max(8, len(ratios) // 4)), alpha=0.88)
        _cnn_cnn_style_axis(ax, "Aspect Ratio Distribution", "Width / Height", "Images", grid_axis="y")
        fig.tight_layout()
        cnn_fig_show(fig)


def cnn_show_sample_gallery(df: pd.DataFrame, n_per_class=5):
    classes = df["label"].value_counts().index.tolist()
    if not classes:
        st.warning("No images available for the gallery.")
        return

    st.markdown("### Sample Images Grid")
    max_cols = max(1, min(int(n_per_class), 5))
    for cname in classes:
        sub = df[df["label"] == cname].sample(min(max_cols, (df["label"] == cname).sum()), random_state=42)
        st.markdown(f"**{cname}**")
        cols = st.columns(max_cols, gap="small")
        for i, (_, row) in enumerate(sub.iterrows()):
            try:
                with Image.open(row["filepath"]) as img:
                    cols[min(i, max_cols - 1)].image(
                        img,
                        use_container_width=True,
                        caption=Path(row["filepath"]).name,
                    )
            except Exception as e:
                cols[min(i, max_cols - 1)].warning(f"Could not read image: {e}")


def cnn_plot_training_curves(history: dict):
    if not history:
        st.warning("No training history available.")
        return

    df = pd.DataFrame(history)
    if df.empty:
        st.warning("Training history is empty.")
        return

    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(4.4, 2.55))
        if "loss" in df:
            ax.plot(df.index + 1, df["loss"], linewidth=1.8, label="Train Loss")
        if "val_loss" in df:
            ax.plot(df.index + 1, df["val_loss"], linewidth=1.8, label="Validation Loss")
            best_idx = int(df["val_loss"].idxmin()) + 1
            ax.axvline(best_idx, linestyle="--", alpha=0.55, linewidth=1.0, label=f"Best {best_idx}")
        _cnn_cnn_style_axis(ax, "Training Loss vs Validation Loss", "Epoch", "Loss")
        ax.legend(loc="upper right", fontsize=7.3, frameon=True)
        fig.tight_layout()
        cnn_fig_show(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4.4, 2.55))
        metric_pairs = [
            ("accuracy", "val_accuracy", "Accuracy"),
            ("sparse_categorical_accuracy", "val_sparse_categorical_accuracy", "Accuracy"),
            ("categorical_accuracy", "val_categorical_accuracy", "Accuracy"),
        ]
        plotted = False
        for train_key, val_key, label in metric_pairs:
            if train_key in df or val_key in df:
                if train_key in df:
                    ax.plot(df.index + 1, df[train_key], linewidth=1.8, label=f"Train {label}")
                    plotted = True
                if val_key in df:
                    ax.plot(df.index + 1, df[val_key], linewidth=1.8, label=f"Validation {label}")
                    plotted = True
                break
        if not plotted:
            numeric_cols = [c for c in df.columns if c not in {"loss", "val_loss", "lr"}]
            for c in numeric_cols[:2]:
                ax.plot(df.index + 1, df[c], linewidth=1.8, label=c)
                plotted = True
        _cnn_cnn_style_axis(ax, "Training Accuracy vs Validation Accuracy", "Epoch", "Accuracy")
        if plotted:
            ax.legend(loc="lower right", fontsize=7.3, frameon=True)
        fig.tight_layout()
        cnn_fig_show(fig)

    if "lr" in df:
        fig, ax = plt.subplots(figsize=(4.6, 2.15))
        ax.plot(df.index + 1, df["lr"], linewidth=1.8)
        _cnn_cnn_style_axis(ax, "Learning Rate Schedule", "Epoch", "Learning Rate")
        fig.tight_layout()
        cnn_fig_show(fig)


def cnn_plot_confusion_matrices(cm, class_names):
    cm = np.asarray(cm)
    if cm.size == 0:
        st.warning("Confusion matrix is empty.")
        return
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = [_cnn_short_label(c, 14) for c in class_names]

    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(4.25, 3.45))
        im = ax.imshow(cm, interpolation="nearest")
        _cnn_cnn_style_axis(ax, "Confusion Matrix", "Predicted", "Actual", grid_axis="none")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        cnn_fig_show(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4.25, 3.45))
        im = ax.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
        _cnn_cnn_style_axis(ax, "Normalized Confusion Matrix", "Predicted", "Actual", grid_axis="none")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        thresh = cm_norm.max() / 2 if cm_norm.size else 0
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm_norm[i, j] > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        cnn_fig_show(fig)


def cnn_plot_per_class_metrics(report_dict, class_names):
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

    st.dataframe(df, use_container_width=True, hide_index=True)

    fig_h = min(3.8, max(2.55, 0.28 * len(df) + 1.35))
    fig, ax = plt.subplots(figsize=(5.35, fig_h))
    x = np.arange(len(df))
    w = 0.24
    ax.bar(x - w, df["precision"], width=w, label="Precision")
    ax.bar(x, df["recall"], width=w, label="Recall")
    ax.bar(x + w, df["f1-score"], width=w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([_cnn_short_label(c, 15) for c in df["class"]], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    _cnn_cnn_style_axis(ax, "Per-Class Precision / Recall / F1", "Class", "Score", grid_axis="y")
    ax.legend(loc="lower right", fontsize=7.5, frameon=True)
    fig.tight_layout()
    cnn_fig_show(fig)


def cnn_plot_roc_pr(eval_artifacts):
    roc_info = eval_artifacts.get("roc_info", {})
    pr_info = eval_artifacts.get("pr_info", {})
    class_names = eval_artifacts.get("class_names", [])

    if not roc_info and not pr_info:
        st.info("ROC/PR curves are not available for this validation run.")
        return

    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(4.35, 2.75))
        plotted = False
        if "binary" in roc_info:
            d = roc_info["binary"]
            ax.plot(d["fpr"], d["tpr"], linewidth=1.8, label=f"AUC = {d['auc']:.3f}")
            plotted = True
        else:
            for cname in class_names[:8]:
                if cname in roc_info:
                    d = roc_info[cname]
                    ax.plot(d["fpr"], d["tpr"], linewidth=1.35, label=f"{_cnn_short_label(cname, 10)} ({d['auc']:.2f})")
                    plotted = True
        ax.plot([0, 1], [0, 1], linestyle="--", alpha=0.55, linewidth=1.0)
        _cnn_cnn_style_axis(ax, "ROC Curve", "False Positive Rate", "True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        if plotted:
            ax.legend(fontsize=6.6, loc="lower right", frameon=True)
        fig.tight_layout()
        cnn_fig_show(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4.35, 2.75))
        plotted = False
        if "binary" in pr_info:
            d = pr_info["binary"]
            ax.plot(d["recall"], d["precision"], linewidth=1.8, label=f"AP = {d['ap']:.3f}")
            plotted = True
        else:
            for cname in class_names[:8]:
                if cname in pr_info:
                    d = pr_info[cname]
                    ax.plot(d["recall"], d["precision"], linewidth=1.35, label=f"{_cnn_short_label(cname, 10)} ({d['ap']:.2f})")
                    plotted = True
        _cnn_cnn_style_axis(ax, "Precision-Recall Curve", "Recall", "Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        if plotted:
            ax.legend(fontsize=6.6, loc="lower left", frameon=True)
        fig.tight_layout()
        cnn_fig_show(fig)


def cnn_plot_prediction_confidence_distribution(eval_artifacts):
    probs = np.asarray(eval_artifacts.get("y_prob", []))
    y_true = np.asarray(eval_artifacts.get("y_true", []))
    y_pred = np.asarray(eval_artifacts.get("y_pred", []))
    if probs.size == 0 or y_true.size == 0 or y_pred.size == 0:
        st.info("Prediction confidence data is not available.")
        return

    conf = probs.max(axis=1)
    correct = y_true == y_pred
    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(4.35, 2.55))
        if np.any(correct):
            ax.hist(conf[correct], bins=16, alpha=0.72, label="Correct")
        if np.any(~correct):
            ax.hist(conf[~correct], bins=16, alpha=0.72, label="Incorrect")
        _cnn_cnn_style_axis(ax, "Prediction Confidence Distribution", "Max Class Probability", "Images", grid_axis="y")
        ax.set_xlim(0, 1.0)
        ax.legend(fontsize=7.2, frameon=True)
        fig.tight_layout()
        cnn_fig_show(fig)

    with col2:
        class_names = eval_artifacts.get("class_names", [])
        rows = []
        for i, cname in enumerate(class_names):
            mask = y_pred == i
            if np.any(mask):
                rows.append((cname, float(np.mean(conf[mask])), int(np.sum(mask))))
        if rows:
            plot_df = pd.DataFrame(rows, columns=["class", "mean_confidence", "count"]).sort_values("mean_confidence")
            fig, ax = plt.subplots(figsize=(4.35, 2.55))
            ax.barh([_cnn_short_label(c, 18) for c in plot_df["class"]], plot_df["mean_confidence"])
            _cnn_cnn_style_axis(ax, "Mean Confidence by Predicted Class", "Mean Confidence", "Predicted Class", grid_axis="x")
            ax.set_xlim(0, 1.0)
            for i, (v, n) in enumerate(zip(plot_df["mean_confidence"], plot_df["count"])):
                ax.text(v, i, f" {v:.2f} ({n})", va="center", fontsize=7)
            fig.tight_layout()
            cnn_fig_show(fig)


def cnn_plot_correct_incorrect_summary(eval_artifacts):
    y_true = np.asarray(eval_artifacts.get("y_true", []))
    y_pred = np.asarray(eval_artifacts.get("y_pred", []))
    class_names = eval_artifacts.get("class_names", [])
    if y_true.size == 0 or y_pred.size == 0:
        return

    correct = y_true == y_pred
    col1, col2 = st.columns(2, gap="small")

    with col1:
        fig, ax = plt.subplots(figsize=(4.1, 2.45))
        values = [int(np.sum(correct)), int(np.sum(~correct))]
        ax.bar(["Correct", "Incorrect"], values)
        _cnn_cnn_style_axis(ax, "Correct vs Incorrect Predictions", "Prediction Result", "Images", grid_axis="y")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        cnn_fig_show(fig)

    with col2:
        rows = []
        for i, cname in enumerate(class_names):
            mask = y_true == i
            total = int(np.sum(mask))
            if total:
                acc = float(np.mean(y_pred[mask] == y_true[mask]))
                rows.append((cname, acc, total))
        if rows:
            plot_df = pd.DataFrame(rows, columns=["class", "accuracy", "support"]).sort_values("accuracy")
            fig, ax = plt.subplots(figsize=(4.1, 2.45))
            ax.barh([_cnn_short_label(c, 18) for c in plot_df["class"]], plot_df["accuracy"])
            _cnn_cnn_style_axis(ax, "Per-Class Validation Accuracy", "Accuracy", "Class", grid_axis="x")
            ax.set_xlim(0, 1.0)
            for i, (v, n) in enumerate(zip(plot_df["accuracy"], plot_df["support"])):
                ax.text(v, i, f" {v:.2f} ({n})", va="center", fontsize=7)
            fig.tight_layout()
            cnn_fig_show(fig)


def cnn_show_misclassified(eval_artifacts, max_items=12):
    y_true = np.asarray(eval_artifacts["y_true"])
    y_pred = np.asarray(eval_artifacts["y_pred"])
    probs = np.asarray(eval_artifacts["y_prob"])
    paths = eval_artifacts["val_paths"]
    class_names = eval_artifacts["class_names"]

    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        st.success("No misclassified validation samples found.")
        return

    wrong = sorted(wrong.tolist(), key=lambda idx: float(probs[idx, y_pred[idx]]), reverse=True)
    selected = wrong[:max_items]
    st.markdown("### Top Misclassified Images")
    cols = st.columns(4, gap="small")
    for i, idx in enumerate(selected):
        try:
            with Image.open(paths[idx]) as img:
                prob = float(probs[idx, y_pred[idx]])
                cols[i % len(cols)].image(
                    img,
                    caption=f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}\nConf: {prob:.3f}",
                    use_container_width=True,
                )
        except Exception as e:
            cols[i % len(cols)].warning(f"Could not load image: {e}")


def cnn_show_correct_incorrect_examples(eval_artifacts, max_each=4):
    y_true = np.asarray(eval_artifacts["y_true"])
    y_pred = np.asarray(eval_artifacts["y_pred"])
    probs = np.asarray(eval_artifacts["y_prob"])
    paths = eval_artifacts["val_paths"]
    class_names = eval_artifacts["class_names"]

    correct_idx = np.where(y_true == y_pred)[0].tolist()
    wrong_idx = np.where(y_true != y_pred)[0].tolist()
    correct_idx = sorted(correct_idx, key=lambda idx: float(probs[idx, y_pred[idx]]), reverse=True)[:max_each]
    wrong_idx = sorted(wrong_idx, key=lambda idx: float(probs[idx, y_pred[idx]]), reverse=True)[:max_each]

    st.markdown("### Correct vs Incorrect Prediction Examples")
    c1, c2 = st.columns(2, gap="small")

    with c1:
        st.markdown("**High-confidence correct**")
        if not correct_idx:
            st.info("No correct examples available.")
        else:
            cols = st.columns(2, gap="small")
            for i, idx in enumerate(correct_idx):
                try:
                    with Image.open(paths[idx]) as img:
                        conf = float(probs[idx, y_pred[idx]])
                        cols[i % 2].image(
                            img,
                            caption=f"{class_names[y_pred[idx]]}\nConf: {conf:.3f}",
                            use_container_width=True,
                        )
                except Exception as e:
                    cols[i % 2].warning(f"Could not load image: {e}")

    with c2:
        st.markdown("**High-confidence incorrect**")
        if not wrong_idx:
            st.success("No incorrect examples available.")
        else:
            cols = st.columns(2, gap="small")
            for i, idx in enumerate(wrong_idx):
                try:
                    with Image.open(paths[idx]) as img:
                        conf = float(probs[idx, y_pred[idx]])
                        cols[i % 2].image(
                            img,
                            caption=f"True: {class_names[y_true[idx]]}\nPred: {class_names[y_pred[idx]]}\nConf: {conf:.3f}",
                            use_container_width=True,
                        )
                except Exception as e:
                    cols[i % 2].warning(f"Could not load image: {e}")


def cnn_plot_embedding_map(model, eval_artifacts, cfg):
    if model is None:
        st.info("Train or load a model to view the feature embedding.")
        return
    paths = eval_artifacts["val_paths"]
    y_true = np.asarray(eval_artifacts["y_true"])
    class_names = eval_artifacts["class_names"]
    try:
        feats = cnn_extract_features(model, paths, cfg, batch_limit=min(len(paths), 256))
    except Exception as e:
        st.info(f"Feature embedding is unavailable for this model: {e}")
        return
    if feats.shape[0] < 3:
        st.warning("Not enough samples to compute PCA embedding.")
        return
    pca = PCA(n_components=2, random_state=42)
    emb = pca.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(4.65, 3.05))
    for i, cname in enumerate(class_names):
        mask = y_true[:len(emb)] == i
        ax.scatter(emb[mask, 0], emb[mask, 1], label=_cnn_short_label(cname, 12), alpha=0.74, s=18)
    _cnn_cnn_style_axis(ax, "Validation Feature Embedding (PCA)", "PC 1", "PC 2")
    ax.legend(fontsize=6.7, loc="best", frameon=True)
    fig.tight_layout()
    cnn_fig_show(fig)


def _lstm_new_fig(width: float = LSTM_PLOT_W, height: float = LSTM_PLOT_H):
    fig, ax = plt.subplots(figsize=(width, height), dpi=LSTM_PLOT_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    return fig, ax


def _lstm_style_ax(ax, title: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None, legend: bool = True):
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=LSTM_LSTM_COLORS["text"], pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.6, color=LSTM_LSTM_COLORS["text"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.6, color=LSTM_LSTM_COLORS["text"])
    ax.grid(True, alpha=0.34, linewidth=0.65, color=LSTM_LSTM_COLORS["grid"])
    ax.tick_params(axis="both", labelsize=7.8, colors="#334155")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#94a3b8")
        ax.spines[spine].set_linewidth(0.8)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                loc="best",
                fontsize=7.6,
                frameon=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#cbd5e1",
            )
    return ax


def _lstm_finish_fig(fig):
    fig.tight_layout(pad=0.7)
    return fig


def lstm_show_lstm_fig(fig):
    """Show compact LSTM figures without stretching them to full page width."""
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    try:
        plt.close(fig)
    except Exception:
        pass


def _lstm_numeric_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def _lstm_valid_pair(actual, pred) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(pred, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(p)
    return a[mask], p[mask]


def _lstm_downsample_df(df: pd.DataFrame, max_points: int = LSTM_MAX_PLOT_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).round().astype(int)
    return df.iloc[idx]


def _lstm_downsample_pair(a: np.ndarray, p: np.ndarray, max_points: int = LSTM_MAX_PLOT_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    if len(a) <= max_points:
        return a, p
    idx = np.linspace(0, len(a) - 1, max_points).round().astype(int)
    return a[idx], p[idx]


def _lstm_clean_timeseries_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Plot-only cleanup.
    - Converts values to numeric.
    - Sorts datetime index.
    - Collapses duplicate timestamps so well-by-well rows do not draw vertical spike artifacts.
      For production-rate data, summing duplicate timestamps gives a readable field-level rate.
    """
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(how="all")
    if len(out) == 0:
        return out

    if isinstance(out.index, pd.DatetimeIndex):
        out = out.sort_index()
        if out.index.has_duplicates:
            out = out.groupby(level=0).sum(min_count=1)
    else:
        try:
            idx = pd.to_datetime(out.index, errors="coerce")
            if idx.notna().mean() >= 0.80:
                out.index = idx
                out = out.dropna(how="all").sort_index()
                if out.index.has_duplicates:
                    out = out.groupby(level=0).sum(min_count=1)
        except Exception:
            pass

    return _lstm_downsample_df(out)


def _lstm_target_label_from_text(text: str = "", transform_mode: str = "raw") -> str:
    lower = str(text).lower()
    if transform_mode in {"pct_change", "log_return"}:
        return "Modeled Return"
    if "oil" in lower and ("rate" in lower or "production" in lower):
        return "Oil Production Rate (bbl/day)"
    if "water" in lower and "inject" in lower:
        return "Water Injection Rate (bbl/day)"
    if "water" in lower and "production" in lower:
        return "Water Production Rate (bbl/day)"
    if "gas" in lower and ("rate" in lower or "production" in lower):
        return "Gas Production Rate"
    if "pressure" in lower:
        return "Pressure"
    return "Value"


def _lstm_target_label_from_columns(cols: List[str], transform_mode: str = "raw") -> str:
    joined = " ".join([str(c) for c in cols])
    return _lstm_target_label_from_text(joined, transform_mode=transform_mode)


def _lstm_format_date_axis(fig, ax):
    try:
        fig.autofmt_xdate(rotation=25, ha="right")
    except Exception:
        pass
    return ax


def lstm_fig_line(df: pd.DataFrame, title: str, ylabel: str = "Value"):
    clean = _lstm_clean_timeseries_frame(df)
    fig, ax = _lstm_new_fig(LSTM_PLOT_WIDE, LSTM_PLOT_H)
    if clean.empty:
        ax.text(0.5, 0.5, "No numeric data to plot", ha="center", va="center", transform=ax.transAxes)
    else:
        for i, col in enumerate(clean.columns):
            color_cycle = [LSTM_LSTM_COLORS["blue"], LSTM_LSTM_COLORS["orange"], LSTM_LSTM_COLORS["green"], LSTM_LSTM_COLORS["purple"], LSTM_LSTM_COLORS["red"]]
            ax.plot(
                clean.index,
                clean[col],
                label=str(col),
                linewidth=1.55,
                color=color_cycle[i % len(color_cycle)],
                solid_capstyle="round",
            )
    _lstm_format_date_axis(fig, ax)
    _lstm_style_ax(ax, title, xlabel="Time", ylabel=ylabel, legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_training_history(history: Dict):
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    loss = np.asarray(history.get("loss", []), dtype=float)
    val_loss = np.asarray(history.get("val_loss", []), dtype=float)

    if len(loss):
        ax.plot(loss, label="Train Loss", linewidth=1.8, color=LSTM_LSTM_COLORS["blue"])
    if len(val_loss):
        ax.plot(val_loss, label="Validation Loss", linewidth=1.6, color=LSTM_LSTM_COLORS["orange"], linestyle="--")
        best_epoch = int(np.nanargmin(val_loss))
        ax.axvline(best_epoch, linestyle=":", linewidth=1.0, color=LSTM_LSTM_COLORS["gray"], label=f"Best epoch {best_epoch + 1}")

    positive = np.concatenate([loss[np.isfinite(loss)], val_loss[np.isfinite(val_loss)]])
    positive = positive[positive > 0]
    if len(positive) and positive.max() / max(positive.min(), 1e-12) > 25:
        ax.set_yscale("log")

    _lstm_style_ax(ax, "Training Loss vs Validation Loss", xlabel="Epoch", ylabel="Loss", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_actual_vs_predicted(actual: np.ndarray, pred: np.ndarray, title: str):
    a, p = _lstm_valid_pair(actual, pred)
    a_plot, p_plot = _lstm_downsample_pair(a, p)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if len(a_plot):
        ax.plot(a_plot, label="Actual", linewidth=1.65, color=LSTM_LSTM_COLORS["blue"])
        ax.plot(p_plot, label="LSTM Predicted", linewidth=1.55, color=LSTM_LSTM_COLORS["orange"], linestyle="--")
    _lstm_style_ax(ax, title, xlabel="Sample", ylabel=_lstm_target_label_from_text(title), legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_distribution(actual: np.ndarray, pred: np.ndarray, title: str):
    a, p = _lstm_valid_pair(actual, pred)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if len(a):
        bins = min(34, max(12, int(np.sqrt(len(a)))))
        ax.hist(a, bins=bins, alpha=0.48, label="Actual", color=LSTM_LSTM_COLORS["blue"], edgecolor="white", linewidth=0.35)
        ax.hist(p, bins=bins, alpha=0.48, label="Predicted", color=LSTM_LSTM_COLORS["orange"], edgecolor="white", linewidth=0.35)
    _lstm_style_ax(ax, title, xlabel=_lstm_target_label_from_text(title), ylabel="Frequency", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_scatter(actual: np.ndarray, pred: np.ndarray, title: str):
    a, p = _lstm_valid_pair(actual, pred)
    a_plot, p_plot = _lstm_downsample_pair(a, p, max_points=900)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if len(a_plot):
        ax.scatter(a_plot, p_plot, alpha=0.52, s=13, color=LSTM_LSTM_COLORS["blue"], linewidths=0)
        low = float(min(np.min(a_plot), np.min(p_plot)))
        high = float(max(np.max(a_plot), np.max(p_plot)))
        ax.plot([low, high], [low, high], linestyle="--", linewidth=1.0, color="#111827", label="Perfect fit")
    _lstm_style_ax(ax, "Actual vs Predicted Scatter", xlabel="Actual", ylabel="Predicted", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_residuals_vs_pred(pred: np.ndarray, residuals: np.ndarray, title: str):
    p = np.asarray(pred, dtype=float).reshape(-1)
    r = np.asarray(residuals, dtype=float).reshape(-1)
    mask = np.isfinite(p) & np.isfinite(r)
    p, r = p[mask], r[mask]
    if len(p) > 900:
        idx = np.linspace(0, len(p) - 1, 900).round().astype(int)
        p, r = p[idx], r[idx]

    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if len(p):
        ax.scatter(p, r, alpha=0.50, s=13, color=LSTM_LSTM_COLORS["blue"], linewidths=0)
        ax.axhline(0, linewidth=1.05, color="#111827", linestyle="--", label="Zero error")
        std = float(np.nanstd(r))
        if np.isfinite(std) and std > 0:
            ax.axhline(2 * std, linestyle=":", linewidth=0.9, color=LSTM_LSTM_COLORS["gray"], label="±2σ")
            ax.axhline(-2 * std, linestyle=":", linewidth=0.9, color=LSTM_LSTM_COLORS["gray"])
    _lstm_style_ax(ax, title, xlabel="Predicted", ylabel="Residual", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_residual_hist(residuals: np.ndarray, title: str):
    r = _lstm_numeric_array(residuals)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if len(r):
        bins = min(34, max(12, int(np.sqrt(len(r)))))
        ax.hist(r, bins=bins, alpha=0.78, color=LSTM_LSTM_COLORS["blue"], edgecolor="white", linewidth=0.4)
        ax.axvline(0, linestyle="--", linewidth=1.0, color="#111827", label="Zero error")
        ax.axvline(np.nanmean(r), linestyle=":", linewidth=1.0, color=LSTM_LSTM_COLORS["orange"], label="Mean")
    _lstm_style_ax(ax, "Residual Histogram", xlabel="Residual", ylabel="Frequency", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_qq(residuals: np.ndarray, title: str):
    r = _lstm_numeric_array(residuals)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if len(r) >= 3:
        stats.probplot(r, dist="norm", plot=ax)
        # Re-style scipy's default artists without changing the calculation.
        if len(ax.lines) >= 1:
            ax.lines[0].set_markerfacecolor(LSTM_LSTM_COLORS["blue"])
            ax.lines[0].set_markeredgecolor(LSTM_LSTM_COLORS["blue"])
            ax.lines[0].set_markersize(3.0)
            ax.lines[0].set_alpha(0.62)
        if len(ax.lines) >= 2:
            ax.lines[1].set_color(LSTM_LSTM_COLORS["orange"])
            ax.lines[1].set_linewidth(1.2)
    _lstm_style_ax(ax, title, xlabel="Theoretical Quantiles", ylabel="Ordered Residuals", legend=False)
    return _lstm_finish_fig(fig)


def lstm_fig_rolling_rmse(actual: np.ndarray, pred: np.ndarray, title: str, window: int = 30):
    a, p = _lstm_valid_pair(actual, pred)
    errors = (a - p) ** 2
    rolling = pd.Series(errors).rolling(window=window, min_periods=max(5, window // 3)).mean().pow(0.5)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    ax.plot(rolling.values, linewidth=1.6, color=LSTM_LSTM_COLORS["blue"])
    _lstm_style_ax(ax, title, xlabel="Sample", ylabel="Rolling RMSE", legend=False)
    return _lstm_finish_fig(fig)


def lstm_fig_correlation_heatmap(df: pd.DataFrame, title: str):
    corr = df.corr(numeric_only=True)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if not corr.empty:
        im = ax.imshow(corr.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=7)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _lstm_style_ax(ax, title, legend=False)
    return _lstm_finish_fig(fig)


def lstm_fig_missing_bars(df: pd.DataFrame, title: str):
    miss = df.isna().sum().sort_values(ascending=False)
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    ax.bar(miss.index.astype(str), miss.values, color=LSTM_LSTM_COLORS["blue"], alpha=0.82)
    ax.tick_params(axis="x", rotation=35)
    _lstm_style_ax(ax, title, ylabel="Missing Count", legend=False)
    return _lstm_finish_fig(fig)


def lstm_fig_split_overview(dates: pd.Series, train_end_idx: int, val_end_idx: int, title: str):
    dates = pd.to_datetime(pd.Series(dates), errors="coerce").dropna().reset_index(drop=True)
    fig, ax = _lstm_new_fig(LSTM_PLOT_WIDE, 1.55)
    if len(dates):
        train_end_idx = min(max(int(train_end_idx), 1), len(dates) - 1)
        val_end_idx = min(max(int(val_end_idx), train_end_idx + 1), len(dates) - 1)
        ax.axvspan(dates.iloc[0], dates.iloc[train_end_idx - 1], alpha=0.18, color=LSTM_LSTM_COLORS["blue"], label="Train")
        ax.axvspan(dates.iloc[train_end_idx], dates.iloc[val_end_idx - 1], alpha=0.18, color=LSTM_LSTM_COLORS["orange"], label="Validation")
        ax.axvspan(dates.iloc[val_end_idx], dates.iloc[-1], alpha=0.18, color=LSTM_LSTM_COLORS["green"], label="Test")
        ax.plot(dates, np.ones(len(dates)), alpha=0)
    ax.set_yticks([])
    _lstm_format_date_axis(fig, ax)
    _lstm_style_ax(ax, title, xlabel="Time", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_class_distribution(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], title: str):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    labels = np.arange(len(class_names))
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))

    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    x = np.arange(len(class_names))
    width = 0.38
    ax.bar(x - width / 2, true_counts, width=width, label="Actual", color=LSTM_LSTM_COLORS["blue"], alpha=0.82)
    ax.bar(x + width / 2, pred_counts, width=width, label="Predicted", color=LSTM_LSTM_COLORS["orange"], alpha=0.82)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=25, ha="right")
    _lstm_style_ax(ax, title, xlabel="Class", ylabel="Count", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_lstm_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], title: str):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    im = ax.imshow(cm, aspect="auto", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color="#111827")
    _lstm_style_ax(ax, title, xlabel="Predicted", ylabel="Actual", legend=False)
    return _lstm_finish_fig(fig)


def lstm_fig_class_probability_hist(y_prob: np.ndarray, y_pred: np.ndarray, class_names: List[str], title: str):
    probs = np.asarray(y_prob)
    pred = np.asarray(y_pred).astype(int).reshape(-1)
    conf = probs[np.arange(len(pred)), pred] if probs.ndim == 2 and len(pred) else np.array([])
    fig, ax = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    if len(conf):
        ax.hist(conf, bins=min(30, max(8, int(np.sqrt(len(conf))))), color=LSTM_LSTM_COLORS["blue"], alpha=0.78, edgecolor="white", linewidth=0.4)
        ax.axvline(np.nanmean(conf), color=LSTM_LSTM_COLORS["orange"], linestyle="--", linewidth=1.1, label="Mean confidence")
    _lstm_style_ax(ax, title, xlabel="Predicted Class Confidence", ylabel="Frequency", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_binary_roc_pr(y_true: np.ndarray, y_prob: np.ndarray, title_prefix: str):
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    score = np.asarray(y_prob)[:, 1]
    fig1, ax1 = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    try:
        fpr, tpr, _ = roc_curve(y_true, score)
        ax1.plot(fpr, tpr, color=LSTM_LSTM_COLORS["blue"], linewidth=1.7, label="ROC")
        ax1.plot([0, 1], [0, 1], linestyle="--", color=LSTM_LSTM_COLORS["gray"], linewidth=1.0, label="Random")
        _lstm_style_ax(ax1, f"{title_prefix} ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", legend=True)
    except Exception:
        _lstm_style_ax(ax1, f"{title_prefix} ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", legend=False)

    fig2, ax2 = _lstm_new_fig(LSTM_PLOT_W, LSTM_PLOT_H)
    try:
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax2.plot(recall, precision, color=LSTM_LSTM_COLORS["orange"], linewidth=1.7, label="PR")
        _lstm_style_ax(ax2, f"{title_prefix} Precision-Recall Curve", xlabel="Recall", ylabel="Precision", legend=True)
    except Exception:
        _lstm_style_ax(ax2, f"{title_prefix} Precision-Recall Curve", xlabel="Recall", ylabel="Precision", legend=False)
    return _lstm_finish_fig(fig1), _lstm_finish_fig(fig2)


def lstm_fig_equity_curves(equity_df: pd.DataFrame, title: str):
    clean = _lstm_downsample_df(equity_df.copy())
    fig, ax = _lstm_new_fig(LSTM_PLOT_WIDE, LSTM_PLOT_H)
    for i, col in enumerate(clean.columns):
        color_cycle = [LSTM_LSTM_COLORS["blue"], LSTM_LSTM_COLORS["orange"], LSTM_LSTM_COLORS["green"], LSTM_LSTM_COLORS["purple"]]
        ax.plot(clean.index, clean[col], label=col, linewidth=1.55, color=color_cycle[i % len(color_cycle)])
    _lstm_format_date_axis(fig, ax)
    _lstm_style_ax(ax, title, xlabel="Time", ylabel="Equity (Start = 1.0)", legend=True)
    return _lstm_finish_fig(fig)


def lstm_fig_drawdowns(equity_df: pd.DataFrame, title: str):
    clean = equity_df.copy()

    # Force all equity columns to numeric.
    # Any accidental string/date/text becomes NaN instead of crashing.
    clean = clean.apply(pd.to_numeric, errors="coerce")

    # Remove infinity and fill gaps safely.
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.ffill().bfill()

    # If a column is completely bad, remove it.
    clean = clean.dropna(axis=1, how="all")

    if clean.empty:
        raise ValueError("Drawdown plot failed because equity_df has no numeric columns.")

    # Avoid divide-by-zero.
    running_max = clean.cummax().replace(0, np.nan)
    dd = clean / running_max - 1.0
    dd = dd.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    for col in dd.columns:
        ax.plot(dd.index, dd[col], label=col, linewidth=1.4)

    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    ax.legend()
    return fig
