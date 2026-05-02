from __future__ import annotations

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
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # pragma: no cover
    Image = ImageOps = UnidentifiedImageError = None

from core.state import model_state_context
from models.ann import *
from models.cnn import *
from models.lstm import *
from services.preprocessing import *
from services.training import *
from services.evaluation import *
from services.prediction import *
from visualization.plots import *
from storage.persistence import *

from pages.home import ann_inject_css, ann_hero, cnn_inject_css, cnn_hero, cnn_status_bar, cnn_render_uploaded_images_section, lstm_inject_css, lstm_hero, lstm_info_card, lstm_top_status_bar

PAGE_NAME = "Evaluate"

def render_ann_evaluate_ui():
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
            st.pyplot(ann_make_confusion_fig(y_true, y_pred, class_names), clear_figure=True, use_container_width=True)
        with c2:
            st.pyplot(ann_make_predicted_distribution_fig(y_pred), clear_figure=True, use_container_width=True)

        report_df = pd.DataFrame(results["classification_report"]).T
        st.subheader("Classification Report")
        st.dataframe(report_df, use_container_width=True)

        if len(class_names) == 2:
            fig1, fig2 = ann_make_binary_curve_figs(y_true, results.get("y_score"))
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

        fig1, fig2, fig3, fig4 = ann_make_regression_figures(y_true, y_pred)
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


def render_cnn_evaluate_ui():
    cnn_hero()
    cnn_status_bar()
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

    cnn_plot_confusion_matrices(ev["confusion_matrix"], ev["class_names"])
    cnn_plot_per_class_metrics(ev["classification_report"], ev["class_names"])
    cnn_plot_prediction_confidence_distribution(ev)
    cnn_plot_correct_incorrect_summary(ev)

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

    cnn_plot_roc_pr(ev)
    cnn_show_misclassified(ev, max_items=12)
    cnn_show_correct_incorrect_examples(ev, max_each=4)


def render_lstm_evaluate_ui():
    lstm_hero("Evaluate", "Evaluate the selected LSTM task with the right metrics. Regression and classification are not mixed.")
    training = st.session_state.get("training")
    processed = st.session_state.get("processed")
    if training is None or processed is None:
        st.info("Train the model first.")
        return

    cfg = st.session_state["config"]
    split = st.selectbox("Evaluation split", ["train", "val", "test"], index=2)

    if processed.get("task") == "classification":
        class_names = processed["class_names"]
        y_true = training["actuals"][split]
        y_pred = training["predictions"][split]
        y_prob = training["probabilities"][split]

        metrics_df = training["metrics"][split]
        cols = st.columns(min(4, len(metrics_df)))
        for i, row in metrics_df.head(4).iterrows():
            cols[i].metric(str(row["metric"]), f"{float(row['value']):.4f}")

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        a, b = st.columns(2)
        with a:
            lstm_show_lstm_fig(lstm_fig_lstm_confusion(y_true, y_pred, class_names, f"{split.title()} Confusion Matrix"))
        with b:
            lstm_show_lstm_fig(lstm_fig_class_distribution(y_true, y_pred, class_names, f"{split.title()} Actual vs Predicted Class Distribution"))

        c, d = st.columns(2)
        with c:
            lstm_show_lstm_fig(lstm_fig_class_probability_hist(y_prob, y_pred, class_names, f"{split.title()} Prediction Confidence"))
        with d:
            report = pd.DataFrame(classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)).T
            st.dataframe(report, use_container_width=True)

        if len(class_names) == 2:
            fig1, fig2 = lstm_fig_binary_roc_pr(y_true, y_prob, split.title())
            e, f = st.columns(2)
            with e:
                lstm_show_lstm_fig(fig1)
            with f:
                lstm_show_lstm_fig(fig2)
        return

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
        lstm_show_lstm_fig(lstm_fig_actual_vs_predicted(actual, pred, f"{split.title()} | {target} | Step {horizon_step}: Actual vs Predicted"))
    with b:
        lstm_show_lstm_fig(lstm_fig_scatter(actual, pred, f"{split.title()} | {target} | Scatter"))

    c, d = st.columns(2)
    with c:
        lstm_show_lstm_fig(lstm_fig_residuals_vs_pred(pred, residuals, f"{split.title()} | {target} | Residuals vs Predicted"))
    with d:
        lstm_show_lstm_fig(lstm_fig_residual_hist(residuals, f"{split.title()} | {target} | Residual Histogram"))

    e, f = st.columns(2)
    with e:
        lstm_show_lstm_fig(lstm_fig_distribution(actual, pred, f"{split.title()} | {target} | Actual vs Predicted Distribution"))
    with f:
        lstm_show_lstm_fig(lstm_fig_qq(residuals, f"{split.title()} | {target} | Residual QQ Plot"))


def _apply_model_ui(model_name: str) -> None:
    if model_name == "ANN":
        ann_inject_css()
    elif model_name == "CNN":
        cnn_inject_css()
    elif model_name == "LSTM":
        lstm_inject_css()


def render(model_name: str) -> None:
    with model_state_context(model_name):
        _apply_model_ui(model_name)
        if model_name == "ANN":
            return render_ann_evaluate_ui()
        if model_name == "CNN":
            return render_cnn_evaluate_ui()
        if model_name == "LSTM":
            return render_lstm_evaluate_ui()
        st.error(f"Unknown model: {model_name}")
