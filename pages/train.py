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

from pages.home import (
    ann_inject_css,
    ann_hero,
    cnn_inject_css,
    cnn_hero,
    cnn_status_bar,
    cnn_render_uploaded_images_section,
    lstm_inject_css,
    lstm_hero,
    lstm_info_card,
    lstm_top_status_bar,
)

PAGE_NAME = "Train"


def render_ann_train_ui():
    st.title("Train")
    prepared = st.session_state.prepared_data

    if prepared is None:
        st.info("Prepare the dataset first.")
        return

    cfg = st.session_state.config

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

    st.markdown("### Training Controls")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        cfg["epochs"] = st.number_input(
            "Epochs",
            min_value=1,
            max_value=5000,
            value=int(cfg.get("epochs", 50)),
            step=1,
            key="ann_train_epochs",
        )

    with t2:
        cfg["batch_size"] = st.number_input(
            "Batch size",
            min_value=1,
            max_value=2048,
            value=int(cfg.get("batch_size", 32)),
            step=1,
            key="ann_train_batch_size",
        )

    with t3:
        cfg["learning_rate"] = st.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1.0,
            value=float(cfg.get("learning_rate", 0.001)),
            step=0.0001,
            format="%.6f",
            key="ann_train_learning_rate",
        )

    with t4:
        cfg["patience"] = st.number_input(
            "Early stopping patience",
            min_value=1,
            max_value=200,
            value=int(cfg.get("patience", 10)),
            step=1,
            key="ann_train_patience",
        )

    st.session_state.config = cfg

    with st.expander("Training configuration", expanded=False):
        config_rows = [
            {"setting": "Optimizer", "value": cfg["optimizer"]},
            {"setting": "Learning Rate", "value": cfg["learning_rate"]},
            {"setting": "Epochs", "value": cfg["epochs"]},
            {"setting": "Batch Size", "value": cfg["batch_size"]},
            {"setting": "Validation Split", "value": cfg["validation_split"]},
            {"setting": "Early Stopping", "value": cfg["early_stopping"]},
            {"setting": "Patience", "value": cfg["patience"]},
            {"setting": "Class Weights", "value": cfg["use_class_weights"]},
        ]

        st.dataframe(
            pd.DataFrame(config_rows),
            use_container_width=True,
            hide_index=True,
        )

    if st.button("Start Training", type="primary", use_container_width=True):
        try:
            model, history, results = ann_train_model(prepared, st.session_state.config)
            st.session_state.trained_model = model
            st.session_state.training_history = history
            st.session_state.results = results
            st.success("Model trained successfully.")
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.exception(e)

    if st.session_state.training_history:
        st.subheader("Latest training history")
        st.dataframe(
            pd.DataFrame(st.session_state.training_history),
            use_container_width=True,
        )


def render_cnn_train_ui():
    cnn_hero()
    cnn_status_bar()
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
        cfg["epochs_stage1"] = st.slider(
            "Stage 1 Epochs (Frozen Backbone)",
            1,
            40,
            int(cfg["epochs_stage1"]),
        )

    with c2:
        cfg["epochs_stage2"] = st.slider(
            "Stage 2 Epochs (Fine-Tuning)",
            0,
            30,
            int(cfg["epochs_stage2"]),
        )

    if st.button("Start Training", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            cnn_seed_everything(cfg["seed"])

            class_names = sorted(st.session_state.class_names)
            label_to_idx = {c: i for i, c in enumerate(class_names)}

            y_train = train_df["label"].map(label_to_idx).astype(int).to_numpy()
            y_val = val_df["label"].map(label_to_idx).astype(int).to_numpy()

            train_ds = cnn_make_tf_dataset(
                train_df["filepath"].tolist(),
                y_train,
                cfg,
                training=True,
            )

            val_ds = cnn_make_tf_dataset(
                val_df["filepath"].tolist(),
                y_val,
                cfg,
                training=False,
            )

            model, base_model = cnn_build_model(class_names, cfg)

            tf = cnn_get_tf()

            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=4,
                    restore_best_weights=True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.3,
                    patience=2,
                    min_lr=1e-7,
                ),
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
                    metrics=[
                        "accuracy",
                        tf.keras.metrics.SparseTopKCategoricalAccuracy(
                            k=min(3, len(class_names)),
                            name="top_k_acc",
                        ),
                    ],
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

            st.session_state.trained_model = model
            st.session_state.history = merged_history
            st.session_state.eval_artifacts = cnn_evaluate_model(model, val_df, class_names, cfg)
            st.session_state.training_complete = True

        st.success("Training complete.")

    if st.session_state.history is not None:
        cnn_plot_training_curves(st.session_state.history)


def render_lstm_train_ui():
    lstm_hero("Train", "Train the selected LSTM task using the already-preprocessed sequence data.")

    processed = st.session_state.get("processed")
    if processed is None:
        st.info("Preprocess data first.")
        return

    cfg = st.session_state["config"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        cfg["epochs"] = st.number_input(
            "Epochs",
            min_value=1,
            max_value=1000,
            value=int(cfg["epochs"]),
            step=1,
        )

    with c2:
        cfg["batch_size"] = st.number_input(
            "Batch size",
            min_value=1,
            max_value=1024,
            value=int(cfg["batch_size"]),
            step=1,
        )

    with c3:
        cfg["learning_rate"] = st.number_input(
            "Learning rate",
            min_value=1e-6,
            max_value=1.0,
            value=float(cfg["learning_rate"]),
            format="%.6f",
        )

    with c4:
        cfg["patience"] = st.number_input(
            "Early stopping patience",
            min_value=1,
            max_value=200,
            value=int(cfg["patience"]),
            step=1,
        )

    if processed.get("task") == "regression":
        cfg["loss"] = st.selectbox(
            "Regression loss",
            ["mse", "mae", "huber"],
            index=["mse", "mae", "huber"].index(cfg["loss"]),
        )

    train_now = st.button("Start Training", type="primary", use_container_width=True)

    if train_now:
        try:
            with st.spinner("Training model..."):
                training = lstm_train_model(processed, cfg)

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
    t1.metric("Task", processed["task"].title())
    t2.metric("Train Sequences", len(processed["X_train"]))
    t3.metric("Validation Sequences", len(processed["X_val"]))
    t4.metric("Test Sequences", len(processed["X_test"]))

    t5, t6, t7, t8 = st.columns(4)
    t5.metric("Input Shape", f"{processed['X_train'].shape[1]} × {processed['X_train'].shape[2]}")
    t6.metric("Batch Size", int(cfg["batch_size"]))
    t7.metric("Learning Rate", f"{float(cfg['learning_rate']):.6f}")
    t8.metric("Patience", int(cfg["patience"]))

    hist_left, hist_center, hist_right = st.columns([1, 2.4, 1])

    with hist_center:
        st.pyplot(
            lstm_fig_training_history(training["history"]),
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown("**Model summary**")
        st.code(training["model_summary"], language="text")

    st.write("**Validation metrics**")
    st.dataframe(
        training["metrics"]["val"],
        use_container_width=True,
        hide_index=True,
    )


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
            return render_ann_train_ui()

        if model_name == "CNN":
            return render_cnn_train_ui()

        if model_name == "LSTM":
            return render_lstm_train_ui()

        st.error(f"Unknown model: {model_name}")
