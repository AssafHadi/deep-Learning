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

PAGE_NAME = "Home"
def inner_set_page_config(*args, **kwargs):
    """Compatibility no-op.

    Page config is owned once by core.config.configure_page_once().
    Old extracted code may still call inner_set_page_config; this prevents a crash.
    """
    return None

def ann_inject_css():
    # Page config is owned once by core.config.configure_page_once().
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


def ann_hero():
    st.markdown(
        f"""
        <div class="hero">
            <h1>{ANN_APP_TITLE}</h1>
            <p>{ANN_APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ann_home_ui():
    ann_hero()
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


def cnn_inject_css() -> None:
    # Page config is owned once by core.config.configure_page_once().

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


def cnn_status_bar():
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


def cnn_hero():
    st.markdown(
        f"""
        <div class="hero">
            <h2 style="margin:0 0 0.35rem 0;">{CNN_APP_TITLE}</h2>
            <div style="font-size:1rem; line-height:1.55;">
                Professional CNN workflow for oil & gas image classification:
                seismic slices, core photos, rock thin sections, corrosion images,
                facility inspections, and other 2D visual datasets.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def cnn_render_uploaded_images_section(
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
                img, err = cnn_open_uploaded_image(up)
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
            img, err = cnn_open_uploaded_image(up)
            if err:
                raise ValueError(err)
            clean_img = cnn_ensure_rgb(img, "RGB" if cfg["color_mode"] == "RGB" else "GRAY")
            arr = cnn_prepare_single_image(clean_img, cfg)
            probs = model.predict(arr, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            pred_name = class_names[pred_idx]
            top_indices = np.argsort(probs)[::-1][:min(5, len(class_names))]
            heatmap = cnn_gradcam_heatmap(model, arr) if show_gradcam else None
            overlay = None
            if heatmap is not None:
                overlay = cnn_overlay_heatmap_on_image(
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
            cnn_fig_show(fig)

            if row.get("overlay") is not None:
                st.write("**Grad-CAM**")
                st.image(row["overlay"], use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_cnn_home_ui():
    cnn_hero()
    cnn_status_bar()

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


def lstm_inject_css() -> None:
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


def lstm_hero(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def lstm_info_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="section-title">{title}</div>
            <div class="small-muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def lstm_top_status_bar():
    raw_df = st.session_state.get("raw_df")
    processed = st.session_state.get("processed")
    training = st.session_state.get("training")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Loaded", 0 if raw_df is None else len(raw_df))
    c2.metric("Features Selected", len(st.session_state["config"].get("feature_cols", [])))
    c3.metric("Targets Selected", len(st.session_state["config"].get("target_cols", [])))
    c4.metric("Model Status", "Ready" if training is not None else ("Preprocessed" if processed is not None else "Not Trained"))


def render_lstm_home_ui():
    lstm_hero(LSTM_APP_TITLE, LSTM_APP_SUBTITLE)
    lstm_top_status_bar()
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
        lstm_info_card("Workflow", "Home → Data Upload → Model → Preprocess → Train → Evaluate → Predict → Visualize → Save/Load")
    with c2:
        lstm_info_card("Best Use Case", "Daily or weekly oil, gas, refined product, or energy-market datasets with a date column and multiple numeric series")
    with c3:
        lstm_info_card("Built For", "WTI, Brent, Natural Gas, Heating Oil, Diesel, LPG, or any structured commodity time series with enough history")

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
            return render_ann_home_ui()
        if model_name == "CNN":
            return render_cnn_home_ui()
        if model_name == "LSTM":
            return render_lstm_home_ui()
        st.error(f"Unknown model: {model_name}")
