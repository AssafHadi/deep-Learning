from __future__ import annotations

import io
from typing import Any

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

from core.config import model_gradient
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


MODEL_CONTENT = {
    "ANN": {
        "title": "Oil & Gas ANN Model",
        "subtitle": "Deep Learning for structured oil and gas data.",
        "description": (
            "Use ANN for table-based oil and gas data such as production records, "
            "pressure values, drilling parameters, equipment readings, and field operation data."
        ),
        "metric_labels": ("Rows", "Columns", "Target", "Task"),
        "cards": [
            ("Data Upload", "Load CSV or Excel data and review the dataset."),
            ("Model Setup", "Choose the model structure and training settings."),
            ("Preprocess", "Select inputs, targets, scaling, and cleaning options."),
            ("Train", "Train the model using the prepared dataset."),
            ("Evaluate", "Check performance, errors, and prediction quality."),
            ("Predict", "Generate predictions from new field data."),
        ],
        "examples": [
            "Production forecasting from well and reservoir data.",
            "Pressure prediction from operating conditions.",
            "Water-cut estimation from production features.",
            "Equipment status classification from field records.",
        ],
    },
    "CNN": {
        "title": "Oil & Gas CNN Model",
        "subtitle": "Deep Learning for oil and gas image data.",
        "description": (
            "Use CNN for image-based oil and gas tasks such as equipment photos, "
            "inspection images, rock samples, core images, seismic slices, and visual classification."
        ),
        "metric_labels": ("Samples", "Classes", "Input", "Task"),
        "cards": [
            ("Data Upload", "Load a ZIP image dataset and review the classes."),
            ("Model Setup", "Choose image size, training settings, and augmentation."),
            ("Preprocess", "Prepare labels, folders, and validation data."),
            ("Train", "Train the model using the image dataset."),
            ("Evaluate", "Check accuracy, class results, and prediction quality."),
            ("Predict", "Upload new images and predict their classes."),
        ],
        "examples": [
            "Classifying equipment images by condition.",
            "Sorting inspection images into defect categories.",
            "Classifying rock, core, or seismic image groups.",
            "Organizing visual field records into useful categories.",
        ],
    },
    "LSTM": {
        "title": "Oil & Gas LSTM Model",
        "subtitle": "Deep Learning for time-based oil and gas data.",
        "description": (
            "Use LSTM for ordered data where time matters, such as production history, "
            "pressure trends, sensor readings, operating sequences, and forecasting tasks."
        ),
        "metric_labels": ("Rows", "Features", "Targets", "Task"),
        "cards": [
            ("Data Upload", "Load time-based CSV or Excel data."),
            ("Model Setup", "Choose sequence length, forecast horizon, and training settings."),
            ("Preprocess", "Select date, features, targets, scaling, and sequence options."),
            ("Train", "Train the model using the prepared sequence data."),
            ("Evaluate", "Check forecast results, errors, and trend quality."),
            ("Predict", "Generate future values or sequence-based predictions."),
        ],
        "examples": [
            "Forecasting production trends from historical records.",
            "Predicting pressure changes over time.",
            "Using sensor readings to estimate future behavior.",
            "Finding patterns in ordered field data.",
        ],
    },
}


def inner_set_page_config(*args, **kwargs):
    return None


def _inject_home_css(model_name: str) -> None:
    c1, c2, c3 = model_gradient(model_name)

    st.markdown(
        f"""
        <style>
            .block-container {{
                padding-top: 1.3rem;
                padding-bottom: 2rem;
                max-width: 1320px;
            }}

            .home-hero {{
                width: 100%;
                height: 130px;
                min-height: 130px;
                max-height: 130px;
                padding: 1.15rem 1.35rem;
                border-radius: 20px;
                background: linear-gradient(135deg, {c1} 0%, {c2} 55%, {c3} 100%);
                color: white;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(2,6,23,0.16);
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: center;
                overflow: hidden;
            }}

            .home-hero h2 {{
                margin: 0 0 0.35rem 0;
                font-size: 2rem;
                line-height: 1.15;
                color: white;
            }}

            .home-hero div {{
                margin: 0;
                font-size: 1rem;
                line-height: 1.45;
                color: white;
            }}

            .home-note {{
                padding: 0.8rem 1rem;
                border-left: 4px solid {c2};
                background: #eff6ff;
                border-radius: 10px;
                margin-bottom: 1rem;
                line-height: 1.65;
                color: #0f172a;
            }}

            .home-card {{
                height: 145px;
                min-height: 145px;
                max-height: 145px;
                background: #ffffff;
                border: 1px solid rgba(15,23,42,0.08);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 6px 22px rgba(15,23,42,0.05);
                box-sizing: border-box;
                overflow: hidden;
                margin-bottom: 0.8rem;
            }}

            .home-card h4 {{
                margin: 0 0 0.55rem 0;
                color: #0f172a;
                font-size: 1.2rem;
                line-height: 1.2;
            }}

            .home-card p {{
                margin: 0;
                font-size: 0.9rem;
                color: #475569;
                line-height: 1.6;
            }}

            .home-section-title {{
                margin-top: 1.25rem;
                margin-bottom: 0.55rem;
                font-size: 1.55rem;
                color: #0f172a;
                font-weight: 700;
            }}

            .home-list {{
                margin-top: 0.5rem;
                color: #334155;
                line-height: 1.65;
            }}

            .home-list li {{
                margin-bottom: 0.35rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_values(model_name: str) -> tuple[str, str, str, str]:
    if model_name == "ANN":
        df = st.session_state.get("raw_df")
        cfg = st.session_state.get("config", {})
        results = st.session_state.get("results", {})

        return (
            "0" if df is None else str(len(df)),
            "0" if df is None else str(len(df.columns)),
            cfg.get("target_column") or "Not set",
            results.get("task") or "Not trained",
        )

    if model_name == "CNN":
        df = st.session_state.get("dataset_df")
        class_names = st.session_state.get("class_names", [])
        trained_model = st.session_state.get("trained_model")

        return (
            "0" if df is None else str(len(df)),
            str(len(class_names)),
            "ZIP images",
            "Trained" if trained_model is not None else "Not trained",
        )

    if model_name == "LSTM":
        df = st.session_state.get("raw_df")
        cfg = st.session_state.get("config", {})
        training = st.session_state.get("training")

        trained = False
        if isinstance(training, dict):
            trained = training.get("model") is not None
        trained = trained or st.session_state.get("model") is not None

        return (
            "0" if df is None else str(len(df)),
            str(len(cfg.get("feature_cols", []) or [])),
            str(len(cfg.get("target_cols", []) or [])),
            "Trained" if trained else "Not trained",
        )

    return "0", "0", "Not set", "Not trained"


def _render_hero(content: dict) -> None:
    st.markdown(
        f"""
        <div class="home-hero">
            <h2>{content["title"]}</h2>
            <div>{content["subtitle"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_description(content: dict) -> None:
    st.markdown(
        f"""
        <div class="home-note">
            {content["description"]}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_metrics(model_name: str, content: dict) -> None:
    labels = content["metric_labels"]
    values = _metric_values(model_name)

    cols = st.columns(4)

    for col, label, value in zip(cols, labels, values):
        with col:
            st.metric(label, value)


def _render_cards(content: dict) -> None:
    cards = content["cards"]

    for start in range(0, len(cards), 3):
        cols = st.columns(3)

        for col, card in zip(cols, cards[start:start + 3]):
            title, body = card
            with col:
                st.markdown(
                    f"""
                    <div class="home-card">
                        <h4>{title}</h4>
                        <p>{body}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_examples(content: dict) -> None:
    st.markdown(
        '<div class="home-section-title">Recommended oil & gas examples</div>',
        unsafe_allow_html=True,
    )

    for example in content["examples"]:
        st.markdown(f"- {example}")


def render_home_ui(model_name: str) -> None:
    if model_name not in MODEL_CONTENT:
        model_name = "ANN"

    _inject_home_css(model_name)

    content = MODEL_CONTENT[model_name]

    _render_hero(content)
    _render_description(content)
    _render_metrics(model_name, content)
    _render_cards(content)
    _render_examples(content)


def render_ann_home_ui() -> None:
    render_home_ui("ANN")


def render_cnn_home_ui() -> None:
    render_home_ui("CNN")


def render_lstm_home_ui() -> None:
    render_home_ui("LSTM")


def render(model_name: str) -> None:
    with model_state_context(model_name):
        render_home_ui(model_name)


def cnn_render_uploaded_images_section(
    uploader_label: str = "Upload image(s)",
    uploader_key: str = "shared_image_uploader",
    show_title: bool = True,
    show_gradcam: bool = True,
):
    cfg = st.session_state.get("model_config")
    model = st.session_state.get("trained_model")
    class_names = st.session_state.get("class_names", [])

    uploaded_imgs = st.file_uploader(
        uploader_label,
        accept_multiple_files=True,
        key=uploader_key,
        help="JPG, JPEG, PNG, BMP, TIFF, WEBP, and GIF work best.",
    )

    if not uploaded_imgs:
        return

    st.session_state.last_uploaded_images = [up.name for up in uploaded_imgs]

    if show_title:
        st.write("### Uploaded Images")

    if model is None or not st.session_state.get("training_complete") or not class_names:
        st.info("Images uploaded successfully. Prediction is locked until you train or load a model.")
        preview_cols = st.columns(5, gap="small")

        for i, up in enumerate(uploaded_imgs):
            try:
                if Image is None:
                    raise RuntimeError("Pillow is not available.")
                img = Image.open(up)
                preview_cols[i % len(preview_cols)].image(
                    img,
                    caption=up.name,
                    use_container_width=True,
                )
            except Exception as e:
                preview_cols[i % len(preview_cols)].error(f"{up.name}: {e}")

        return

    st.warning(
        "Image prediction is available from the Predict page after a CNN model is trained or loaded."
    )
