from __future__ import annotations

import streamlit as st

APP_TITLE = "Oil & Gas Deep Learning Platform"

MODEL_OPTIONS = ["ANN", "CNN", "LSTM"]

WORKFLOW_PAGES = [
    "Home",
    "Data Upload",
    "Model",
    "Preprocess",
    "Train",
    "Evaluate",
    "Predict",
    "Visualize",
    "Save/Load",
]


def model_color(model_name: str) -> str:
    colors = {
        "ANN": "#1e3a8a",   # dark blue
        "CNN": "#065f46",   # dark green
        "LSTM": "#4b5563",  # gray
    }
    return colors.get(model_name, colors["ANN"])


def model_gradient(model_name: str) -> tuple[str, str, str]:
    gradients = {
        "ANN": ("#0f172a", "#1d4ed8", "#0ea5e9"),
        "CNN": ("#052e16", "#166534", "#22c55e"),
        "LSTM": ("#111827", "#4b5563", "#9ca3af"),
    }
    return gradients.get(model_name, gradients["ANN"])


def configure_page_once() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_shell_css() -> None:
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 3.5rem !important;
                padding-bottom: 1.2rem;
            }

            div[data-testid="stHorizontalBlock"] {
                gap: 0.55rem;
            }

            .ns-shell-hero {
                background: linear-gradient(
                    135deg,
                    var(--model-c1) 0%,
                    var(--model-c2) 55%,
                    var(--model-c3) 100%
                );
                padding: 1.15rem 1.35rem;
                border-radius: 20px;
                color: white;
                margin-top: 4rem;
                margin-bottom: 0.85rem;
                box-shadow: 0 10px 30px rgba(2, 6, 23, 0.16);
                height: 130px;
                min-height: 130px;
                max-height: 130px;
                box-sizing: border-box;
                display: flex;
                flex-direction: column;
                justify-content: center;
                overflow: hidden;
            }

            .ns-shell-hero h2 {
                margin: 0 0 0.35rem 0;
                font-size: 2rem;
                line-height: 1.15;
                color: white;
            }

            .ns-shell-hero div {
                margin: 0;
                font-size: 1rem;
                line-height: 1.45;
                color: white;
            }

            .ns-shell-chip {
                display: inline-block;
                padding: 0.28rem 0.6rem;
                border-radius: 999px;
                background: #eff6ff;
                border: 1px solid #bfdbfe;
                color: #1d4ed8;
                font-size: 0.85rem;
                margin-right: 0.35rem;
                margin-bottom: 0.35rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_shell_header(model_name: str, page_name: str) -> None:
    c1, c2, c3 = model_gradient(model_name)

    st.markdown(
        f"""
        <style>
            :root {{
                --model-c1: {c1};
                --model-c2: {c2};
                --model-c3: {c3};
            }}
        </style>

        <div class="ns-shell-hero">
            <h2>Oil &amp; Gas Deep Learning Platform</h2>
            <div>
                A simple workspace for building ANN, CNN, and LSTM Deep Learning models.
            </div>
        </div>

        <span class="ns-shell-chip">Model: {model_name}</span>
        <span class="ns-shell-chip">Page: {page_name}</span>
        <span class="ns-shell-chip">Workflow: Guided</span>
        """,
        unsafe_allow_html=True,
    )
