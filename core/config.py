from __future__ import annotations

import streamlit as st

APP_TITLE = "Oil & Gas Neural Studio"
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
            .main .block-container {padding-top: 1.0rem; padding-bottom: 1.2rem;}
            div[data-testid="stHorizontalBlock"] {gap: 0.55rem;}
            .ns-shell-hero {
                background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
                padding: 1.15rem 1.35rem;
                border-radius: 20px;
                color: white;
                margin-bottom: 0.85rem;
                box-shadow: 0 10px 30px rgba(2,6,23,0.16);
            }
            .ns-shell-chip {
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


def render_shell_header(model_name: str, page_name: str) -> None:
    st.markdown(
        f"""
        <div class="ns-shell-hero">
            <h2 style="margin:0 0 0.35rem 0;">Oil &amp; Gas Neural Studio</h2>
            <div style="font-size:1rem; line-height:1.55;">
                Professional layered Streamlit architecture for ANN, CNN, and LSTM workflows.
            </div>
        </div>
        <span class='ns-shell-chip'>Model: {model_name}</span>
        <span class='ns-shell-chip'>Page: {page_name}</span>
        <span class='ns-shell-chip'>Architecture: Layered</span>
        """,
        unsafe_allow_html=True,
    )
