from __future__ import annotations

from importlib import import_module

import streamlit as st

from core.config import MODEL_OPTIONS, WORKFLOW_PAGES
from core.state import (
    get_selected_page,
    reset_model_project,
    set_selected_page,
    switch_model,
)


_PAGE_MODULES = {
    "Home": "pages.home",
    "Data Upload": "pages.data",
    "Model": "pages.model",
    "Preprocess": "pages.preprocess",
    "Train": "pages.train",
    "Evaluate": "pages.evaluate",
    "Predict": "pages.predict",
    "Visualize": "pages.visualize",
    "Save/Load": "pages.save_load",
}


def _model_radio_color(model_name: str) -> str:
    colors = {
        "ANN": "#1e3a8a",   # dark blue
        "CNN": "#065f46",   # dark green
        "LSTM": "#4b5563",  # gray
    }
    return colors.get(model_name, colors["ANN"])


def _inject_sidebar_radio_css(model_name: str) -> None:
    color = _model_radio_color(model_name)

    st.markdown(
        f"""
        <style>
            section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
                border-color: {color} !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label,
            section[data-testid="stSidebar"] div[role="radiogroup"] label * {{
                color: #0f172a !important;
                font-weight: 400 !important;
                background: transparent !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {{
                width: 16px !important;
                height: 16px !important;
                min-width: 16px !important;
                min-height: 16px !important;
                border-radius: 50% !important;
                border: 1.5px solid #9ca3af !important;
                background-color: white !important;
                box-sizing: border-box !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                margin-right: 0.45rem !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {{
                border-color: {color} !important;
                background-color: white !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:has(input:checked) > div:first-child > div {{
                width: 8px !important;
                height: 8px !important;
                min-width: 8px !important;
                min-height: 8px !important;
                border-radius: 50% !important;
                background-color: {color} !important;
                display: block !important;
            }}

            section[data-testid="stSidebar"] div[role="radiogroup"] label[data-baseweb="radio"]:not(:has(input:checked)) > div:first-child > div {{
                background-color: transparent !important;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _sync_sidebar_page() -> None:
    page_name = st.session_state.get("__ns_workflow_radio", "Home")
    set_selected_page(page_name)


def _goto_page(page_name: str) -> None:
    set_selected_page(page_name)
    st.session_state["__ns_workflow_radio"] = page_name


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown("## Deep Learning Platform")
        st.caption("Choose a model and follow the workflow.")

        model_name = st.selectbox(
            "Active Model",
            MODEL_OPTIONS,
            key="__ns_selected_model",
        )

        switch_model(model_name)
        _inject_sidebar_radio_css(model_name)

        selected_page = get_selected_page()

        if st.session_state.get("__ns_workflow_radio") != selected_page:
            st.session_state["__ns_workflow_radio"] = selected_page

        st.radio(
            "Workflow",
            WORKFLOW_PAGES,
            index=WORKFLOW_PAGES.index(selected_page)
            if selected_page in WORKFLOW_PAGES
            else 0,
            key="__ns_workflow_radio",
            on_change=_sync_sidebar_page,
        )

        page_name = get_selected_page()

        st.markdown("---")
        render_project_actions(model_name)

    return model_name, page_name


def render_project_actions(model_name: str) -> None:
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "➕ New",
            use_container_width=True,
            key=f"__ns_new_{model_name}",
        ):
            reset_model_project(model_name, "Data Upload")

    with col2:
        if st.button(
            "🗑️ Clear",
            use_container_width=True,
            key=f"__ns_clear_{model_name}",
        ):
            reset_model_project(model_name, "Home")


def render_current_page(model_name: str, page_name: str) -> None:
    module_path = _PAGE_MODULES.get(page_name, "pages.home")
    module = import_module(module_path)
    module.render(model_name)


def render_bottom_navigation() -> None:
    current_page = get_selected_page()
    idx = WORKFLOW_PAGES.index(current_page) if current_page in WORKFLOW_PAGES else 0

    st.write("")
    left, right = st.columns(2, gap="large")

    with left:
        if idx == 0:
            st.button(
                "⬅ Back",
                disabled=True,
                use_container_width=True,
                key="__ns_back_disabled",
            )
        else:
            prev_page = WORKFLOW_PAGES[idx - 1]
            st.button(
                f"⬅ Back to {prev_page}",
                use_container_width=True,
                key=f"__ns_back_{current_page}_to_{prev_page}",
                on_click=_goto_page,
                args=(prev_page,),
            )

    with right:
        if idx >= len(WORKFLOW_PAGES) - 1:
            st.button(
                "Continue ➜",
                disabled=True,
                use_container_width=True,
                key="__ns_next_disabled",
            )
        else:
            next_page = WORKFLOW_PAGES[idx + 1]
            st.button(
                f"Continue to {next_page} ➜",
                use_container_width=True,
                key=f"__ns_next_{current_page}_to_{next_page}",
                on_click=_goto_page,
                args=(next_page,),
            )
