from __future__ import annotations

from importlib import import_module

import streamlit as st

from core.config import MODEL_OPTIONS, WORKFLOW_PAGES
from core.state import get_selected_page, reset_model_project, set_selected_page, switch_model, model_state_context

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


def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown("## Oil & Gas Neural Studio")
        st.caption("True layered architecture")
        model_name = st.selectbox("Active Model", MODEL_OPTIONS, key="__ns_selected_model")
        switch_model(model_name)

        selected = get_selected_page()
        page_name = st.radio(
            "Workflow",
            WORKFLOW_PAGES,
            index=WORKFLOW_PAGES.index(selected) if selected in WORKFLOW_PAGES else 0,
            key="__ns_workflow_radio",
        )
        set_selected_page(page_name)

        st.markdown("---")
        render_status(model_name)
        st.markdown("---")
        render_project_actions(model_name)
    return model_name, page_name


def render_status(model_name: str) -> None:
    with model_state_context(model_name):
        st.markdown(f"### {model_name} Status")

        if model_name == "ANN":
            data_ready = st.session_state.get("raw_df") is not None
            prepared = st.session_state.get("prepared_data") is not None
            trained = st.session_state.get("trained_model") is not None
            st.markdown(f'<span class="status-pill">Dataset: {"Ready" if data_ready else "Missing"}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="status-pill">Prepared: {"Yes" if prepared else "No"}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="status-pill">Model: {"Trained/Loaded" if trained else "None"}</span>', unsafe_allow_html=True)

        elif model_name == "CNN":
            data_ready = st.session_state.get("dataset_df") is not None
            trained = st.session_state.get("trained_model") is not None
            st.markdown(f'<span class="status-pill">Dataset: {"Ready" if data_ready else "Missing"}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="status-pill">Model: {"Trained/Loaded" if trained else "None"}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="status-pill">Classes: {len(st.session_state.get("class_names", []))}</span>', unsafe_allow_html=True)

        elif model_name == "LSTM":
            data_ready = st.session_state.get("raw_df") is not None
            prepared = st.session_state.get("processed") is not None
            training = st.session_state.get("training")
            trained = isinstance(training, dict) and training.get("model") is not None
            st.markdown(f'<span class="status-pill">Dataset: {"Ready" if data_ready else "Missing"}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="status-pill">Prepared: {"Yes" if prepared else "No"}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="status-pill">Model: {"Trained/Loaded" if trained else "None"}</span>', unsafe_allow_html=True)


def render_project_actions(model_name: str) -> None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New", use_container_width=True, key=f"__ns_new_{model_name}"):
            reset_model_project(model_name, "Data Upload")
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, key=f"__ns_clear_{model_name}"):
            reset_model_project(model_name, "Home")


def render_current_page(model_name: str, page_name: str) -> None:
    module = import_module(_PAGE_MODULES.get(page_name, "pages.home"))
    module.render(model_name)


def render_bottom_navigation() -> None:
    current = get_selected_page()
    idx = WORKFLOW_PAGES.index(current) if current in WORKFLOW_PAGES else 0
    st.write("")
    left, right = st.columns(2, gap="large")
    with left:
        if idx == 0:
            st.button("⬅ Back", disabled=True, use_container_width=True, key="__ns_back_disabled")
        else:
            prev_page = WORKFLOW_PAGES[idx - 1]
            if st.button(f"⬅ Back to {prev_page}", use_container_width=True, key=f"__ns_back_{prev_page}"):
                set_selected_page(prev_page)
                st.rerun()
    with right:
        if idx >= len(WORKFLOW_PAGES) - 1:
            st.button("Continue ➜", disabled=True, use_container_width=True, key="__ns_next_disabled")
        else:
            next_page = WORKFLOW_PAGES[idx + 1]
            if st.button(f"Continue to {next_page} ➜", use_container_width=True, key=f"__ns_next_{next_page}"):
                set_selected_page(next_page)
                st.rerun()
