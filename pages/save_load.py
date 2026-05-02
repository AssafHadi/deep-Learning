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

PAGE_NAME = "Save/Load"

def render_ann_save_load_ui():
    st.title("Save / Load")
    st.caption("Export the whole ANN project or reload a previous bundle.")

    c1, c2 = st.columns(2)
    with c1:
        if st.session_state.trained_model is not None and st.session_state.prepared_data is not None:
            try:
                bundle_bytes = ann_create_project_bundle_bytes()
                st.download_button(
                    "Download Project Bundle (.zip)",
                    data=bundle_bytes,
                    file_name=f"{ann_safe_filename(st.session_state.config['project_name'])}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Failed to package project: {e}")
        else:
            st.info("Train or load a project to enable export.")

        if st.button("Save Project Locally", use_container_width=True):
            try:
                path = ann_save_project_locally()
                st.success(f"Saved to {path}")
            except Exception as e:
                st.error(f"Local save failed: {e}")

    with c2:
        uploaded = st.file_uploader("Load bundle (.zip)", type=["zip"], key="load_bundle")
        if uploaded is not None:
            try:
                ann_load_project_from_zip_bytes(uploaded.read())
                st.success("Project loaded successfully.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.subheader("Local Saved Bundles")
    local_files = sorted(ANN_PROJECTS_DIR.glob("*.zip"), reverse=True)
    if local_files:
        df = pd.DataFrame({
            "file": [p.name for p in local_files],
            "size_kb": [round(p.stat().st_size / 1024, 2) for p in local_files],
            "modified": [datetime.fromtimestamp(p.stat().st_mtime).isoformat(sep=" ", timespec="seconds") for p in local_files],
        })
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No local bundles found yet.")


def render_cnn_save_load_ui():
    cnn_hero()
    cnn_status_bar()
    st.subheader("Save / Load")

    col1, col2 = st.columns(2, gap="small")

    with col1:
        st.write("### Save Current Project")
        st.session_state.project_name = st.text_input("Save Name", value=st.session_state.project_name, key="save_name_input")
        if st.button("Save Project", type="primary", use_container_width=True):
            try:
                project_dir = cnn_save_project(st.session_state.project_name)
                st.success(f"Project saved to: {project_dir}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with col2:
        st.write("### Load Existing Project")
        projects = cnn_available_projects()
        if not projects:
            st.info("No saved projects found yet.")
        else:
            choice = st.selectbox("Saved Projects", projects, format_func=lambda p: p.name)
            if st.button("Load Selected Project", use_container_width=True):
                try:
                    cnn_load_project(choice)
                    st.success(f"Loaded: {choice.name}")
                except Exception as e:
                    st.error(f"Load failed: {e}")

    st.write("### Saved Project Inventory")
    rows = []
    for p in cnn_available_projects():
        rows.append({
            "project_dir": str(p),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "has_model": (p / "model.keras").exists(),
            "has_metadata": (p / "metadata.joblib").exists(),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


def render_lstm_save_load_ui():
    lstm_hero("Save / Load", "Persist the full project so you do not have to rebuild everything from scratch every time.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Save current project")
        if st.session_state.get("training") is None:
            st.info("Train a model before saving.")
        else:
            try:
                bundle_bytes = lstm_make_bundle_bytes()
                file_name = f"{st.session_state['config']['project_name']}.zip"
                st.download_button(
                    "Download project bundle",
                    data=bundle_bytes,
                    file_name=file_name,
                    mime="application/zip",
                )
                if st.button("Write a local copy", key="write_local_bundle"):
                    local_path = LSTM_PROJECTS_DIR / file_name
                    with open(local_path, "wb") as f:
                        f.write(bundle_bytes)
                    st.success(f"Local copy saved to: {local_path}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    with c2:
        st.subheader("Load existing project")
        uploaded_zip = st.file_uploader("Upload saved project zip", type=["zip"], key="bundle_loader")
        if uploaded_zip is not None:
            try:
                lstm_load_bundle(uploaded_zip)
                st.success("Project loaded.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    existing = sorted(LSTM_PROJECTS_DIR.glob("*.zip"))
    if existing:
        st.subheader("Local saved bundles")
        st.dataframe(pd.DataFrame({"file": [p.name for p in existing], "path": [str(p) for p in existing]}), use_container_width=True, hide_index=True)


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
            return render_ann_save_load_ui()
        if model_name == "CNN":
            return render_cnn_save_load_ui()
        if model_name == "LSTM":
            return render_lstm_save_load_ui()
        st.error(f"Unknown model: {model_name}")
