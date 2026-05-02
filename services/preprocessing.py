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

def ann_read_uploaded_table(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Only CSV, XLSX, and XLS files are supported.")


def ann_is_datetime_like(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            return False
        converted = pd.to_datetime(sample, errors="coerce")
        ratio = converted.notna().mean()
        return ratio >= 0.8
    return False


def ann_expand_datetime_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    expanded = []
    for col in list(out.columns):
        s = out[col]
        if ann_is_datetime_like(s):
            dt = pd.to_datetime(s, errors="coerce")
            if dt.notna().sum() == 0:
                continue
            out[f"{col}__year"] = dt.dt.year
            out[f"{col}__month"] = dt.dt.month
            out[f"{col}__day"] = dt.dt.day
            out[f"{col}__dayofweek"] = dt.dt.dayofweek
            out[f"{col}__hour"] = dt.dt.hour
            out[f"{col}__is_month_end"] = dt.dt.is_month_end.astype("float64")
            out[f"{col}__is_month_start"] = dt.dt.is_month_start.astype("float64")
            out.drop(columns=[col], inplace=True)
            expanded.append(col)
    return out, expanded


def ann_get_feature_schema(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    schema = {}
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        entry = {
            "dtype": dtype_str,
            "is_numeric": bool(pd.api.types.is_numeric_dtype(s)),
            "is_bool": bool(pd.api.types.is_bool_dtype(s)),
            "example": None if s.dropna().empty else str(s.dropna().iloc[0]),
            "categories": None,
        }
        if not entry["is_numeric"]:
            cats = s.dropna().astype(str).value_counts().head(30).index.tolist()
            entry["categories"] = cats
        schema[col] = entry
    return schema


def ann_ensure_feature_frame(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in required_columns:
        if col not in out.columns:
            out[col] = np.nan
    out = out[required_columns]
    return out


def ann_build_preprocessor(X: pd.DataFrame):
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor, num_cols, cat_cols


def ann_prepare_dataset(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    if df is None or df.empty:
        raise ValueError("No dataset loaded.")

    work_df = df.copy()

    if config["drop_duplicates"]:
        work_df = work_df.drop_duplicates().reset_index(drop=True)

    if config["target_column"] is None or config["target_column"] not in work_df.columns:
        raise ValueError("Choose a valid target column in Preprocess.")

    target_col = config["target_column"]

    # Drop rows with missing target
    work_df = work_df.loc[work_df[target_col].notna()].copy()
    if len(work_df) < 10:
        raise ValueError("Dataset is too small after removing rows with missing target.")

    drop_cols = [c for c in config["drop_columns"] if c in work_df.columns and c != target_col]
    if drop_cols:
        work_df = work_df.drop(columns=drop_cols)

    # Feature selection
    feature_cols = config["feature_columns"]
    if not feature_cols:
        feature_cols = [c for c in work_df.columns if c != target_col]
    feature_cols = [c for c in feature_cols if c in work_df.columns and c != target_col]
    if not feature_cols:
        raise ValueError("No valid feature columns selected.")

    X_raw = work_df[feature_cols].copy()
    y_raw = work_df[target_col].copy()

    datetime_expanded = []
    if config["auto_datetime"]:
        X_raw, datetime_expanded = ann_expand_datetime_columns(X_raw)

    task = ann_infer_task(y_raw) if config["task_mode"] == "Auto Detect" else config["task_mode"].lower()
    if task not in {"classification", "regression"}:
        task = "regression"

    preprocessor, num_cols, cat_cols = ann_build_preprocessor(X_raw)

    if task == "classification":
        y_raw = y_raw.astype(str)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        class_names = label_encoder.classes_.tolist()
        stratify = y if len(np.unique(y)) > 1 and min(pd.Series(y).value_counts()) >= 2 else None
    else:
        label_encoder = None
        class_names = None
        y = pd.to_numeric(y_raw, errors="coerce").astype(float).values
        valid = ~np.isnan(y)
        X_raw = X_raw.loc[valid].reset_index(drop=True)
        y = y[valid]
        stratify = None

    if len(X_raw) < 10:
        raise ValueError("Not enough valid rows to continue.")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=float(config["test_size"]),
        random_state=int(config["random_seed"]),
        shuffle=bool(config["shuffle_data"]),
        stratify=stratify,
    )

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except Exception:
        feature_names = [f"f_{i}" for i in range(X_train.shape[1])]

    class_weights = None
    if task == "classification" and config.get("use_class_weights", True):
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    return {
        "df_used": work_df,
        "task": task,
        "target_column": target_col,
        "feature_columns_original": feature_cols,
        "feature_columns_after_datetime": X_raw.columns.tolist(),
        "datetime_expanded_columns": datetime_expanded,
        "feature_schema": ann_get_feature_schema(work_df[feature_cols]),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "preprocessor": preprocessor,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "X_train": np.asarray(X_train).astype("float32"),
        "X_test": np.asarray(X_test).astype("float32"),
        "X_train_raw": X_train_raw.reset_index(drop=True),
        "X_test_raw": X_test_raw.reset_index(drop=True),
        "y_train": np.asarray(y_train),
        "y_test": np.asarray(y_test),
        "feature_names": feature_names,
        "class_weights": class_weights,
    }


def cnn_save_uploaded_zip(uploaded_file, project_name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = CNN_DATA_DIR / f"{project_name}_{ts}"
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / uploaded_file.name
    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    extract_dir = target_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def cnn_open_uploaded_image(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass
    try:
        img = Image.open(uploaded_file)
        img.load()
        return img, None
    except UnidentifiedImageError:
        name = getattr(uploaded_file, "name", "file")
        suffix = Path(name).suffix.lower()
        if suffix in {".heic", ".heif"}:
            return None, (
                f"{name}: HEIC/HEIF was selected, but Pillow usually cannot decode it by default. "
                "Convert it to JPG/PNG first, or install pillow-heif in your environment."
            )
        return None, f"{name}: This file is not a readable image for Pillow. Use JPG, PNG, BMP, TIFF, WEBP, or GIF."
    except Exception as e:
        return None, f"{getattr(uploaded_file, 'name', 'file')}: {e}"


def cnn_validate_image_file(filepath):
    try:
        with Image.open(filepath) as img:
            img = ImageOps.exif_transpose(img)
            img.load()
            img.convert("RGB")
        return True, None
    except UnidentifiedImageError:
        suffix = Path(filepath).suffix.lower()
        if suffix in {".heic", ".heif"}:
            return False, "HEIC/HEIF is not supported by default in this environment. Convert to JPG or PNG first."
        return False, "Unreadable image file."
    except Exception as e:
        return False, str(e)


def cnn_filter_valid_images(df: pd.DataFrame):
    if df is None or df.empty:
        return df, []
    keep_rows = []
    dropped = []
    for _, row in df.iterrows():
        ok, err = cnn_validate_image_file(row["filepath"])
        if ok:
            keep_rows.append(row.to_dict())
        else:
            dropped.append({
                "filepath": row["filepath"],
                "label": row["label"],
                "reason": err,
            })
    out_df = pd.DataFrame(keep_rows)
    if out_df.empty:
        return out_df, dropped
    out_df = out_df.reset_index(drop=True)
    return out_df, dropped


def cnn_list_image_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in CNN_IMAGE_EXTS]


def cnn_infer_dataset_structure(extract_dir: Path) -> pd.DataFrame:
    files = cnn_list_image_files(extract_dir)
    if not files:
        raise ValueError("No image files found in the uploaded ZIP.")

    rows = []
    for fp in files:
        parts = [x for x in fp.relative_to(extract_dir).parts]
        lower_parts = [x.lower() for x in parts]

        split = None
        label = None

        if len(parts) >= 2 and lower_parts[0] in {"train", "training", "val", "valid", "validation", "test"}:
            split_map = {
                "train": "train", "training": "train",
                "val": "val", "valid": "val", "validation": "val",
                "test": "test",
            }
            split = split_map[lower_parts[0]]
            label = parts[1]
        else:
            # assume class folder is the first folder name
            if len(parts) < 2:
                continue
            label = parts[0]

        rows.append({"filepath": str(fp), "label": label, "split": split})

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("ZIP structure is invalid. Expected class folders containing images.")

    df = df[df["label"].notna()].copy()
    counts = df["label"].value_counts()
    valid_labels = counts[counts >= 2].index.tolist()
    df = df[df["label"].isin(valid_labels)].reset_index(drop=True)

    if df.empty:
        raise ValueError("Each class must contain at least 2 images.")

    return df


def cnn_finalize_splits(df: pd.DataFrame, val_ratio: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    # Pandas on Streamlit Cloud can infer an all-missing split column as float64.
    # Assigning strings like "train"/"val" into that column then crashes with a TypeError.
    # Force object dtype first so split labels remain valid across pandas versions.
    if "split" not in df.columns:
        df["split"] = pd.Series(index=df.index, dtype="object")
    else:
        df["split"] = df["split"].astype("object")
    if df["split"].notna().any():
        # keep explicit train/val/test splits; if only train exists, create val from train
        present = set(df["split"].dropna().unique().tolist())
        if "val" not in present and "train" in present:
            train_df = df[df["split"] == "train"].copy()
            if train_df["label"].value_counts().min() >= 2 and len(train_df) >= len(train_df["label"].unique()) * 2:
                tr_idx, va_idx = train_test_split(
                    train_df.index,
                    test_size=val_ratio,
                    random_state=seed,
                    stratify=train_df["label"],
                )
                df.loc[tr_idx, "split"] = "train"
                df.loc[va_idx, "split"] = "val"
        if "train" not in present:
            non_test = df[df["split"] != "test"].copy()
            tr_idx, va_idx = train_test_split(
                non_test.index,
                test_size=val_ratio,
                random_state=seed,
                stratify=non_test["label"],
            )
            df.loc[tr_idx, "split"] = "train"
            df.loc[va_idx, "split"] = "val"
    else:
        tr_idx, va_idx = train_test_split(
            df.index,
            test_size=val_ratio,
            random_state=seed,
            stratify=df["label"],
        )
        df.loc[tr_idx, "split"] = "train"
        df.loc[va_idx, "split"] = "val"

    return df.reset_index(drop=True)


def cnn_dataset_summary(df: pd.DataFrame) -> dict:
    dims = []
    broken = 0
    for fp in df["filepath"].sample(min(len(df), 100), random_state=42).tolist():
        try:
            with Image.open(fp) as img:
                dims.append(img.size)
        except Exception:
            broken += 1
    width_stats = [d[0] for d in dims] if dims else [0]
    height_stats = [d[1] for d in dims] if dims else [0]

    return {
        "total_images": int(len(df)),
        "classes": int(df["label"].nunique()),
        "splits": df["split"].value_counts().to_dict(),
        "class_counts": df["label"].value_counts().to_dict(),
        "width_min": int(np.min(width_stats)),
        "width_max": int(np.max(width_stats)),
        "height_min": int(np.min(height_stats)),
        "height_max": int(np.max(height_stats)),
        "sampled_broken_files": int(broken),
    }


def _cnn_load_image_with_pil(path_bytes, image_size: int, color_mode: str):
    path = path_bytes.decode("utf-8")
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if color_mode == "Grayscale → 3-channel":
            img = ImageOps.grayscale(img).convert("RGB")
        else:
            img = img.convert("RGB")
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        return arr


def cnn_read_image_tf(path, label, image_size: int, color_mode: str):
    tf = cnn_get_tf()
    img = tf.numpy_function(
        func=lambda p: _cnn_load_image_with_pil(p, image_size, color_mode),
        inp=[path],
        Tout=tf.float32,
    )
    img.set_shape([image_size, image_size, 3])
    return img, label


def cnn_make_tf_dataset(paths, labels, cfg, training: bool):
    tf = cnn_get_tf()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(min(len(paths), cfg["shuffle_buffer"]), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(
        lambda x, y: cnn_read_image_tf(x, y, cfg["image_size"], cfg["color_mode"]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(cfg["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds


def cnn_prepare_single_image(img: Image.Image, cfg):
    tf = cnn_get_tf()
    img = cnn_ensure_rgb(img, "RGB" if cfg["color_mode"] == "RGB" else "GRAY")
    img = img.resize((cfg["image_size"], cfg["image_size"]))
    arr = np.array(img).astype("float32")
    if cfg["color_mode"] == "Grayscale → 3-channel":
        gray = np.mean(arr, axis=-1, keepdims=True)
        arr = np.concatenate([gray, gray, gray], axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr


def lstm_read_uploaded_data(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")


def lstm_detect_date_column(df: pd.DataFrame) -> Optional[str]:
    name_hits = [c for c in df.columns if any(x in str(c).lower() for x in ["date", "time", "timestamp"])]
    for col in name_hits + list(df.columns):
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.70:
                return col
        except Exception:
            continue
    return None


def lstm_get_numeric_columns(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = exclude or []
    numeric = []
    for col in df.columns:
        if col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().mean() >= 0.70:
            numeric.append(col)
    return numeric


def lstm_coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def lstm_frequency_from_dates(dates: pd.Series) -> str:
    if dates is None or len(dates) < 3:
        return "Unknown"
    diffs = dates.sort_values().diff().dropna().dt.total_seconds() / 86400.0
    if len(diffs) == 0:
        return "Unknown"
    med = float(np.median(diffs))
    if med <= 2:
        return "Daily"
    if med <= 10:
        return "Weekly"
    if med <= 40:
        return "Monthly"
    return "Irregular"


def lstm_annualization_factor(freq_name: str) -> int:
    if freq_name == "Daily":
        return 252
    if freq_name == "Weekly":
        return 52
    if freq_name == "Monthly":
        return 12
    return 252


def lstm_apply_missing(df: pd.DataFrame, method: str) -> pd.DataFrame:
    out = df.copy()
    if method == "drop":
        return out.dropna()
    if method == "ffill":
        return out.ffill()
    if method == "bfill":
        return out.bfill()
    if method == "ffill_bfill":
        return out.ffill().bfill()
    if method == "interpolate":
        return out.interpolate(method="linear").ffill().bfill()
    if method == "median_impute":
        imp = SimpleImputer(strategy="median")
        out[out.columns] = imp.fit_transform(out)
        return out
    return out


def lstm_apply_transform(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "raw":
        return df.copy()
    if mode == "pct_change":
        previous = df.shift(1).replace(0, np.nan)
        out = (df - previous) / previous
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.mask(out.abs() > 1e300, np.nan)
        return out
    if mode == "log_return":
        safe = df.replace(0, np.nan)
        return np.log(safe / safe.shift(1))
    raise ValueError("Unknown transform mode")


def lstm_clip_quantiles(df: pd.DataFrame, low_q: float, high_q: float) -> pd.DataFrame:
    low = df.quantile(low_q)
    high = df.quantile(high_q)
    return df.clip(low, high, axis=1)


def lstm_preprocess_dataset(df: pd.DataFrame, cfg: Dict) -> Dict:
    if df is None or df.empty:
        raise ValueError("No dataset loaded.")

    task = str(cfg.get("task_mode", "Regression")).lower()
    if task not in {"regression", "classification"}:
        task = "regression"

    date_col = cfg["date_col"]
    feature_cols = [c for c in cfg.get("feature_cols", []) if c in df.columns]
    target_cols = [c for c in cfg.get("target_cols", []) if c in df.columns]
    class_target_col = cfg.get("classification_target_col")
    lookback = int(cfg["lookback"])
    horizon = int(cfg["horizon"])

    if not feature_cols:
        raise ValueError("Select at least one feature column.")
    if task == "regression" and not target_cols:
        raise ValueError("Select at least one numeric target column for regression.")
    if task == "classification" and (not class_target_col or class_target_col not in df.columns):
        raise ValueError("Select one classification target column.")
    if lookback < 2:
        raise ValueError("Lookback must be at least 2.")
    if horizon < 1:
        raise ValueError("Horizon must be at least 1.")

    work = df.copy()
    if date_col:
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        dates = work[date_col].copy()
    else:
        dates = pd.Series(pd.date_range("2000-01-01", periods=len(work), freq="D"))

    if task == "regression":
        keep_cols = list(dict.fromkeys(feature_cols + target_cols))
        numeric_df = lstm_coerce_numeric(work[keep_cols], keep_cols)

        if cfg["resample_rule"] != "None" and date_col:
            tmp = pd.concat([dates, numeric_df], axis=1).set_index(date_col)
            numeric_df = tmp.resample(cfg["resample_rule"]).mean()
            dates = pd.Series(numeric_df.index)
            numeric_df = numeric_df.reset_index(drop=True)

        numeric_df = lstm_apply_missing(numeric_df, cfg["missing_method"])
        if cfg["clip_outliers"]:
            numeric_df = lstm_clip_quantiles(numeric_df, cfg["clip_low_q"], cfg["clip_high_q"])

        transformed = lstm_apply_transform(numeric_df, cfg["transform_mode"])
        transformed = lstm_apply_missing(transformed, cfg["missing_method"])
        valid_mask = transformed.notna().all(axis=1)
        transformed = transformed.loc[valid_mask].reset_index(drop=True)
        numeric_df = numeric_df.loc[valid_mask].reset_index(drop=True)
        dates = dates.loc[valid_mask].reset_index(drop=True)

        feature_frame = transformed[feature_cols].copy()
        target_frame = transformed[target_cols].copy()

        n = len(transformed)
        if n <= lookback + horizon + 10:
            raise ValueError("Dataset is too small after preprocessing for the chosen lookback and horizon.")

        train_end_idx = int(n * cfg["train_frac"])
        val_end_idx = int(n * (cfg["train_frac"] + cfg["val_frac"]))
        train_end_idx = max(train_end_idx, lookback + horizon)
        val_end_idx = max(val_end_idx, train_end_idx + 1)
        val_end_idx = min(val_end_idx, n - 1)

        feature_scaler = lstm_make_scaler(cfg["scale_method"])
        target_scaler = lstm_make_scaler(cfg["scale_method"])
        feature_scaler.fit(feature_frame.iloc[:train_end_idx])
        target_scaler.fit(target_frame.iloc[:train_end_idx])

        X_scaled = feature_scaler.transform(feature_frame)
        y_scaled = target_scaler.transform(target_frame)

        seq = lstm_create_sequences(
            X_scaled,
            y_scaled,
            dates.to_numpy(),
            lookback,
            horizon,
            train_end_idx=train_end_idx,
            val_end_idx=val_end_idx,
        )

        X_seq = seq["X_seq"]
        y_seq = seq["y_seq"]
        tags = seq["split_tags"]

        X_train = X_seq[tags == "train"]
        X_val = X_seq[tags == "val"]
        X_test = X_seq[tags == "test"]
        y_train = y_seq[tags == "train"]
        y_val = y_seq[tags == "val"]
        y_test = y_seq[tags == "test"]

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            raise ValueError("Split produced an empty train/validation/test segment. Adjust split fractions or reduce lookback.")

        return {
            "task": "regression",
            "dates": dates,
            "original_numeric": numeric_df,
            "transformed": transformed,
            "feature_cols": feature_cols,
            "target_cols": target_cols,
            "target_col": target_cols[0] if len(target_cols) == 1 else ", ".join(target_cols),
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "label_encoder": None,
            "class_names": None,
            "X_seq": X_seq,
            "y_seq": y_seq,
            "target_times": seq["target_times"],
            "split_tags": tags,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "y_train_flat": lstm_flatten_y(y_train),
            "y_val_flat": lstm_flatten_y(y_val),
            "y_test_flat": lstm_flatten_y(y_test),
            "train_end_idx": train_end_idx,
            "val_end_idx": val_end_idx,
            "freq_name": lstm_frequency_from_dates(pd.to_datetime(dates, errors="coerce")),
            "config_snapshot": dict(cfg),
        }

    # Classification path: numeric sequence features -> future class label.
    feature_df = lstm_coerce_numeric(work[feature_cols], feature_cols)
    label_series = work[class_target_col].copy()

    if cfg["resample_rule"] != "None" and date_col:
        raise ValueError("Resampling is disabled for LSTM classification because class labels cannot be averaged safely.")

    feature_df = lstm_apply_missing(feature_df, cfg["missing_method"])
    if cfg["clip_outliers"]:
        feature_df = lstm_clip_quantiles(feature_df, cfg["clip_low_q"], cfg["clip_high_q"])

    transformed_features = lstm_apply_transform(feature_df, cfg["transform_mode"])
    transformed_features = lstm_apply_missing(transformed_features, cfg["missing_method"])

    valid_mask = transformed_features.notna().all(axis=1) & label_series.notna()
    transformed_features = transformed_features.loc[valid_mask].reset_index(drop=True)
    feature_df = feature_df.loc[valid_mask].reset_index(drop=True)
    label_series = label_series.loc[valid_mask].reset_index(drop=True)
    dates = dates.loc[valid_mask].reset_index(drop=True)

    n = len(transformed_features)
    if n <= lookback + horizon + 10:
        raise ValueError("Dataset is too small after preprocessing for the chosen lookback and horizon.")

    train_end_idx = int(n * cfg["train_frac"])
    val_end_idx = int(n * (cfg["train_frac"] + cfg["val_frac"]))
    train_end_idx = max(train_end_idx, lookback + horizon)
    val_end_idx = max(val_end_idx, train_end_idx + 1)
    val_end_idx = min(val_end_idx, n - 1)

    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(label_series.astype(str).values)
    class_names = label_encoder.classes_.tolist()
    if len(class_names) < 2:
        raise ValueError("Classification target must contain at least two classes.")

    feature_scaler = lstm_make_scaler(cfg["scale_method"])
    feature_scaler.fit(transformed_features.iloc[:train_end_idx])
    X_scaled = feature_scaler.transform(transformed_features)

    seq = lstm_create_classification_sequences(
        X_scaled,
        y_labels,
        dates.to_numpy(),
        lookback,
        horizon,
        train_end_idx=train_end_idx,
        val_end_idx=val_end_idx,
    )

    X_seq = seq["X_seq"]
    y_seq = seq["y_seq"]
    tags = seq["split_tags"]

    X_train = X_seq[tags == "train"]
    X_val = X_seq[tags == "val"]
    X_test = X_seq[tags == "test"]
    y_train = y_seq[tags == "train"]
    y_val = y_seq[tags == "val"]
    y_test = y_seq[tags == "test"]

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Split produced an empty train/validation/test segment. Adjust split fractions or reduce lookback.")

    return {
        "task": "classification",
        "dates": dates,
        "original_numeric": feature_df,
        "transformed": transformed_features,
        "feature_cols": feature_cols,
        "target_cols": [class_target_col],
        "target_col": class_target_col,
        "feature_scaler": feature_scaler,
        "target_scaler": None,
        "label_encoder": label_encoder,
        "class_names": class_names,
        "X_seq": X_seq,
        "y_seq": y_seq,
        "target_times": seq["target_times"],
        "split_tags": tags,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "y_train_flat": y_train,
        "y_val_flat": y_val,
        "y_test_flat": y_test,
        "train_end_idx": train_end_idx,
        "val_end_idx": val_end_idx,
        "freq_name": lstm_frequency_from_dates(pd.to_datetime(dates, errors="coerce")),
        "config_snapshot": dict(cfg),
    }


# Public workflow API
def preprocess_ann(*args, **kwargs):
    return ann_prepare_dataset(*args, **kwargs)

def preprocess_cnn(*args, **kwargs):
    return cnn_finalize_splits(*args, **kwargs)

def preprocess_lstm(*args, **kwargs):
    return lstm_preprocess_dataset(*args, **kwargs)
