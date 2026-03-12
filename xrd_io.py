from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import pandas as pd


ALIASES_2THETA = {
    "2theta",
    "two_theta",
    "twotheta",
    "theta2",
    "2θ",
    "角度",
    "衍射角",
    "x",
}

ALIASES_INTENSITY = {
    "intensity",
    "counts",
    "count",
    "强度",
    "计数",
    "y",
}


def _normalize_col_name(name: object) -> str:
    s = str(name).strip().lower()
    s = s.replace(" ", "").replace("_", "")
    return s


def _select_by_header(df: pd.DataFrame) -> Tuple[pd.Series | None, pd.Series | None]:
    norm = {_normalize_col_name(c): c for c in df.columns}

    two_theta_col = None
    intensity_col = None

    for k, c in norm.items():
        if any(alias in k for alias in ALIASES_2THETA):
            two_theta_col = c
            break

    for k, c in norm.items():
        if any(alias in k for alias in ALIASES_INTENSITY):
            intensity_col = c
            break

    if two_theta_col is not None and intensity_col is not None and two_theta_col != intensity_col:
        return df[two_theta_col], df[intensity_col]
    return None, None


def _select_by_numeric_heuristic(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    candidates = [c for c in numeric_df.columns if numeric_df[c].notna().sum() >= max(10, int(0.3 * len(numeric_df)))]
    if len(candidates) < 2:
        raise ValueError("未找到足够的数值列，请检查文件格式。")

    # 优先选“更单调且在合理角度区间”的列作为 2theta
    best_col = None
    best_score = -np.inf
    for c in candidates:
        col = numeric_df[c].dropna()
        if len(col) < 10:
            continue
        diffs = np.diff(col.to_numpy())
        monotonic_ratio = max(np.mean(diffs >= 0), np.mean(diffs <= 0))
        in_range_ratio = np.mean((col >= 0) & (col <= 180))
        score = 0.7 * monotonic_ratio + 0.3 * in_range_ratio
        if score > best_score:
            best_score = score
            best_col = c

    if best_col is None:
        best_col = candidates[0]

    # intensity 选与 2theta 不同且有效数据最多的列
    remain = [c for c in candidates if c != best_col]
    intensity_col = max(remain, key=lambda c: numeric_df[c].notna().sum())

    return numeric_df[best_col], numeric_df[intensity_col]


def read_xrd_table_from_upload(file_name: str, raw_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    name = file_name.lower()

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw_bytes))
    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw_bytes), comment="#")
    else:
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", comment="#")
        except Exception:
            df = pd.read_csv(io.BytesIO(raw_bytes), header=None, comment="#")

    if df.empty:
        raise ValueError("文件为空或无法读取。")

    s2t, sint = _select_by_header(df)
    if s2t is None or sint is None:
        s2t, sint = _select_by_numeric_heuristic(df)

    out = pd.DataFrame({"two_theta": pd.to_numeric(s2t, errors="coerce"), "intensity": pd.to_numeric(sint, errors="coerce")}).dropna()
    if len(out) < 20:
        raise ValueError("有效数据点过少（<20），请确认列选择和数据内容。")

    out = out.sort_values("two_theta")
    return out["two_theta"].to_numpy(), out["intensity"].to_numpy()
