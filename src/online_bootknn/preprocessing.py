"""Preprocessing, spectral transformations, and score cleaning utilities."""

from __future__ import annotations

import ast
import math
import re
from typing import Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import zscore

try:
    from capymoa.instance import Instance
except ImportError:
    Instance = Any  # type: ignore

INF: float = 1.0e200


def transform_array(x: np.ndarray, transf: str | None) -> np.ndarray:
    """Apply a spectral transformation directly to a 1D NumPy array.

    Separating the pure NumPy logic from CapyMOA's `Instance` makes this
    function reusable in REST APIs, unit tests, and standalone pipelines.
    """
    arr = np.asarray(x, dtype=float).reshape(-1)

    if transf == "MA":
        return pd.Series(arr).rolling(window=5, min_periods=None).mean().to_numpy()
    if transf == "DIL":
        return arr[::5]
    if transf in ("FOD", "SOD"):
        return np.diff(np.diff(arr))
    if transf == "FT":
        return np.fft.fft(arr).real
    if transf == "iFT":
        return np.fft.ifft(arr).real
    if transf == "SQRT&ZNORM":
        sqrt_arr = np.power(arr, 0.5)
        return zscore(sqrt_arr, nan_policy="propagate", axis=0)
    if transf == "SQRT":
        return np.power(arr, 0.5)
    if transf == "ZNORM":
        return zscore(arr, nan_policy="propagate", axis=0)

    return arr


def transform_instance(instance: Instance, transf: str | None) -> Instance:
    """Apply a spectral transformation to a CapyMOA `Instance`."""
    if not transf or transf == "None":
        return instance

    t_array = transform_array(instance.x, transf)
    return Instance.from_array(instance.schema, t_array.reshape(-1))


def featurewise_distance(
    vec1: np.ndarray,
    vec2: np.ndarray,
    metric: str = "cityblock",
    p: int = 2,
) -> np.ndarray:
    """Compute element-wise distance between two spectra for anomaly explanation."""
    v1 = np.asarray(vec1, dtype=float)
    v2 = np.asarray(vec2, dtype=float)

    if metric == "cityblock":
        return np.abs(v1 - v2)
    if metric == "minkowski":
        return np.abs(v1 - v2) ** p
    if metric == "euclidean":
        return np.abs(v1 - v2) ** 2

    raise ValueError(f"Featurewise distance for '{metric}' is not supported.")


def clean_score(score: Any) -> Tuple[float, str]:
    """Sanitize raw anomaly detector output into a valid finite float."""
    error_score: list[str] = []

    if isinstance(score, str) and len(score) > 0:
        try:
            score = ast.literal_eval(score)
        except (ValueError, SyntaxError):
            score = 0.0
            error_score.append("String Output Assigned to 0.")

    if isinstance(score, list):
        if len(score) > 0:
            score = score[0]
            error_score.append("List Output Assigned the First Value.")
            if np.isinf(score):
                score = INF
                error_score.append(f"Infinity Output Assigned to {INF}.")
        else:
            score = 0.0
            error_score.append("Empty List Assigned to 0.")

    if isinstance(score, np.ndarray):
        if score.size > 0:
            score = score.flat[0]
            error_score.append("Array Output Assigned the First Value.")
            if np.isinf(score):
                score = INF
                error_score.append(f"Infinity Output Assigned to {INF}.")
        else:
            score = 0.0
            error_score.append("Empty Array Assigned to 0.")

    if score is None:
        score = 0.0
        error_score.append("None Output Assigned to 0.")

    if isinstance(score, str):
        score = 0.0
        error_score.append("String Output Assigned to 0.")

    if pd.isna(score):
        score = 0.0
        error_score.append("NaN Output Assigned to 0 (Pandas).")

    if isinstance(score, (int, float, np.number)):
        if np.isinf(score):
            score = INF
            error_score.append(f"Infinity Output Assigned to {INF}.")
        elif math.isnan(float(score)):
            score = 0.0
            error_score.append("NaN Output Assigned to 0 (Math).")

    cleaned = float(score)
    message = " | ".join(error_score)
    return cleaned, message


def split_summary_methods(
    method_window_and_param: str,
) -> Tuple[str, str | None, str | None]:
    """Parse a benchmark identifier string into (method, window, params)."""
    match = re.match(r"([^_]+)_([^_]+)_\{(.+)\}", method_window_and_param)
    if match:
        method = match.group(1)
        window = match.group(2)
        params = "{" + match.group(3) + "}"
        return method, window, params
    return method_window_and_param, None, None
