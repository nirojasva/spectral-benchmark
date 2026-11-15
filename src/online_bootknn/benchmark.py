"""Comparative benchmark model factory and summary aggregation utilities."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

import pandas as pd

from online_bootknn.detector import OnlineBootKNN
from online_bootknn.metrics import calculate_roc_pr_auc
from online_bootknn.preprocessing import clean_score

logger = logging.getLogger(__name__)

# Default hyperparameter grid for the comparative SOTA benchmark
DEFAULT_PARAM_GRID: Dict[str, Dict[str, list[Any]]] = {
    "xStream": {
        "num_components": [100],
        "n_chains": [100],
        "depth": [25],
    },
    "RSHash": {
        "decay": [0.015],
        "num_components": [100],
        "num_hash_fns": [1],
        "feature_mins": [[0]],
        "feature_maxes": [[10000]],
    },
    "IForestASD": {
        "initial_window_X": [None],
    },
    "KitNet": {
        "max_size_ae": [10],
        "learning_rate": [0.1],
        "hidden_ratio": [0.75],
    },
    "ExactStorm": {
        "max_radius": [0.1],
    },
    "oIF": {
        "num_trees": [32],
        "max_leaf_samples": [32],
        "growth_criterion": ["adaptive"],
        "n_jobs": [-1],
    },
    "HStree": {
        "number_of_trees": [25],
        "anomaly_threshold": [0.5],
        "size_limit": [0.1],
        "max_depth": [15],
    },
    "RobustRandomCutForest": {
        "tree_size": [256],
        "num_trees": [4],
    },
    "OnlineBootKNN": {
        "chunk_size": [240],
        "ensemble_size": [240],
        "alpha": [0.05],
        "dmetric": ["cityblock"],
        "algorithm": ["brute"],
        "n_jobs": [-1],
        "transf": ["ZNORM"],
    },
}


def get_model_with_params(
    model_name: str,
    params: Dict[str, Any],
    window_size: int,
    schema: Any = None,
) -> Any:
    """Instantiate any benchmarked streaming anomaly detector with unified window_size."""
    if model_name == "OnlineBootKNN":
        return OnlineBootKNN(schema=schema, window_size=window_size, **params)

    # Lazy-import external benchmark libraries so unit tests don't require Java/C++ unless needed
    if model_name in ("oIF", "HStree"):
        from capymoa.anomaly import HalfSpaceTrees as HStreeCapy
        from capymoa.anomaly import OnlineIsolationForest

        if model_name == "oIF":
            return OnlineIsolationForest(schema=schema, window_size=window_size, **params)
        return HStreeCapy(schema=schema, window_size=window_size, **params)

    if model_name in (
        "xStream",
        "RSHash",
        "IForestASD",
        "RobustRandomCutForest",
        "KitNet",
        "ExactStorm",
    ):
        from pysad.models import (
            ExactStorm,
            IForestASD,
            KitNet,
            RobustRandomCutForest,
            RSHash,
            xStream,
        )

        if model_name == "xStream":
            return xStream(window_size=window_size, **params)
        if model_name == "RSHash":
            return RSHash(sampling_points=window_size, **params)
        if model_name == "IForestASD":
            return IForestASD(window_size=window_size, **params)
        if model_name == "RobustRandomCutForest":
            return RobustRandomCutForest(shingle_size=window_size, **params)
        if model_name == "KitNet":
            return KitNet(
                grace_feature_mapping=window_size,
                grace_anomaly_detector=window_size,
                **params,
            )
        if model_name == "ExactStorm":
            return ExactStorm(window_size=window_size, **params)

    raise ValueError(f"Unknown model: {model_name}")


def step_online_detector(learner: Any, instance: Any) -> Tuple[Any, float, str, float, float]:
    """Run one test-then-train step on either a PySAD or CapyMOA detector.

    Returns:
        (raw_score, cleaned_score, error_message, training_time_sec, scoring_time_sec)
    """
    if hasattr(learner, "fit_partial"):  # PySAD interface
        t0 = time.perf_counter()
        learner.fit_partial(instance.x)
        training_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        raw_score = learner.score_partial(instance.x)
        scoring_time = time.perf_counter() - t0

    elif hasattr(learner, "train"):  # CapyMOA / OnlineBootKNN interface
        t0 = time.perf_counter()
        raw_score = learner.score_instance(instance)
        scoring_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        learner.train(instance)
        training_time = time.perf_counter() - t0
    else:
        raise TypeError(f"Detector {type(learner).__name__} lacks 'fit_partial' or 'train'")

    cleaned, error_msg = clean_score(raw_score)
    return raw_score, cleaned, error_msg, training_time, scoring_time


def summarize_group_metrics(
    group: pd.DataFrame,
    file_name: str,
    iteration: int,
    method: str,
    mwp: str,
    score_direction: str = "direct",
) -> Dict[str, Any] | None:
    """Compute summary performance and latency statistics for a single benchmark run."""
    try:
        roc_auc, pr_auc, max_f1 = calculate_roc_pr_auc(
            group, "ground_truth", "cleaned_score", score_direction=score_direction
        )
    except Exception as exc:
        logger.error(
            "Metric calculation failed for %s | Iter %s | %s: %s",
            file_name,
            iteration,
            mwp,
            exc,
        )
        return None

    return {
        "iteration": iteration,
        "scenario": file_name.split("_")[0],
        "method_window_and_param": mwp,
        "AUC_ROC": roc_auc,
        "AUC_PR": pr_auc,
        "Max_F1": max_f1,
        "mean_training_time": float(group["training_time"].mean()),
        "max_training_time": float(group["training_time"].max()),
        "min_training_time": float(group["training_time"].min()),
        "mean_scoring_time": float(group["scoring_time"].mean()),
        "max_scoring_time": float(group["scoring_time"].max()),
        "min_scoring_time": float(group["scoring_time"].min()),
        "count_cleaned_score": int(group["cleaned_score"].count()),
        "count_raw_score": int(group["score"].count()),
        "count_anomalies": int((group["ground_truth"] == 1).sum()),
        "count_normal": int((group["ground_truth"] == 0).sum()),
    }
