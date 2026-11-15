"""Evaluation metrics for streaming anomaly detection."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

try:
    from vus.metrics import get_metrics
    from vus.utils.utility import get_list_anomaly

    HAS_VUS = True
except ImportError:
    HAS_VUS = False


def calculate_roc_pr_auc(
    df: pd.DataFrame,
    gt_column: str,
    score_column: str,
    score_direction: str = "direct",
) -> Tuple[float, float, float]:
    """Calculate ROC AUC, PR AUC, and max F1 Score (threshold-independent)."""
    scores = df[score_column].values
    labels = df[gt_column].values

    if score_direction == "inverse":
        scores = -scores
    elif score_direction != "direct":
        raise ValueError("score_direction must be 'direct' or 'inverse'")

    roc_auc = float(roc_auc_score(labels, scores))
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = float(auc(recall, precision))

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    max_f1 = float(f1_scores.max())

    return roc_auc, pr_auc, max_f1


def calculate_performance_metrics(
    df: pd.DataFrame,
    gt_column: str,
    score_column: str,
    t_window_size: int | None = None,
    score_direction: str = "direct",
) -> Tuple[Any, ...] | None:
    """Calculate standard and range-based (VUS) anomaly detection metrics."""
    labels = df[gt_column].values
    scores = df[score_column].values

    if len(np.unique(labels)) < 2:
        return None

    if score_direction == "inverse":
        scores_to_use = -scores
    elif score_direction == "direct":
        scores_to_use = scores
    else:
        raise ValueError("score_direction must be 'direct' or 'inverse'")

    roc_auc_wtd = float(roc_auc_score(labels, scores_to_use))
    precision_wtd, recall_wtd, _ = precision_recall_curve(labels, scores_to_use)
    pr_auc_wtd = float(auc(recall_wtd, precision_wtd))
    f1_scores_wtd = 2 * (precision_wtd * recall_wtd) / (precision_wtd + recall_wtd + 1e-8)
    max_f1_wtd = float(np.max(f1_scores_wtd))

    if t_window_size is not None:
        final_labels = labels[t_window_size:]
        final_scores = scores_to_use[t_window_size:]
    else:
        final_labels = labels
        final_scores = scores_to_use

    if len(final_labels) == 0 or len(np.unique(final_labels)) < 2:
        return None

    roc_auc = float(roc_auc_score(final_labels, final_scores))
    precision, recall, thresholds = precision_recall_curve(final_labels, final_scores)
    pr_auc = float(auc(recall, precision))

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = int(np.argmax(f1_scores))
    max_f1 = float(f1_scores[best_f1_idx])

    if best_f1_idx < len(thresholds):
        best_threshold = float(thresholds[best_f1_idx])
    else:
        best_threshold = float(np.max(final_scores))

    binary_preds = (final_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(final_labels, binary_preds).ravel()

    pct_detection = float(recall[best_f1_idx])
    pct_false_positives = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    metrics: Dict[str, Any] = {}
    if HAS_VUS:
        normalized_scores = MinMaxScaler().fit_transform(final_scores.reshape(-1, 1)).ravel()
        anomaly_lengths = get_list_anomaly(labels)
        sliding_window = max(1, int(np.median(anomaly_lengths)) - 1)
        metrics = get_metrics(normalized_scores, final_labels, metric="all", slidingWindow=sliding_window)

    return (
        roc_auc,
        pr_auc,
        max_f1,
        metrics,
        roc_auc_wtd,
        pr_auc_wtd,
        max_f1_wtd,
        pct_detection,
        pct_false_positives,
        int(tn),
        int(fp),
        int(fn),
        int(tp),
        best_threshold,
    )
