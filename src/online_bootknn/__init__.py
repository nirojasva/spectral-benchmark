"""OnlineBootKNN & Spectral Stream Anomaly Detection Benchmark Package."""

from online_bootknn.benchmark import (
    DEFAULT_PARAM_GRID,
    get_model_with_params,
    step_online_detector,
    summarize_group_metrics,
)
from online_bootknn.detector import OnlineBootKNN
from online_bootknn.metrics import calculate_performance_metrics, calculate_roc_pr_auc
from online_bootknn.preprocessing import (
    clean_score,
    featurewise_distance,
    split_summary_methods,
    transform_array,
    transform_instance,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_PARAM_GRID",
    "OnlineBootKNN",
    "calculate_performance_metrics",
    "calculate_roc_pr_auc",
    "clean_score",
    "featurewise_distance",
    "get_model_with_params",
    "split_summary_methods",
    "step_online_detector",
    "summarize_group_metrics",
    "transform_array",
    "transform_instance",
]
