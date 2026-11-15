"""Core implementation of the OnlineBootKNN streaming anomaly detector."""

from __future__ import annotations

import logging
import math
import os
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted

from online_bootknn.preprocessing import (
    featurewise_distance,
    transform_array,
    transform_instance,
)

try:
    from capymoa.base import AnomalyDetector
    from capymoa.instance import Instance
except ImportError:
    class AnomalyDetector:  # type: ignore
        def __init__(self, schema: Any = None, random_seed: Optional[int] = None) -> None:
            self.schema = schema
            self.random_seed = random_seed

    Instance = Any  # type: ignore

logger = logging.getLogger(__name__)


class OnlineBootKNN(AnomalyDetector):
    """Online Bootstrapping K-Nearest Neighbor (OnlineBootKNN) anomaly detector.

    Processes high-dimensional data streams in sliding windows, maintains a
    Poisson-bootstrapped ensemble of 1-NN models, and detects anomalies using
    adaptive one-tailed Z-score thresholds.
    """

    def __init__(
        self,
        schema: Any = None,
        random_seed: Optional[int] = None,
        chunk_size: int = 240,
        ensemble_size: int = 240,
        algorithm: str = "brute",
        n_jobs: int = -1,
        window_size: int = 120,
        dmetric: str = "cityblock",
        transf: Optional[str] = "ZNORM",
        alpha: float = 0.05,
        no_bootstrapp: bool = False,
        no_z_score: bool = False,
    ) -> None:
        if schema is not None:
            super().__init__(schema=schema, random_seed=random_seed)
        else:
            self.schema = None
            self.random_seed = random_seed
        self.random_generator = np.random.RandomState(random_seed)

        self.chunk_size = chunk_size
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.window_size = window_size
        self.dmetric = dmetric
        self.transf = transf
        self.alpha = alpha
        self.no_bootstrapp = no_bootstrapp
        self.no_z_score = no_z_score

        self.data_window: List[np.ndarray] = []
        self.ensemble: List[NearestNeighbors] = [
            NearestNeighbors(
                n_neighbors=1,
                algorithm=algorithm,
                n_jobs=self.n_jobs,
                metric=self.dmetric,
            )
            for _ in range(self.ensemble_size)
        ]

        self.chunks: List[List[np.ndarray]] = []
        self.init: bool = True
        self.last_value_is_anomaly: bool = False
        self.reset_threshold: int = 4200
        self.count_reset: Optional[int] = None
        self.normal_reference: Optional[np.ndarray] = None
        self.abnormal_reference: Optional[np.ndarray] = None
        self.z_critical_one_tail: float = float(norm.ppf(1 - self.alpha))

        # Streaming statistics (Welford's online algorithm)
        self.z: float = 0.0
        self.mean: float = np.nan
        self.std_dev: float = np.nan
        self.min_dist: float = np.nan
        self.n: int = 0
        self.n_anomalies: int = 0
        self.sum_squares: float = np.nan
        self.p_random_number: float = np.nan

        # Monitoring history
        self.z_scores_to_monitor: List[float] = []
        self.means_to_monitor: List[float] = []
        self.std_devs_to_monitor: List[float] = []
        self.min_dists_to_monitor: List[float] = []
        self.z_thresholds_to_monitor: List[float] = []
        self.p_random_number_to_monitor: List[float] = []

    def __str__(self) -> str:
        return "OnlineBootKNN"

    def train(self, instance: Instance) -> None:
        """Train the detector with a CapyMOA `Instance`."""
        instance = transform_instance(instance, self.transf)
        self._learn_batch(instance.x)

    def train_array(self, x: np.ndarray) -> None:
        """Train the detector directly with a 1D NumPy array (no CapyMOA required)."""
        transformed = transform_array(x, self.transf)
        self._learn_batch(transformed)

    def score_instance(self, instance: Instance) -> float:
        """Score a CapyMOA `Instance` and return its anomaly Z-score."""
        instance = transform_instance(instance, self.transf)
        return self._score_vector(instance.x)

    def score_array(self, x: np.ndarray) -> float:
        """Score a 1D NumPy array directly and return its anomaly Z-score."""
        transformed = transform_array(x, self.transf)
        return self._score_vector(transformed)

    def _score_vector(self, vec: np.ndarray) -> float:
        """Internal scoring routine shared by `score_instance` and `score_array`."""
        data = np.asarray(vec, dtype=float).reshape(1, -1)
        distances: List[float] = []
        references: List[np.ndarray] = []

        for i, model in enumerate(self.ensemble):
            try:
                check_is_fitted(model)
                dist, idxv = model.kneighbors(data)
                distances.append(float(dist[0][0]))
                v = self.chunks[i][idxv[0][0]]
                references.append(v)
            except NotFittedError:
                logger.debug("Model %d is not fitted yet (warm-up phase).", i)
                distances.append(0.0)
            except Exception as exc:
                logger.warning("Error scoring instance in model %d: %s", i, exc)
                distances.append(0.0)

        min_dist = float(np.min(distances))
        self.min_dist = min_dist

        if self.no_z_score:
            return self.min_dist

        if not self.init:
            if (self.n == 0) or (self.n > self.reset_threshold):
                self.count_reset = 0 if self.count_reset is None else self.count_reset + 1
                self.start_statistics(min_dist)
            else:
                self.update_z_score(min_dist)
                self.update_statistics(min_dist)

        if self.last_value_is_anomaly and len(references) > 0:
            self.n_anomalies += 1
            min_pos_dist = int(np.argmin(distances))
            self.normal_reference = references[min_pos_dist]
            self.abnormal_reference = data.reshape(-1)

        return float(self.z)

    def start_statistics(self, new_dist: float) -> None:
        """Initialize running statistics at warm-up completion or reset threshold."""
        self.z = 0.0
        self.n = 1
        self.mean = new_dist
        self.sum_squares = 0.0
        self.std_dev = 0.0
        self.last_value_is_anomaly = False

    def update_statistics(self, new_dist: float) -> None:
        """Incrementally update running mean and standard deviation."""
        self.n += 1
        delta = new_dist - self.mean
        self.mean += delta / self.n
        self.sum_squares += delta**2

        if self.n > 1:
            self.std_dev = math.sqrt(self.sum_squares / (self.n - 1))
        else:
            self.std_dev = np.nan

    def update_z_score(self, new_dist: float) -> None:
        """Compute one-tailed Z-score and flag anomaly if above critical value."""
        if self.n == 1:
            self.z = 0.0
            self.std_dev = 0.0
        elif self.std_dev != 0 and not pd.isna(self.std_dev):
            self.z = (new_dist - self.mean) / self.std_dev
        else:
            self.z = np.nan

        if not pd.isna(self.z) and self.z > self.z_critical_one_tail:
            self.last_value_is_anomaly = True
        else:
            self.last_value_is_anomaly = False

    def _learn_batch(self, data: np.ndarray) -> None:
        """Update ensemble models using online Poisson bootstrapping."""
        if not self.init:
            for i in range(self.ensemble_size):
                if self.no_bootstrapp:
                    self.p_random_number = 1
                elif self.last_value_is_anomaly:
                    self.p_random_number = 0
                else:
                    self.p_random_number = int(self.random_generator.poisson(1))

                for _ in range(int(self.p_random_number)):
                    self.chunks[i] = self.chunks[i][1:]
                    self.chunks[i].append(data)

                self.ensemble[i].fit(self.chunks[i])
        else:
            self.data_window.append(data)

            if len(self.data_window) >= self.window_size and self.init:
                window_arr = np.array(self.data_window)

                for i in range(self.ensemble_size):
                    logger.debug("Data window size: %d", len(window_arr))
                    if not self.no_bootstrapp:
                        indices = self.random_generator.choice(
                            len(window_arr), size=self.chunk_size, replace=True
                        )
                    else:
                        indices = np.arange(len(window_arr))

                    chunk = window_arr[indices]
                    self.chunks.append(list(chunk))
                    self.ensemble[i].fit(chunk)

                self.data_window = []
                self.init = False

    def predict(self, data: np.ndarray) -> int:
        """Return binary anomaly prediction (1 = anomaly, 0 = normal) for a 1D spectrum."""
        self.score_array(data)
        return int(self.last_value_is_anomaly)

    def explain(
        self,
        headers: np.ndarray,
        region_study_list: List[str],
        path: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> None:
        """Visualize wavelength-level intensity differences for detected anomalies."""
        plt.rcParams.update(
            {
                "font.size": 8,
                "axes.labelsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                "legend.fontsize": 7,
                "figure.titlesize": 8,
                "lines.linewidth": 1.0,
                "font.family": "serif",
            }
        )

        headers_float = np.asarray(headers, dtype=float)
        regions = np.asarray(region_study_list, dtype=str)

        if not self.last_value_is_anomaly:
            return

        if self.normal_reference is None or self.abnormal_reference is None:
            raise ValueError("Error: References are None.")

        differences = featurewise_distance(
            self.abnormal_reference, self.normal_reference, metric=self.dmetric
        )

        _, ax1 = plt.subplots(figsize=(3.5, 2))
        ax1.bar(
            headers_float,
            differences,
            label=f"Diff (Z: {round(self.z, 2)})",
            color="#666666",
            edgecolor="black",
            hatch="///",
            width=(headers_float[1] - headers_float[0]),
            linewidth=0.5,
        )

        ax1.set_xlabel("Wavelengths (nm)")
        ax1.set_ylabel("Intensity Diff.")
        _, y_max = ax1.get_ylim()

        for i, rs in enumerate(regions):
            try:
                parts = rs.split(":")
                rs_s = float(parts[0])
                rs_f = float(parts[1])
                comp = str(parts[2])
            except IndexError:
                continue

            ax1.axvline(x=rs_s, color="black", linestyle="--", linewidth=0.8)
            ax1.axvline(x=rs_f, color="black", linestyle="--", linewidth=0.8)

            level = i % 3
            text_pos_y = y_max * (0.95 - (level * 0.15))
            ax1.text(
                rs_s,
                text_pos_y,
                f" {comp}",
                fontsize=7,
                verticalalignment="top",
                fontweight="bold",
                color="black",
            )

        ax1.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="black",
            framealpha=1.0,
        )
        plt.title(f"Anomaly Explanation (N: {self.n_anomalies})")
        plt.tight_layout()

        if path and file_name:
            plt.savefig(f"{path}/{file_name}.pdf", format="pdf", bbox_inches="tight")

        plt.show()

    def monitor_core_statistics(self) -> dict[str, Any]:
        """Return and log current streaming statistics."""
        stats = {
            "mean": float(self.mean) if not pd.isna(self.mean) else None,
            "std_dev": float(self.std_dev) if not pd.isna(self.std_dev) else None,
            "min_dist": float(self.min_dist) if not pd.isna(self.min_dist) else None,
            "p_random_number": (
                float(self.p_random_number) if not pd.isna(self.p_random_number) else None
            ),
            "n_processed": self.n,
            "last_value_is_anomaly": self.last_value_is_anomaly,
            "n_anomalies": self.n_anomalies,
            "z_critical_one_tail": float(self.z_critical_one_tail),
            "z_score": float(self.z) if not pd.isna(self.z) else None,
        }
        logger.info("OnlineBootKNN Core Statistics: %s", stats)
        return stats

    def plot_core_statistics(self, path: str, file_name: str) -> None:
        """Append current statistics to history and save diagnostic PDF plots."""
        self.means_to_monitor.append(self.mean)
        self.std_devs_to_monitor.append(self.std_dev)
        self.min_dists_to_monitor.append(self.min_dist)
        self.z_scores_to_monitor.append(self.z)
        self.z_thresholds_to_monitor.append(self.z_critical_one_tail)
        self.p_random_number_to_monitor.append(self.p_random_number)

        plt.clf()
        _, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(
            self.means_to_monitor,
            label="Accumulated Mean Dist",
            color="blue",
            linestyle="--",
            marker="o",
            markersize=2,
        )
        ax1.plot(
            self.min_dists_to_monitor,
            label="Min Dist",
            color="green",
            marker="s",
            markersize=2,
        )
        ax1.set_xlabel("Time", fontsize=12)
        ax1.set_ylabel("Accumulated Mean & Min Dist", fontsize=12)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=10)

        ax2 = ax1.twinx()
        ax2.plot(
            self.std_devs_to_monitor,
            label="Accumulated Std Dev",
            color="orange",
            linestyle="--",
            marker="^",
            markersize=2,
        )
        ax2.set_ylabel("Accumulated Std Dev", fontsize=12)
        ax2.legend(loc="lower left", fontsize=10)

        plt.title(
            "Real-time Core Stats: Accumulated Mean Dist, Min Dist & Accumulated Std Dev",
            fontsize=14,
            pad=20,
        )
        plt.tight_layout()
        plot_path1 = os.path.join(path, f"{file_name}_core_stats.pdf")
        plt.savefig(plot_path1, format="pdf", bbox_inches="tight")
        plt.close()

        plt.clf()
        _, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(
            self.z_scores_to_monitor,
            label="Z Score",
            color="red",
            marker="o",
            markersize=4,
        )
        ax3.plot(
            self.z_thresholds_to_monitor,
            label="Z Threshold",
            color="purple",
            linestyle=":",
            marker="s",
            markersize=2,
        )
        ax3.set_xlabel("Time", fontsize=12)
        ax3.set_ylabel("Z Scores", fontsize=12)
        ax3.grid(True, linestyle="--", alpha=0.6)
        ax3.legend(loc="upper right", fontsize=10)

        ax4 = ax3.twinx()
        x_values = range(len(self.p_random_number_to_monitor))
        ax4.bar(
            x_values,
            self.p_random_number_to_monitor,
            label="P Random Number",
            color="black",
            alpha=0.8,
        )
        ax4.set_ylabel("P Random Number", fontsize=12)
        ax4.legend(loc="lower left", fontsize=10)

        plt.title(
            "Real-time Anomaly Detection: Z Scores, Thresholds & P Random Number",
            fontsize=14,
            pad=20,
        )
        plt.tight_layout()
        plot_path2 = os.path.join(path, f"{file_name}_z_scores_and_p.pdf")
        plt.savefig(plot_path2, format="pdf", bbox_inches="tight")
        plt.close()
