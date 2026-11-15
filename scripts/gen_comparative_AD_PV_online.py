"""Run comparative online anomaly detection benchmark across SOTA methods and OnlineBootKNN."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.stream import NumpyStream
from sklearn.model_selection import ParameterGrid

from online_bootknn.benchmark import (
    DEFAULT_PARAM_GRID,
    get_model_with_params,
    step_online_detector,
)

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

current_dir = Path(__file__).resolve().parent.parent
DATA_PATH = current_dir / "datasets" / "raw"
OUTPUT_DIR = current_dir / "datasets" / "processed"

N_RUNS = 5
LIST_WINDOW_SIZE = [60, 120, 240]
COLS_POS_SMIN = 1
COLS_POS_SMAX = 2049

PARAM_GRID = DEFAULT_PARAM_GRID


def run_comparative_benchmark() -> None:
    """Execute the comparative benchmark over all raw spectral datasets."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spectra_files = [
        f for f in DATA_PATH.iterdir() if f.suffix == ".csv" and not f.name.startswith("0_")
    ]

    for file_name in spectra_files:
        full_path_spectra = DATA_PATH / file_name
        result = pd.read_csv(
            full_path_spectra,
            sep=",",
            low_memory=False,
            dtype={"CURRENTTIMESTAMP": str},
        )
        cols = result.columns[COLS_POS_SMIN:COLS_POS_SMAX]

        stream = NumpyStream(
            result[cols].values,
            result["ANOMALY?"].values,
            dataset_name="PV",
            feature_names=cols,
        )

        for window_size in LIST_WINDOW_SIZE:
            for iteration in range(N_RUNS):
                records = []
                schema = stream.get_schema()

                for model_name, param_grid in PARAM_GRID.items():
                    for params in ParameterGrid(param_grid):
                        learner = get_model_with_params(
                            model_name, params, window_size=window_size, schema=schema
                        )
                        stream.restart()
                        row = 0
                        evaluator = AnomalyDetectionEvaluator(schema)

                        while stream.has_more_instances():
                            instance = stream.next_instance()
                            (
                                raw_score,
                                cleaned_score,
                                error_score,
                                training_time,
                                scoring_time,
                            ) = step_online_detector(learner, instance)

                            evaluator.update(instance.y_index, cleaned_score)
                            auc_val = evaluator.auc()

                            records.append(
                                {
                                    "iteration": iteration,
                                    "timestamp": result.iloc[row, 0],
                                    "method": model_name,
                                    "param": params,
                                    "score": raw_score,
                                    "cleaned_score": cleaned_score,
                                    "training_time": training_time,
                                    "scoring_time": scoring_time,
                                    "ground_truth": instance.y_index,
                                    "prog_auc": auc_val,
                                    "error_type_score": error_score,
                                }
                            )
                            row += 1

                df_to_save = pd.DataFrame(records)
                fl_name = file_name.name.split("_")[0]
                out_file = OUTPUT_DIR / f"{fl_name}_results_iter_{iteration}_pv_ds_ws_{window_size}.xlsx"
                df_to_save.to_excel(out_file, index=False)
                logger.info("Saved benchmark results to %s", out_file)


if __name__ == "__main__":
    run_comparative_benchmark()
