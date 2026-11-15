"""Aggregate and summarize comparative online anomaly detection results across scenarios."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from online_bootknn.benchmark import summarize_group_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

current_dir = Path(__file__).resolve().parent.parent
DATA_PATH_AB = current_dir / "datasets" / "processed_ablation"
DATA_PATH_N = current_dir / "datasets" / "processed"
SUMMARY_PATH = current_dir / "datasets" / "summaries"
SUMMARY_FILE = SUMMARY_PATH / "summary_results_online_detectors_pv_ds.xlsx"

DATASET_PATHS = [
    DATA_PATH_N / "processed_server22_A1",
    DATA_PATH_N / "processed_server22_A2",
    DATA_PATH_N / "processed_server22_A3",
    DATA_PATH_N / "processed_server21_A4",
    DATA_PATH_N / "processed_server21_A5",
    DATA_PATH_N / "processed_L40S02_A6",
    DATA_PATH_N / "processed_server18_A7",
    DATA_PATH_N / "processed_server18_A8",
    DATA_PATH_N / "processed_server18_A9",
    current_dir / "datasets" / "processed",
]

USECOLS = [
    "iteration",
    "method",
    "param",
    "ground_truth",
    "cleaned_score",
    "training_time",
    "scoring_time",
    "score",
]


def process_file(file_path: Path, expected_rows: int = 4200) -> List[Dict[str, Any]]:
    """Read a single benchmark Excel file and compute summary metrics per group."""
    try:
        df = pd.read_excel(file_path, usecols=USECOLS)
    except Exception as exc:
        logger.error("Could not read %s: %s", file_path.name, exc)
        return []

    window = file_path.name.split("_")[-1][:-5]
    df["method_window_and_param"] = (
        df["method"] + "_" + window + "_" + df["param"].astype(str)
    )

    grouped = df.groupby(["iteration", "method", "method_window_and_param"])
    local_summary_data: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for (iteration, method, mwp), group in grouped:
            if len(group) != expected_rows:
                logger.warning(
                    "%s | Iter %s | Method %s | %s: Expected %d rows, got %d",
                    file_path.name,
                    iteration,
                    method,
                    mwp,
                    expected_rows,
                    len(group),
                )
                continue
            futures.append(
                executor.submit(
                    summarize_group_metrics,
                    group,
                    file_path.name,
                    int(iteration),
                    str(method),
                    str(mwp),
                )
            )

        for future in as_completed(futures):
            result = future.result()
            if result:
                local_summary_data.append(result)

    return local_summary_data


def process_dataset_folder(
    path: Path, processed_log: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Process all benchmark result Excel files in a dataset directory."""
    if not path.exists():
        logger.warning("Folder not found: %s", path)
        processed_log.append({"path": path.name, "processed": False})
        return []

    result_files = [
        f for f in path.iterdir() if f.suffix == ".xlsx" and f.name.startswith("A")
    ]
    processed_log.append({"path": path.name, "processed": True})
    logger.info("Processing %d files in folder: %s", len(result_files), path.name)

    folder_summaries: List[Dict[str, Any]] = []
    for file in result_files:
        folder_summaries.extend(process_file(path / file))

    return folder_summaries


def generate_summaries() -> None:
    """Aggregate all processed dataset folders into a single summary Excel report."""
    summary_data: List[Dict[str, Any]] = []
    processed_data: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_folder = {
            executor.submit(process_dataset_folder, path, processed_data): path
            for path in DATASET_PATHS
        }

        for future in as_completed(future_to_folder):
            folder_path = future_to_folder[future]
            try:
                folder_summary_data = future.result()
                summary_data.extend(folder_summary_data)
            except Exception as exc:
                logger.error("Error processing folder %s: %s", folder_path, exc)

    summary_df = pd.DataFrame(summary_data)
    processed_df = pd.DataFrame(processed_data)

    SUMMARY_PATH.mkdir(parents=True, exist_ok=True)
    summary_df.to_excel(SUMMARY_FILE, index=False)

    logger.info("Summary saved to: %s", SUMMARY_FILE)
    logger.info("Processed folders:\n%s", processed_df)


if __name__ == "__main__":
    generate_summaries()
