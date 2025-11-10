import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from data_utils import calculate_roc_pr_auc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define paths
current_dir = Path(__file__).resolve().parent.parent

DATA_PATH_AB = current_dir / 'datasets' / 'processed_ablation'

DATA_PATH_N = current_dir / 'datasets' / 'processed'

SUMMARY_PATH = current_dir / 'datasets' / 'summaries'

# Define name of the file to generate
#SUMMARY_FILE = SUMMARY_PATH / 'summary_results_online_detectors_pv_ds.xlsx'
#SUMMARY_FILE = SUMMARY_PATH / 'summary_results_ablation_study_pv_ds.xlsx'
SUMMARY_FILE = SUMMARY_PATH / 'summary_results_test.xlsx'

# Dataset folders to process (Comparative of SOTA Methods)
dataset_paths = [
    DATA_PATH_N / 'processed_server22_A1',
    DATA_PATH_N / 'processed_server22_A2',
    DATA_PATH_N / 'processed_server22_A3',
    DATA_PATH_N / 'processed_server21_A4',
    DATA_PATH_N / 'processed_server21_A5',
    DATA_PATH_N / 'processed_L40S02_A6',
    DATA_PATH_N / 'processed_server18_A7',
    DATA_PATH_N / 'processed_server18_A8',
    DATA_PATH_N / 'processed_server18_A9',
]

# Dataset folders to process (Comparative of Ablation Methods - SWKNN and OBKNN)
dataset_paths = [
    DATA_PATH_N / 'processed_server22_A1',
    DATA_PATH_N / 'processed_server22_A2',
    DATA_PATH_N / 'processed_server22_A3',
    DATA_PATH_N / 'processed_server21_A4',
    DATA_PATH_N / 'processed_server21_A5',
    DATA_PATH_N / 'processed_L40S02_A6',
    DATA_PATH_N / 'processed_server18_A7',
    DATA_PATH_N / 'processed_server18_A8',
    DATA_PATH_N / 'processed_server18_A9',
    current_dir / 'datasets' / 'processed_ablation_lite',
]

# Columns needed from Excel files
usecols = [
    "iteration", "method", "param", "ground_truth", "cleaned_score",
    "training_time", "scoring_time", "score"
]

# Containers for results
summary_data = []
processed_data = []

# Function to process a single group (for parallel processing)
def process_group(group, file_name, iteration, mwp):
    try:
        roc_auc, pr_auc, max_f1 = calculate_roc_pr_auc(group, "ground_truth", "cleaned_score")
    except Exception as e:
        logging.error(f"Metric calculation failed for {file_name} | Iter {iteration} | {mwp}: {e}")
        return None

    return {
        "iteration": iteration,
        "scenario": file_name.split("_")[0],
        "method_window_and_param": mwp,
        "AUC_ROC": roc_auc,
        "AUC_PR": pr_auc,
        "Max_F1": max_f1,
        "mean_training_time": group["training_time"].mean(),
        "max_training_time": group["training_time"].max(),
        "min_training_time": group["training_time"].min(),
        "mean_scoring_time": group["scoring_time"].mean(),
        "max_scoring_time": group["scoring_time"].max(),
        "min_scoring_time": group["scoring_time"].min(),
        "count_cleaned_score": group["cleaned_score"].count(),
        "count_raw_score": group["score"].count(),
        "count_anomalies": (group["ground_truth"] == 1).sum(),
        "count_normal": (group["ground_truth"] == 0).sum()
    }

# Function to process a single file
def process_file(file_path, path_name):
    try:
        df = pd.read_excel(file_path, usecols=usecols)
    except Exception as e:
        logging.error(f"Could not read {file_path.name}: {e}")
        return None

    window = file_path.name.split("_")[-1][:-5]
    df['method_window_and_param'] = df['method'] + "_" + window + "_" + df['param'].astype(str)

    grouped = df.groupby(["iteration", "method_window_and_param"])

    local_summary_data = []
    
    # Use ThreadPoolExecutor to parallelize group processing
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for (iteration, mwp), group in grouped:
            if len(group) != 4200:
                logging.warning(f"{file_path.name} | Iter {iteration} | {mwp}: Expected 4200 rows, got {len(group)}")
                continue
            futures.append(executor.submit(process_group, group, file_path.name, iteration, mwp))

        for future in as_completed(futures):
            result = future.result()
            if result:
                local_summary_data.append(result)

    return local_summary_data

# Function to process a dataset folder
def process_dataset_folder(path):
    if not path.exists():
        logging.warning(f"Folder not found: {path}")
        processed_data.append({"path": path.name, "processed": False})
        return []

    result_files = [f for f in path.iterdir() if f.suffix == '.xlsx' and f.name.startswith('A')]
    processed_data.append({"path": path.name, "processed": True})

    logging.info(f"Processing {len(result_files)} files in folder: {path.name}")

    all_local_summary_data = []
    for file in result_files:
        file_path = path / file
        all_local_summary_data.append(process_file(file_path, path.name))

    return all_local_summary_data

# Run dataset processing in parallel using ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    future_to_folder = {executor.submit(process_dataset_folder, path): path for path in dataset_paths}

    for future in as_completed(future_to_folder):
        folder_path = future_to_folder[future]
        try:
            folder_summary_data = future.result()
            if folder_summary_data:
                for data in folder_summary_data:
                    if data:
                        summary_data.extend(data)
        except Exception as e:
            logging.error(f"Error processing folder {folder_path}: {e}")

# Create output DataFrames
summary_df = pd.DataFrame(summary_data)
processed_df = pd.DataFrame(processed_data)

# Ensure output directory exists
SUMMARY_PATH.mkdir(parents=True, exist_ok=True)

# Save summary
summary_df.to_excel(SUMMARY_FILE, index=False)

# Log summary
logging.info(f"Summary saved to: {SUMMARY_FILE}")
logging.info(f"Processed folders:\n{processed_df}")
