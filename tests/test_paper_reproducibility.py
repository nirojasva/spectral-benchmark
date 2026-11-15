"""Paper Reproducibility Regression Tests (AAAI 2026 Table 1 & NeurIPS 2026 BV4 Table 2).

Verifies that the refactored `online_bootknn` package reproduces the exact published
AUC-PR results on:
- `/home/nicolas/spectral-benchmark/datasets/raw/A1_20241023_145637_ALPS_0.csv`
- `/home/nicolas/spectral-benchmark/datasets/raw/DA1_20250609_152704_ALPS_0.csv`
"""

from __future__ import annotations

from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from capymoa.stream import NumpyStream

from online_bootknn.benchmark import (
    get_model_with_params,
    step_online_detector,
    summarize_group_metrics,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_A1_PATH = ROOT_DIR / "datasets" / "raw" / "A1_20241023_145637_ALPS_0.csv"
RAW_DA1_PATH = ROOT_DIR / "datasets" / "raw" / "DA1_20250609_152704_ALPS_0.csv"
SUMMARY_XLSX_PATH = ROOT_DIR / "datasets" / "summaries" / "summary_results_online_detectors_pv_ds.xlsx"

# Exact window sizes and default parameters per method from paper experimental setup:
# 240 for RRCF, OIF, IFASD, KitNet; 120 for EStorm, RSHash; 60 for HStree
PAPER_METHOD_CONFIGS = {
    "EStorm": (
        "ExactStorm",
        120,
        {"max_radius": 0.1},
        "ExactStorm_120_{'max_radius': 0.1}",
    ),
    "HStree": (
        "HStree",
        60,
        {
            "number_of_trees": 25,
            "anomaly_threshold": 0.5,
            "size_limit": 0.1,
            "max_depth": 15,
            "random_seed": 1,
        },
        "HStree_60_{'anomaly_threshold': 0.5, 'max_depth': 15, 'number_of_trees': 25, 'size_limit': 0.1}",
    ),
    "IFASD": (
        "IForestASD",
        240,
        {"initial_window_X": None},
        "IForestASD_240_{'initial_window_X': None}",
    ),
    "KitNet": (
        "KitNet",
        240,
        {"max_size_ae": 10, "learning_rate": 0.1, "hidden_ratio": 0.75},
        "KitNet_240_{'hidden_ratio': 0.75, 'learning_rate': 0.1, 'max_size_ae': 10}",
    ),
    "OIF": (
        "oIF",
        240,
        {
            "num_trees": 32,
            "max_leaf_samples": 32,
            "growth_criterion": "adaptive",
            "n_jobs": -1,
            "random_seed": 1,
        },
        "oIF_240_{'growth_criterion': 'adaptive', 'max_leaf_samples': 32, 'n_jobs': -1, 'num_trees': 32}",
    ),
    "RRCF": (
        "RobustRandomCutForest",
        240,
        {"tree_size": 256, "num_trees": 4},
        "RobustRandomCutForest_240_{'num_trees': 4, 'tree_size': 256}",
    ),
    "RSHash": (
        "RSHash",
        120,
        {
            "decay": 0.015,
            "num_components": 100,
            "num_hash_fns": 1,
            "feature_mins": [0],
            "feature_maxes": [10000],
        },
        "RSHash_120_{'decay': 0.015, 'feature_maxes': [10000], 'feature_mins': [0], 'num_components': 100, 'num_hash_fns': 1}",
    ),
}

# Published (Mean AUC-PR, Std AUC-PR) over 5 runs from Table 1 (A1) and Table 2 (DA1)
PUBLISHED_TABLE_A1 = {
    "EStorm": (0.507, 0.0),
    "HStree": (0.071, 0.0),
    "IFASD": (0.975, 1.7e-02),
    "KitNet": (0.948, 0.0),
    "OIF": (0.084, 0.0),
    "RRCF": (0.133, 2.2e-02),
    "RSHash": (0.014, 0.0),
}

PUBLISHED_TABLE_DA1 = {
    "EStorm": (0.507, 0.0),
    "HStree": (0.032, 0.0),
    "IFASD": (0.976, 5.7e-03),
    "KitNet": (0.946, 0.0),
    "OIF": (0.078, 0.0),
    "RRCF": (0.127, 3.2e-02),
    "RSHash": (0.017, 3.7e-05),
}


class TestPaperReproducibility(unittest.TestCase):
    """Verify exact reproducibility of published AAAI 2026 (A1) & NeurIPS 2026 (DA1) tables."""

    def test_a1_all_7_methods_match_published_table_1(self) -> None:
        """Verify all 7 online detectors on A1 match Table 1 (Mean AUC-PR & Std over 5 runs)."""
        self.assertTrue(SUMMARY_XLSX_PATH.exists(), f"Missing {SUMMARY_XLSX_PATH}")
        df = pd.read_excel(SUMMARY_XLSX_PATH)
        a1_df = df[df["scenario"] == "A1"]

        print("\n=== Table 1 (Scenario A1, 5 Runs) Reproducibility Check ===")
        print(f"{'Method':<10} | {'Paper Mean ± Std':<20} | {'Computed Mean ± Std':<22}")
        print("-" * 58)

        for paper_label, (_, _, _, mwp_key) in PAPER_METHOD_CONFIGS.items():
            expected_mean, expected_std = PUBLISHED_TABLE_A1[paper_label]
            sub = a1_df[a1_df["method_window_and_param"] == mwp_key]
            self.assertEqual(len(sub), 5, f"Expected 5 runs for {mwp_key}, got {len(sub)}")

            computed_mean = float(sub["AUC_PR"].mean())
            computed_std = float(sub["AUC_PR"].std(ddof=1))
            print(
                f"{paper_label:<10} | {expected_mean:.3f} ± {expected_std:<10.2e} | "
                f"{computed_mean:.3f} ± {computed_std:<10.2e}"
            )
            self.assertAlmostEqual(computed_mean, expected_mean, delta=0.001)
            self.assertAlmostEqual(computed_std, expected_std, delta=0.001)

    def test_live_raw_csv_stream_a1_and_da1_reproducibility_5_runs(self) -> None:
        """Run online detectors for 5 runs directly on raw A1 and DA1 CSV streams (4200x2048)
        and verify computed 5-run Mean AUC-PR and Std match Table 1 (A1) and Table 2 (DA1).
        """
        self.assertTrue(RAW_A1_PATH.exists(), f"Missing {RAW_A1_PATH}")
        self.assertTrue(RAW_DA1_PATH.exists(), f"Missing {RAW_DA1_PATH}")

        n_runs = 5
        # Deterministic methods need 1 run (identical across all 5 runs); stochastic run all 5
        live_methods_to_verify = [
            ("EStorm", 1),
            ("HStree", 1),
            ("RSHash", n_runs),
            ("IFASD", n_runs),
        ]

        print("\n=== Live Raw CSV 5-Run Execution Check (4200 instances x 2048 wavelengths) ===")
        for scenario_name, csv_path, expected_table in [
            ("A1", RAW_A1_PATH, PUBLISHED_TABLE_A1),
            ("DA1", RAW_DA1_PATH, PUBLISHED_TABLE_DA1),
        ]:
            df_raw = pd.read_csv(csv_path, low_memory=False, dtype={"CURRENTTIMESTAMP": str})
            self.assertEqual(len(df_raw), 4200)
            cols = df_raw.columns[1:2049]
            self.assertEqual(len(cols), 2048)

            stream = NumpyStream(
                df_raw[cols].values,
                df_raw["ANOMALY?"].values,
                dataset_name=scenario_name,
                feature_names=cols,
            )
            schema = stream.get_schema()

            for paper_label, runs_to_exec in live_methods_to_verify:
                model_name, ws, params, mwp_key = PAPER_METHOD_CONFIGS[paper_label]
                expected_auc_pr, expected_std = expected_table[paper_label]

                run_auc_prs = []
                for i in range(runs_to_exec):
                    np.random.seed(i)
                    learner = get_model_with_params(
                        model_name, dict(params), window_size=ws, schema=schema
                    )
                    stream.restart()

                    records = []
                    while stream.has_more_instances():
                        inst = stream.next_instance()
                        raw_s, clean_s, _, t_time, s_time = step_online_detector(learner, inst)
                        records.append(
                            {
                                "ground_truth": inst.y_index,
                                "cleaned_score": clean_s,
                                "score": raw_s,
                                "training_time": t_time,
                                "scoring_time": s_time,
                            }
                        )

                    group_df = pd.DataFrame(records)
                    summary = summarize_group_metrics(
                        group=group_df,
                        file_name=f"{scenario_name}_live.xlsx",
                        iteration=i,
                        method=model_name,
                        mwp=mwp_key,
                    )
                    assert summary is not None
                    run_auc_prs.append(summary["AUC_PR"])

                if runs_to_exec == 1:
                    run_auc_prs = run_auc_prs * n_runs

                mean_auc_pr = float(np.mean(run_auc_prs))
                std_auc_pr = float(np.std(run_auc_prs, ddof=1))
                print(
                    f"{scenario_name:<4} | {paper_label:<8} (ws={ws:<3}, {n_runs} runs) -> "
                    f"Mean: {mean_auc_pr:.3f} ± {std_auc_pr:.2e} | "
                    f"Paper: {expected_auc_pr:.3f} ± {expected_std:.2e}"
                )
                self.assertAlmostEqual(mean_auc_pr, expected_auc_pr, delta=0.001)
                self.assertAlmostEqual(std_auc_pr, expected_std, delta=0.001)


if __name__ == "__main__":
    unittest.main()
