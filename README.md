# Benchmark for Anomaly Detection on Spectral Data Streams

This repository contains the official code for the comparative benchmark of multivariate online anomaly detection methods, including the Online Bootstrapping K-Nearest Neighbor (OBKNN) algorithm.

You can read the full research and the supplementary materials here:
* **Published Article:** [Read the official publication](https://doi.org/10.1609/aaai.v40i18.38611)
* **Extended Version:** [Read the preprint on HAL](https://hal.science/hal-05115467)
* **Raw Data:** [Download datasets](https://drive.uca.fr/d/70aec2976f0e45438eb7/)
* **Appendix:** [Download supplementary information](https://drive.uca.fr/f/acd6b29b7e2346efbb82/)

## Installation

- Step 1: System-Wide Prerequisites
Before installing the Python packages, please ensure you have the following system-level tools installed:

    - Python 3.11.2

    - C++ Build Tools: Required to compile dependencies in dSalmon.

        - On Ubuntu: sudo apt-get install build-essential

        - On Windows: Install "C++ build tools".

    - Java (JDK): Required to run the capymoa package.

- Step 2: Evaluation Environment (env_spectra)


This environment is for running the different experiments.

```bash
# Create the environment
python3 -m venv env_spectra

# Activate the environment
source env_spectra/bin/activate
```

You can install all the necessary packages using pip:

```bash
pip install -r requirements.txt
```

- Step 3: Analysis Environment (env_analysis) This separate environment is only for analysing result-generation scripts.

```bash
# Create the environment
python3 -m venv env_analysis

# Activate the new environment
source env_analysis/bin/activate

# Install vus autorank capymoa and openpyxl
pip install vus==0.0.6 autorank==1.3.0 capymoa==0.9.0 openpyxl==3.1.5 

pip install -r requirements.txt
```


## Datasets Files

[Project folder for Raw Datasets](datasets/raw)

### Datasets description 
- The last column in each dataset file refers to the anomaly label (1: anomaly, 0:normal).
- The first colum in each dataset file correspond to the timestamp of the recorded spectral instances.
- The rest of columns in each dataset are associated with different wavelenths of the spectral instances.

## How to run OnlineBootKNN

### Parameters

- chunk_size: size of the chunks (default: 240)
- ensemble_size: size of the ensemble of chunks (default: 240)
- dmetric: distance metric used to compute differences among instances one of ["cityblock", "minkowski"] (default: "cityblock")
- transf: type of data tranformation, one of ["None", "ZNORM"] where "None" for raw data and "ZNORM" for z-normalization (default: "ZNORM")
- alpha: Level of Significance for One-Tailed Z-Tests (default: 0.05)

### Script
```
cd ~/spectral-benchmark
source env_spectra/bin/activate
python3 scripts/model/model_OnlineBootKNN.py
```

### Example of Detected Anomaly

[Link to Detected Anomaly Visualization (PDF)](notebooks/img_anomalies/A6_transf_ZNORM_anomaly_explanation_V2.pdf)


## How to Generate Comparative Anomaly Score of SOTA Methods

### Script

```
cd ~/spectral-benchmark
source env_spectra/bin/activate
python3 scripts/gen_comparative_AD_PV_online.py
```

## Summary of Results

[Link to Summary of Results (Excel)](datasets/summaries/summary_results_online_detectors_pv_ds.xlsx)
