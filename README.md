# Benchmark for Anomaly Detection on Spectral Data Streams

This repository contains the official code for the comparative benchmark of multivariate online anomaly detection methods, including the Online Bootstrapping K-Nearest Neighbor (OBKNN) algorithm.

You can read the full research and supplementary materials here:

- Published Article: https://doi.org/10.1609/aaai.v40i18.38611  
- Extended Version: https://hal.science/hal-05115467  
- Raw Data: https://drive.uca.fr/d/70aec2976f0e45438eb7/  
- Appendix: https://drive.uca.fr/f/acd6b29b7e2346efbb82/  

---

## Installation

### Step 1: System-Wide Prerequisites

Ensure the following tools are installed:

- Python 3.11.2 or compatible
- C++ Build Tools (required for dSalmon)
  - Ubuntu:
    ```bash
    sudo apt-get install build-essential
    ```
  - Windows:
    Install Microsoft C++ Build Tools
- Java (JDK) required for capymoa

---

### Step 2: Evaluation Environment (env_spectra)

This environment is used to run experiments.

#### Linux / macOS

```bash
python3 -m venv env_spectra
source env_spectra/bin/activate
```

#### Windows (PowerShell)

```powershell
py -3.11 -m venv env_spectra
env_spectra\Scripts\Activate.ps1
```

#### Windows (CMD)

```cmd
py -3.11 -m venv env_spectra
env_spectra\Scripts\activate.bat
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Step 3: Analysis Environment (env_analysis)

This environment is used for result analysis.

#### Linux / macOS

```bash
python3 -m venv env_analysis
source env_analysis/bin/activate
```

#### Windows (PowerShell)

```powershell
py -3.11 -m venv env_analysis
env_analysis\Scripts\Activate.ps1
```

#### Windows (CMD)

```cmd
py -3.11 -m venv env_analysis
env_analysis\Scripts\activate.bat
```

Install required packages:

```bash
pip install vus==0.0.6 autorank==1.3.0 capymoa==0.9.0 openpyxl==3.1.5
pip install -r requirements.txt
```



---

## Dataset Files

[datasets/raw](datasets/raw)

### Dataset Description

- Last column: anomaly label (1 = anomaly, 0 = normal)
- First column: timestamp
- Remaining columns: spectral wavelengths

---

## How to Run OnlineBootKNN

### Parameters

- chunk_size: size of chunks (default 240)
- ensemble_size: number of chunks (default 240)
- dmetric: distance metric ("cityblock", "minkowski")
- transf: transformation ("None", "ZNORM")
- alpha: significance level (default 0.05)

---

### Script

```bash
python scripts/model/model_OnlineBootKNN.py
```

---

### Example of Detected Anomaly

[notebooks/img_anomalies/A6_transf_ZNORM_anomaly_explanation_V2.pdf] (notebooks/img_anomalies/A6_transf_ZNORM_anomaly_explanation_V2.pdf)

---

## Generate Comparative Anomaly Scores

```bash
python scripts/gen_comparative_AD_PV_online.py
```

---

## Summary of Results

[datasets/summaries/summary_results_online_detectors_pv_ds.xlsx](datasets/summaries/summary_results_online_detectors_pv_ds.xlsx)
