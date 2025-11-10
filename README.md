# Benckmark for Anomaly Detection on Spectral Datastreams

This is the implementation of the comparative benchmark of multivariate online anomaly dection methods with Online Bootstrapping K-Nearest Neighbor (OBKNN) published in ... [[paper](...)]

## Installation

- Step 1: System-Wide Prerequisites
Before installing the Python packages, please ensure you have the following system-level tools installed:

    - Python 3.11.2

    - C++ Build Tools: Required to compile dependencies in dSalmon.

        - On Ubuntu: sudo apt-get install build-essential

        - On Windows: Install "C++ build tools".

    - Java (JDK): Required to run the capymoa package.

- Step 2: Evaluation Environment (env_spectra)
This environment is for running the runned experiments.

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

## Datasets Files

[Link to Raw Datasets](https://drive.uca.fr/d/86449bf3c17746098071/)

[Project folder for Raw Datasets](datasets/raw)




## Datasets description 
- The last column in each dataset file refers to the anomaly label (1: anomaly, 0:normal).
- The first colum in each dataset file correspond to the timestamp of the recorded spectral instances.
- The rest of columns in each dataset are associated with different wavelenths of the spectral instances.

## How to run OnlineBootKNN?

### Parameters

- chunk_size: size of the chunks (default: 240)
- ensemble_size: size of the ensemble of chunks (default: 240)
- dmetric: distance metric used to compute differences among instances one of ["cityblock", "minkowski"] (default: "cityblock")
- transf: type of data tranformation, one of ["None", "ZNORM"] where "None" for raw data and "ZNORM" for z-normalization (default: "ZNORM")
- alpha: Level of Significance for One-Tailed Z-Tests (default: 0.05)

### Script
```
cd ~/spectral-anomaly-benchmark-optical-emission
source env_spectra/bin/activate
python3 scripts/model/model_OnlineBootKNN.py
```
## Example of Real-Time Anomaly Detection

[Link to Real-Time Anomaly Explanation (PDF)](notebooks/img_anomalies/A6_transf_ZNORM_anomaly_explanation.pdf)

## Example of Detected Anomaly

[Link to Detected Anomaly Visualization (PDF)](notebooks/img_monitoring_score/A6_transf_ZNORM_z_scores_and_p.pdf)


## How to Generate Comparative Anomaly Score of SOTA Methods? 
### Script
```
cd ~/spectral-anomaly-benchmark-optical-emission
source env_spectra/bin/activate
python3 scripts/gen_comparative_AD_PV_online.py
```

## Summary of Results

[Link to Summary of Results (Excel)](datasets/summaries/summary_results_online_detectors_pv_ds.xlsx)