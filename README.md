# Online Bootstrapping K-Nearest Neighbor (OnlineBootKNN) - Anomaly Detection on Spectral Datastreams

This is the implementation of OnlineBootKNN published in ... [[paper](...)]

### Prerequisites

- Python 3.11.2
- Microsoft Visual C++ 14.0 or higher (PySAD Requirement)
- Java (CapyMOA Requirement)

## Required Packages

To ensure the OnlineBootKNN algorithm and all the project runs correctly, please install the following Python packages. It is highly recommended to use a virtual environment to manage these dependencies.

You can install all the necessary packages using pip:

```bash
pip install -r requirements.txt
```

## Datasets Files

[Link to Raw Datasets](datasets/raw)

## Datasets description 
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
python3 scripts/model/model_OnlineBootKNN.py
```

## Example of Real-Time Anomaly Detection

[Link to Real-Time Anomaly Explanation (PDF)](notebooks/img_anomalies/A1_transf_ZNORM_anomaly_explanation.pdf)

## Example of Detected Anomaly

[Link to Detected Anomaly Visualization (PDF)](notebooks/img_monitoring_score/A1_transf_ZNORM_z_scores_and_p.pdf)

## Summary of Results

[Link to Summary of Results (Excel)](datasets/summaries/summary_results_online_detectors_pv_ds.xlsx)