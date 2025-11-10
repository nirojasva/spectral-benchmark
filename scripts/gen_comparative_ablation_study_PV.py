import pandas as pd
import numpy as np
import os, sys

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

sys.stdout = open(os.devnull, "w")


from capymoa.stream import NumpyStream
from capymoa.evaluation import AnomalyDetectionEvaluator
import time 
from model.model_OnlineBootKNN import OnlineBootKNN
from dSalmon.outlier import SWKNN 
from data_utils import clean_score, transform_instance
from sklearn.model_selection import ParameterGrid
from pathlib import Path


# Get the path to the current script
current_dir = Path(__file__).resolve().parent

# Go one level up
current_dir = current_dir.parent

DATA_PATH = current_dir / 'datasets' / 'raw' 

# List of files
spectra_files = [f for f in DATA_PATH.iterdir() if f.suffix == '.csv' and not f.name.startswith('0_')]

N_RUNS = 3
#N_RUNS = 5


LIST_WINDOW_SIZE = [60, 120, 240]
#LIST_WINDOW_SIZE = [60]

COLS_POS_SMIN = 1
COLS_POS_SMAX = 2049

#PARAM GRID FOR ABLATION STUDY OBKNN ans SWKNN
PARAM_GRID = {

    "SWKNN": {
        'k': [1, 10, 50],                        # Default: 'unknown'
        'k_is_max': [False],                     # Default: False
        'metric': ['cityblock'],                 # Default: 'unknown'
        #'metric_params': [{'p':2}],             # Default: 'unknown'
        #'float_type': [np.float64],             # Default: 'unknown'
        'min_node_size': [5],                    # Default: 5
        'max_node_size': [20],                   # Default: 20
        'split_sampling': [5],                   # Default: 5
    },
    "SWKNN_own": {
        'ensemble_size': [1],                    # Default: 10
        #'alpha': [0.05],                        # Default: 0.05
        'dmetric':['cityblock'],                 # Default: cityblock
        'algorithm':['brute'],                   # Default: 'brute'
        'transf': ["NONE"],                      # Default: "ZNORM"
        'no_bootstrapp': [True],                 # Default: False
        'no_z_score': [True],                    # Default: False
        'n_jobs': [-1],                          # Default: -1 (use all processors)
    },
    "BKNN": {
        'chunk_size': [1, 10, 50, 240],          # Default: 10
        'ensemble_size': [1, 10, 50, 240],       # Default: 10
        #'alpha': [0.05],                        # Default: 0.05
        'dmetric':['cityblock'],                 # Default: cityblock
        'algorithm':['brute'],                   # Default: 'brute'
        'transf': ["NONE"],                      # Default: "ZNORM"
        'no_bootstrapp': [False],                # Default: False
        'no_z_score': [True],                    # Default: False
        'n_jobs': [-1],                          # Default: -1 (use all processors)
    },
    "OnlineBootKNN": { #the only missing config 
        'chunk_size': [1, 10, 50, 240],         # Default: 10
        'ensemble_size': [1, 10, 50, 240],      # Default: 10
        'alpha': [0.05, 0.01],            # Default: 0.05
        'dmetric':['cityblock'],    # Default: cityblock
        'algorithm':['brute'],      # Default: brute
        'n_jobs': [-1],             # Default: -1 (use all processors)
        "transf": ["SQRT"],        # Default: "NONE"
    },
}


def get_model_with_params(model_name, param_grid, window_size, schema):
    if model_name == "SWKNN":
        if 'transf' in param_grid:
            current_model_params = param_grid.copy()
            del current_model_params['transf']
        else:
            current_model_params = param_grid.copy()
        return SWKNN(window=window_size, **current_model_params)
    
    elif model_name == "SWKNN_own":
        return OnlineBootKNN(schema=schema, chunk_size=window_size, **param_grid)
    
    elif model_name == "BKNN":
        return OnlineBootKNN(schema=schema, window_size=window_size, **param_grid)

    elif model_name == "OnlineBootKNN":
        return OnlineBootKNN(schema=schema, window_size=window_size, **param_grid)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")

for file_name in spectra_files:
    """ 
    if any(substring in file_name.name for substring in ["A1_","A2_","A3_","A7_","A8_","A9_"]):

        print("File to Use: ",file_name)
        
    else:
        print("Filed not to Use", file_name)
        continue
    """

    # Load spectra and labels data
    full_path_spectra = os.path.join(DATA_PATH, file_name)
    result = pd.read_csv(full_path_spectra, sep=',', low_memory=False, dtype={'CURRENTTIMESTAMP': str})
    cols = result.columns[COLS_POS_SMIN:COLS_POS_SMAX]
    
    """
    # Check for duplicates in the merged DataFrame
    duplicates = result[result.duplicated(subset=['CURRENTTIMESTAMP'])]['CURRENTTIMESTAMP']

    # Count the number of duplicate rows
    duplicate_count = duplicates.shape[0]

    # Print the duplicate rows (optional, for inspection)
    print(f"Number of duplicate rows: {duplicate_count}")
    print(duplicates.head())  # Show the first few duplicate rows


    print("Name DS: ", file_name)

    if len(result)==4200:
        print("Correct Length...")
    else:
        print("Incorrect Length of...", len(result))
        break
    """

    stream = NumpyStream(result[cols].values, result["ANOMALY?"].values, dataset_name="PV", feature_names=cols)

    for window_size in LIST_WINDOW_SIZE:    
        
        for i in range(N_RUNS):
            df_to_save = {}
            list_iter = []
            list_time = []
            list_model = []
            list_param = []
            scores = []
            cleaned_scores = []
            t_time = []
            s_time = []
            list_auc = []
            list_gt = []
            list_error_score = []

            # Initialize all models beforehand to avoid reinitialization inside loops
            schema = stream.get_schema()            
            
            # Now perform grid search using the PARAM_GRID
            for model_name, param_grid in PARAM_GRID.items():
                
                # Create a parameter grid (cartesian product of parameters)
                grid = ParameterGrid(param_grid)

                for params in grid:
                    # Create the model using the parameters for this combination
                    print("Model:",model_name)
                    print("Params:",params)
                    
                    learner = get_model_with_params(model_name, params, window_size=window_size, schema=schema)
                    stream.restart()
                    row = 0
                    evaluator = AnomalyDetectionEvaluator(schema) 

                    while stream.has_more_instances():

                        instance = stream.next_instance()
                        
                        print(f'A new instance ({row})...index:', instance.y_label,', label:', instance.y_index, ", time:", result.iloc[row, 0])
                        print('The new instance:', instance.x)
                        
                        if hasattr(learner, "fit_predict"): # Models Implemented in dSalmon
                            
                            # Calculate the elapsed time for training
                            training_time = 0
                            
                            start_time = time.time()  # Record the start time
                            if 'transf' in params:
                                t_instance = transform_instance(instance, params["transf"])
                                score = learner.fit_predict(t_instance.x)
                            else:
                                score = learner.fit_predict(instance.x)

                            end_time = time.time()  # Record the end time
                            # Calculate the elapsed time for scoring
                            scoring_time = end_time - start_time

                        
                        elif hasattr(learner, "train"): # Models Implemented in Capymoa (OBKNN)

                            start_time = time.time()  # Record the start time
                            score = learner.score_instance(instance)
                            end_time = time.time()  # Record the end time
                            # Calculate the elapsed time for scoring
                            scoring_time = end_time - start_time
                            
                            start_time = time.time()  # Record the start time
                            learner.train(instance)
                            end_time = time.time()  # Record the end time
                            # Calculate the elapsed time for training
                            training_time = end_time - start_time

                        cleaned_score, error_score = clean_score(score)
                        
                        print('model name:', model_name)
                        print('param:', params)
                        print(f'{model_name}-Score ({row}):', score)  
                        print(f'{model_name}-Cleaned Score ({row}):', cleaned_score) # Clean Score for passing it to Evaluator                      

                        evaluator.update(instance.y_index, cleaned_score)
                        auc = evaluator.auc()
                        print(f'{model_name}-AUC ({row}):', auc)

                        list_iter.append(i)
                        list_time.append(result.iloc[row, 0])
                        list_model.append(model_name)
                        list_param.append(params)
                        scores.append(score)
                        cleaned_scores.append(cleaned_score)
                        t_time.append(training_time)
                        s_time.append(scoring_time)
                        list_gt.append(instance.y_index)
                        list_auc.append(auc)
                        list_error_score.append(error_score)

                        row = row + 1 
                        
            df_to_save["iteration"] = list_iter
            df_to_save["timestamp"] = list_time
            df_to_save["method"] = list_model
            df_to_save["param"] = list_param
            df_to_save["score"] = scores
            df_to_save["cleaned_score"] = cleaned_scores
            df_to_save["training_time"] = t_time
            
            df_to_save["scoring_time"] = s_time
            
            df_to_save["ground_truth"] = list_gt
            df_to_save["prog_auc"] = list_auc
            df_to_save["error_type_score"] = list_error_score
            
            df_to_save = pd.DataFrame(df_to_save)
            
            print(df_to_save.head(3))

            fl_name = file_name.name.split("_")[0]
            
            df_to_save.to_excel(f"datasets/processed_ablation/{fl_name}_results_ablation_iter_{i}_pv_ds_ws_{window_size}.xlsx", index=False)


