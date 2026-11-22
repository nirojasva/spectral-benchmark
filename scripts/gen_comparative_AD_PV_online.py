import pandas as pd
import numpy as np
import os, sys

os.environ["OMP_NUM_THREADS"] = "20"
os.environ["OPENBLAS_NUM_THREADS"] = "20"
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"

sys.stdout = open(os.devnull, "w")


from capymoa.anomaly import OnlineIsolationForest, HalfSpaceTrees as HStreeCapy
from capymoa.stream import NumpyStream
from capymoa.evaluation import AnomalyDetectionEvaluator
from pysad.models import ExactStorm, IForestASD, KitNet, LODA, RobustRandomCutForest, RSHash, xStream
import time 
from model.model_OnlineBootKNN import OnlineBootKNN
from data_utils import clean_score
from sklearn.model_selection import ParameterGrid
from pathlib import Path


# Get the path to the current script
current_dir = Path(__file__).resolve().parent

# Go one level up
current_dir = current_dir.parent

DATA_PATH = current_dir / 'datasets' / 'raw'

# List of files
spectra_files = [f for f in DATA_PATH.iterdir() if f.suffix == '.csv' and not f.name.startswith('0_')]

#N_RUNS = 1
N_RUNS = 5


LIST_WINDOW_SIZE = [60, 120, 240]
#LIST_WINDOW_SIZE = [5]

COLS_POS_SMIN = 1
COLS_POS_SMAX = 2049

#PARAM GRID FOR DEFAULT HYPERPARAMETER OF EACH METHOD
PARAM_GRID = {
    "xStream": {
        'num_components': [100],     # Default: 100
        'n_chains': [100],           # Default: 100
        'depth': [25],               # Default: 25
    },
    "RSHash": {
        'decay': [0.015],               # Default: 0.015
        'num_components': [100],        # Default: 100
        'num_hash_fns': [1],            # Default: 1
        'feature_mins': [[0]],          # Default: [0]
        'feature_maxes': [[10000]],     # Default: [10000]
    },
    "IForestASD": {
        'initial_window_X': [None],   # Default: None
    },
    "KitNet": {
        'max_size_ae': [10],        # Default: 10
        'learning_rate': [0.1],     # Default: 0.1
        'hidden_ratio': [0.75],     # Default: 0.75
    },
    "ExactStorm": {
        'max_radius': [0.1],        # Default: 0.1
    },
    "oIF": {
        'num_trees': [32],                 # Default: 32
        'max_leaf_samples': [32],          # Default: 32
        'growth_criterion': ['adaptive' ], # Default: 'adaptive'
        'n_jobs': [-1],                    # Default: -1 (use all processors)
    },
    "HStree": {
        'number_of_trees': [25],    # Default: 25
        'anomaly_threshold': [0.5], # Default: 0.5
        'size_limit': [0.1],        # Default: 0.1
        'max_depth': [15],          # Default: 15
    },
    "RobustRandomCutForest": {
        #'shingle_size': [4 , 20],  # Default: 4
        'tree_size': [256],         # Default: 256 
        'num_trees': [4],           # Default: 4
    },
    "OnlineBootKNN": {
        'chunk_size': [240],        # Default: 10
        'ensemble_size': [ 240],    # Default: 10
        'alpha': [0.05],            # Default: 0.05
        'dmetric':['cityblock'],    # Default: cityblock
        'algorithm':['brute'],      # Default: brute
        'n_jobs': [-1],             # Default: -1 (use all processors)
        "transf": ["ZNORM"],        # Default: "NONE"
    },
}

def get_model_with_params(model_name, param_grid, window_size, schema):
    if model_name == "xStream":
        return xStream(window_size=window_size, **param_grid)
    elif model_name == "RSHash":
        return RSHash(sampling_points=window_size, **param_grid)
    elif model_name == "IForestASD":
        return IForestASD(window_size=window_size, **param_grid)
    elif model_name == "RobustRandomCutForest":
        return RobustRandomCutForest(shingle_size=window_size, **param_grid)
    elif model_name == "KitNet":
        return KitNet(grace_feature_mapping=window_size, grace_anomaly_detector=window_size , **param_grid)
    elif model_name == "ExactStorm":
        return ExactStorm(window_size=window_size, **param_grid)
    elif model_name == "oIF":
        return OnlineIsolationForest(schema=schema, window_size=window_size, **param_grid)
    elif model_name == "HStree":
        return HStreeCapy(schema=schema, window_size=window_size, **param_grid)
    elif model_name == "OnlineBootKNN":
        return OnlineBootKNN(schema=schema, window_size=window_size, **param_grid)
    else:
        raise ValueError(f"Unknown model: {model_name}")

for file_name in spectra_files:
    """
    if any(substring in file_name.name for substring in ["A1_"]):

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
                    learner = get_model_with_params(model_name, params, window_size=window_size, schema=schema)
                    stream.restart()
                    row = 0
                    evaluator = AnomalyDetectionEvaluator(schema) 

                    while stream.has_more_instances():

                        instance = stream.next_instance()
                        
                        print(f'A new instance ({row})...index:', instance.y_label,', label:', instance.y_index, ", time:", result.iloc[row, 0])
                        print('The new instance:', instance.x)
                        
                        if hasattr(learner, "fit_partial"): # Models Implemented in Pysad
                            start_time = time.time()  # Record the start time                        
                            learner.fit_partial(instance.x)
                            end_time = time.time()  # Record the end time
                            # Calculate the elapsed time for training
                            training_time = end_time - start_time
                            
                            start_time = time.time()  # Record the start time
                            score = learner.score_partial(instance.x)
                            end_time = time.time()  # Record the end time
                            # Calculate the elapsed time for scoring
                            scoring_time = end_time - start_time
                        
                        elif hasattr(learner, "train"): # Models Implemented in Capymoa

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
            
            df_to_save.to_excel(f"datasets/processed/{fl_name}_results_iter_{i}_pv_ds_ws_{window_size}.xlsx", index=False)


