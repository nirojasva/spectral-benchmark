
import numpy as np
import pandas as pd

import sys, os
from pathlib import Path


# Get the path to the current script
current_dir = Path(__file__).resolve().parent

# Go one level up
current_dir = current_dir.parent

# Add the 'scripts' directory to sys.path to be able to import data_utils.py
sys.path.append(str(current_dir))

from pysad.models.integrations.reference_window_model import ReferenceWindowModel
from pyod.models.knn import KNN



class SWKNN(ReferenceWindowModel):
    """
    A KNN-based anomaly detection model using a sliding window approach.
    Based on the method described in: 
    "An Anomaly Detection Approach Based on Isolation Forest Algorithm for Streaming Data using Sliding Window" 
    :cite:`ding2013anomaly`.

    Note: Concept drift adaptation is not implemented, as it is part of the simulation.

    Args:
        initial_window_X (np.ndarray of shape (n_samples, n_features), optional): 
            Initial data window for fitting the model. Default is None.
        window_size (int): 
            Size of the reference window and the sliding interval. Default is 2048.
        **kwargs: 
            Additional keyword arguments passed to the underlying KNN estimator.
    """

    def __init__(self, p_window_size=2048, p_sliding_size=1, **kwargs):
        super().__init__(KNN, window_size=p_window_size, sliding_size=p_sliding_size, **kwargs)
        self.k = kwargs.get("n_neighbors", None)
        # TODO: Implement concept drift method

    def fit_partial(self, X, y=None):
        """Fits the model to next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to fit.
            y (int): The label of the instance (Optional for unsupervised models, default=None).

        Returns:
            object: self.
        """
        self.cur_window_X.append(X)

        if y is not None:
            self.cur_window_y.append(y)

        if not self.initial_ref_window and len(self.cur_window_X) < self.window_size and (self.reference_window_X is None or len(self.reference_window_X) < self.window_size):
            self.reference_window_X = self.cur_window_X.copy()
            self.reference_window_y = self.cur_window_y.copy() if y is not None else None
            if len(self.reference_window_X) > self.k:
                self._fit_model()
        elif len(self.cur_window_X) % self.sliding_size == 0:
            self.reference_window_X = np.concatenate([self.reference_window_X, self.cur_window_X], axis=0)
            self.reference_window_X = self.reference_window_X[max(0, len(self.reference_window_X) - self.window_size):]

            if y is not None:
                self.reference_window_y = np.concatenate([self.reference_window_y, self.cur_window_y], axis=0)
                self.reference_window_y = self.reference_window_y[max(0, len(self.reference_window_y) - self.window_size):]

            self.cur_window_X = []
            self.cur_window_y = []
            self._fit_model()

        return self

    def score_partial(self, X):
        """Scores the anomalousness of the next instance.

        Args:
            X (np.float64 array of shape (num_features,)): The instance to score. Higher scores represent more anomalous instances whereas lower scores correspond to more normal instances.

        Returns:
            float: The anomalousness score of the input instance.
        """

        if not self.model:
            score = np.inf 
        else:
            print("score_partial: "+str(self.model.decision_function([X])))
            score = self.model.decision_function([X])[0]

        return score

if __name__ == "__main__":

    from capymoa.stream import NumpyStream
    from capymoa.evaluation import AnomalyDetectionEvaluator
    from data_utils import calculate_roc_pr_auc, clean_score
    import time

    
    # Get the path to the current script
    current_dir = Path(__file__).resolve().parent
    
    # Go two level up
    current_dir = current_dir.parent.parent

    DATA_PATH = current_dir / 'datasets' / 'raw' / 'ScenariosV3'
    #DATA_PATH = current_dir / 'datasets' / 'raw' / 'ScenariosV3'
    PATH_PLOT_FILE_NAME_INTERPRETATION = current_dir / 'notebooks' / 'img_anomalies'
    PATH_PLOT_FILE_NAME_SCORE = current_dir / 'notebooks' / 'img_monitoring_score'

    
    COLS_POS_SMIN = 1
    COLS_POS_SMAX = 2049
    
    # List of files
    spectra_files = [f for f in DATA_PATH.iterdir() if f.suffix == '.csv' and not f.name.startswith('0_')]

    # Results dataframe
    summary_data = []

    NUMBER_RUNS = 1
    WINDOW_SIZE = 10
    K_PARAM = 1
    MODEL = "SWKNN"
    METRIC = "cityblock" 
    SLIDING_WINDOW = 1
    SLEEP_TIME = 1
    #DATASETS_LIST = ["A1_","A2_","A3_","A4_","A5_","A6_","A7_","A8_","A9_","A10_","A11_","A12_"]
    
    #DATASETS_LIST = ["DA1_", "SA1_", "TA1_","DA2_", "SA2_", "TA2_","DA3_", "SA3_", "TA3_"]
    #DATASETS_LIST = ["A1_","A2_","A3_","A4_","A5_","A6_","A7_","A8_","A9_"]
    DATASETS_LIST = ["A1_"]
    MIN_Z_SCORE = 4
    REGION_STUDY = ["386.45:393.38:N2", "773.38:780.40:O2","652.47:659.53:H","304.46:311.54:OH","748.38:752.19:Ar"] 

    f_break=False

    for file_name in spectra_files:
        
        if any(substring in file_name.name for substring in DATASETS_LIST):
            print("File to Use: ",file_name)
            
        else:
            print("Filed not to Use", file_name)
            continue

        # Load spectra and labels data
        full_path_spectra = os.path.join(DATA_PATH, file_name)
        result = pd.read_csv(full_path_spectra, sep=',', low_memory=False, dtype={'CURRENTTIMESTAMP': str})        
        cols = result.columns[COLS_POS_SMIN:COLS_POS_SMAX]
        #cols = ["PRESSURE", "VOLTAGE", "CURRENT", "RS1", "RS2", "RS3"]
        #cols = ["VOLTAGE"]

        

        print("Name DS: ", file_name)
        print("# of Columns: ", len(result.columns))
        print("# of Columns for Wavelengths: ", len(cols))
        
        stream = NumpyStream(result[cols].values, result["ANOMALY?"].values, dataset_name="PV", feature_names=cols)
        #stream = NumpyStream(result[cols].multiply(result["PRESSURE"], axis=0).values, result["ANOMALY?"].values, dataset_name="PV", feature_names=cols)
        schema = stream.get_schema()    
        evaluator = AnomalyDetectionEvaluator(schema)
        
        
        if f_break:
            break
        
        for iter in range(NUMBER_RUNS):

            stream.restart()
            scores = []
            list_auc = []
            row = 0
            learner = SWKNN(p_window_size=WINDOW_SIZE, p_sliding_size=SLIDING_WINDOW, n_neighbors=K_PARAM, metric=METRIC, contamination=0.1)
            #learner = OnlineBootKNN(schema=schema)

            if f_break:
                break
            
            while stream.has_more_instances():
        
                time.sleep(SLEEP_TIME)
            
                instance = stream.next_instance()
                row = row + 1 
                
                print(f'A new instance ({row})...index:', instance.y_label,', label:',instance.y_index)
                print('The new instance:',instance.x)
                
                score = learner.score_partial(instance.x)
                learner.fit_partial(instance.x)
                
                cleaned_score, error_score = clean_score(score)
                scores.append(cleaned_score)
                print(f'Score ({row}):', score)                
                
                if np.isnan(score):
                    f_break = True
                    break
                
                evaluator.update(instance.y_index, score)
                auc = evaluator.auc()
                list_auc.append(auc)
                print(f'AUC ({row}):', auc)
                
                #learner.train(instance)

                #learner.monitor_core_statistics()

                '''
                if learner.z > MIN_Z_SCORE:
                    learner.explain(cols, REGION_STUDY, PATH_PLOT_FILE_NAME_INTERPRETATION, plot_file_name)
                
                learner.plot_core_statistics(PATH_PLOT_FILE_NAME_SCORE, file_name=plot_file_name)
                '''

                

            result['Score'] = list(scores)
            result['AUC'] = list(list_auc)

            # Calculate metrics and store results
            score_column = 'Score'
            gt_column = "ANOMALY?"
            
            roc_auc, pr_auc, max_f1 = calculate_roc_pr_auc(result, gt_column, score_column)
            summary_data.append({
                "iteration": iter,
                "scenario": file_name.name.split("_")[0],
                "method": MODEL,
                "AUC_ROC": roc_auc,
                "AUC_PR": pr_auc,
                "Max_F1": max_f1,
            })
            
    print("########################")
    print("Summary Online Algorithms:")
    # Create DataFrame from collected results
    summary_data = pd.DataFrame(summary_data)
    # Sample pivot table (replace this with your pivot table)
    pivot = summary_data.pivot_table(values=[ "AUC_PR"],
                                        columns=['scenario'], index=['method'], aggfunc='mean')

    # Adding a "Total" row
    pivot['Avg'] = pivot.mean(axis=1)  # Row-wise mean, can use sum(axis=1) for total sum

    # Rounding the pivot table values to 3 decimal places for better readability
    pivot = pivot.round(3)

    # Sorting the pivot table by the "Avg" column in descending order
    pivot = pivot.sort_values(by='Avg', ascending=False)

    # Display the sorted pivot table
    print(pivot)