
import numpy as np
import pandas as pd

import sys, os
from pathlib import Path
from dSalmon import outlier

# Get the path to the current script
current_dir = Path(__file__).resolve().parent

# Go one level up
current_dir = current_dir.parent

# Add the 'scripts' directory to sys.path to be able to import data_utils.py
sys.path.append(str(current_dir))



if __name__ == "__main__":

    from capymoa.stream import NumpyStream
    from capymoa.evaluation import AnomalyDetectionEvaluator
    from data_utils import calculate_roc_pr_auc, clean_score, transform_instance
    import time

    
    # Get the path to the current script
    current_dir = Path(__file__).resolve().parent
    
    # Go two level up
    current_dir = current_dir.parent.parent

    #DATA_PATH = current_dir / 'datasets' / 'raw' / 'ScenariosV4'
    DATA_PATH = current_dir / 'datasets' / 'raw' / 'ScenariosV3'
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
    TRANSF = "NONE"
    SLEEP_TIME = 1
    #DATASETS_LIST = ["A1_","A2_","A3_","A4_","A5_","A6_","A7_","A8_","A9_","A10_","A11_","A12_"]
    
    #DATASETS_LIST = ["DA1_", "SA1_", "TA1_","DA2_", "SA2_", "TA2_","DA3_", "SA3_", "TA3_"]
    #DATASETS_LIST = ["A1_","A2_","A3_","A4_","A5_","A6_","A7_","A8_","A9_"]
    DATASETS_LIST = ["A9_"]
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
            learner = outlier.SWKNN(window=WINDOW_SIZE, k=K_PARAM, metric=METRIC, max_node_size=20, min_node_size=5, split_sampling=5)
            #learner = OnlineBootKNN(schema=schema)

            if f_break:
                break
            
            while stream.has_more_instances():
        
                time.sleep(SLEEP_TIME)
            
                instance = stream.next_instance()
                row = row + 1 
                
                print(f'A new instance ({row})...index:', instance.y_label,', label:',instance.y_index)
                print('The new instance:',instance.x)
                print('Get Window:',learner.get_window())
                
                t_instance = transform_instance(instance, TRANSF)
                score = learner.fit_predict(t_instance.x)
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