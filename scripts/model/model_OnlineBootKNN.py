
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import sys, os
from pathlib import Path

# Get the path to the current script
current_dir = Path(__file__).resolve().parent

# Go one level up
current_dir = current_dir.parent

# Add the 'scripts' directory to sys.path to be able to import data_utils.py
sys.path.append(str(current_dir))


from data_utils import transform_instance, featurewise_distance

class OnlineBootKNN(AnomalyDetector):
    """
    Anomaly detection using an online ensemble of k-nearest neighbors (KNN).
    This class processes data in chunks, detects anomalies based on statistical thresholds,
    and uses an ensemble approach to increase robustness.

    Parameters:
    - schema: Optional, schema for data processing (if needed).
    - random_seed: Optional, seed for random number generation.
    - chunk_size: Size of each data chunk to process.
    - ensemble_size: Number of KNN models in the ensemble.
    - algorithm: KNN algorithm to use ('brute', 'kd_tree', etc.).
    - n_jobs: Number of CPU cores to use for parallel processing.
    - window_size: Size of the sliding window for statistics.
    - dmetric: Distance metric for KNN (e.g., 'cityblock', 'euclidean').
    - transf: Optional, transformation function to apply to data (e.g., Z-Normalization(ZNORM), First Order Difference(FOD), None, etc).
    - alpha: Significance level for statistical tests (default is 0.05).
    """
    def __init__(
        self,
        schema=None,
        random_seed=None,
        chunk_size=240,
        ensemble_size=240,
        algorithm='brute',
        n_jobs=-1,
        window_size=120,
        dmetric="cityblock",
        transf="ZNORM",
        alpha=0.05,  # 5% significance level for anomaly detection
        no_bootstrapp=False,
        no_z_score=False
    ):
        # Initialize base class and set random seed
        super().__init__(schema=schema, random_seed=random_seed)
        self.random_generator = np.random.RandomState(random_seed)
        
        # Initialize parameters
        self.chunk_size = chunk_size
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.window_size = window_size
        self.dmetric = dmetric
        self.transf = transf
        self.alpha = alpha
        
        self.data_window = []  # Sliding window of recent data
        self.ensemble = [
            NearestNeighbors(n_neighbors=1, algorithm=algorithm, n_jobs=self.n_jobs, metric=self.dmetric)
            for _ in range(self.ensemble_size)
        ]
        
        self.chunks = []  # Data chunks
        self.init = True  # Flag to indicate if the model is initialized
        self.last_value_is_anomaly = False 
        self.reset_threshold = 4200  # Number of values before resetting statistics 
        self.count_reset = None # Number of times the counter n is reseted 
        self.normal_reference = None  # To track reference for normal data distribution
        self.abnormal_reference = None  # To track reference for abnormal data distribution
        self.z_critical_one_tail = norm.ppf(1 - self.alpha)  # Critical value for one-tailed test
        
        # Monitoring statistics for debugging/monitoring purposes

        self.z = 0.0  # Initial Z-Score Distance
        self.mean = np.nan
        self.std_dev = np.nan  # Standard deviation of the data
        self.min_dist = np.nan  # Minimum distance for anomaly detection
        self.n = 0  # Number of data points processed
        self.n_anomalies = 0  # Number of anomalies detected
        self.sum_squares = np.nan
        self.p_random_number = np.nan  # Probability for random anomaly detection
        
        self.z_scores_to_monitor = []
        self.means_to_monitor = []
        self.std_devs_to_monitor = []
        self.min_dists_to_monitor = []
        self.z_thresholds_to_monitor = []
        self.p_random_number_to_monitor = []
        self.no_bootstrapp = no_bootstrapp
        self.no_z_score = no_z_score



    def __str__(self):
        return "OnlineBootKNN"

    def train(self, instance):
        """
        Train the model with a new instance.
        """
        instance = transform_instance(instance, self.transf)

        data = instance.x
        self._learn_batch(data)

    def score_instance(self, instance: Instance):
        """
        Scores an instance using the models in the ensemble.
        Each model predicts distances, and the minimum distance is returned.
        """
        instance = transform_instance(instance, self.transf)

        data = instance.x.reshape(1, -1)  # Reshape data to match model expectations
        distances = []
        references = []

        for i, model in enumerate(self.ensemble):
            try:
                # Ensure the model is fitted before making predictions
                check_is_fitted(model)
                dist, idxv = model.kneighbors(data)
                distances.append(dist[0][0])  # Add the nearest neighbor distance

                # Retrieve nearest neighbors for each model
                v = self.chunks[i][idxv[0][0]]
                references.append(v)

            except NotFittedError:
                print(f"Model {i} is not fitted.")
                distances.append(0)  
            except Exception as exc:
                print(f"An error occurred while scoring the instance: {exc}")
                distances.append(None)

        min_dist = np.min(distances)  # Find the minimum distance
        self.min_dist = min_dist
        
        #return mininum distance for ablation study
        if self.no_z_score:
            return self.min_dist
        
        if not self.init:
            if  (self.n == 0) | (self.n > self.reset_threshold):
                #Init Stats
                self.count_reset = 0 if self.count_reset is None else self.count_reset + 1
                
                self.start_statistics(min_dist)
            else:
                #Update Stats
                self.update_z_score(min_dist)
                self.update_statistics(min_dist)
                
        if self.last_value_is_anomaly:
            self.n_anomalies += 1
            min_pos_dist = np.argmin(distances)  # Find the minimum position distance
            self.normal_reference = references[min_pos_dist]
            self.abnormal_reference = data.reshape(-1)

        # Return the minimum distance among the models
        return self.z

    def start_statistics(self, new_dist):
        """
        Initialize statistics when the reset threshold is reached or at an initial step.
        """
        self.z = 0.0          
        self.n = 1
        self.mean = new_dist
        self.sum_squares = 0.0
        self.std_dev = 0.0
        self.last_value_is_anomaly = False

    def update_statistics(self, new_dist):
        """
        Increment statistics and update mean and standard deviation.
        """
        self.n += 1
        delta = new_dist - self.mean
        
        self.mean += delta / self.n
        self.sum_squares += delta ** 2
        
        if self.n > 1:
            self.std_dev = math.sqrt(self.sum_squares / (self.n - 1))
        else:
            self.std_dev = np.nan
    
    def update_z_score(self, new_dist):
        """
        Calculate the z-score and check for anomalies.
        """
        if self.n == 1:
            self.z = 0
            self.std_dev = 0
        elif self.std_dev != 0 and not pd.isna(self.std_dev):
            self.z = (new_dist - self.mean) / self.std_dev
        else:
            self.z = np.nan
        
        if self.z > self.z_critical_one_tail and not pd.isna(self.z):
            self.last_value_is_anomaly = True
        else:
            self.last_value_is_anomaly = False
    
    def _learn_batch(self, data):
        
        if self.init == False:
            
            # Train each model in the ensemble using bootstrap sampling
            
            for i in range(self.ensemble_size):

                #if true "self.no_bootstrapp" do not apply bootstrapp for ablation study
                if self.no_bootstrapp:
                    self.p_random_number = 1
                
                elif self.last_value_is_anomaly:
                    self.p_random_number = 0
                
                else:
                    self.p_random_number = self.random_generator.poisson(1)
                    
                
                for j in range(self.p_random_number):
                    # Remove the first element
                    self.chunks[i] = self.chunks[i][1:]
                    
                    # Add the new vector to the end of the array
                    self.chunks[i].append(data)
                    

                # Fit the model with the bootstrap sample
                
                self.ensemble[i].fit(self.chunks[i])

        else:
            """Update the model with a new batch of data."""

            self.data_window.append(data)        

            if (len(self.data_window) >= self.window_size) and self.init == True:
                
                self.data_window = np.array(self.data_window)

                for i in range(self.ensemble_size):
                    
                    print("Data window size:", len(self.data_window))
                    
                    #if true "self.no_bootstrapp" do not apply bootstrapp for ablation study
                    if not self.no_bootstrapp:
                        indices = self.random_generator.choice(len(self.data_window), size=self.chunk_size, replace=True)
                    else:
                        indices = range(len(self.data_window))
                    
                    print("Indices:", indices)
                    
                    chunk = self.data_window[indices]

                    print("Chunk:", chunk)


                    self.chunks.append(list(chunk))
                    # Fit the model with the bootstrap sample
                    self.ensemble[i].fit(chunk)

                self.data_window = []  # Clear window after update
                self.init = False

    def predict(self, data: np.ndarray):
        """
        Predict method (to be implemented by subclasses).
        """
        raise NotImplementedError("The 'predict' method must be implemented by subclasses.")

    def explain(self, headers, region_study_list, path: str, file_name: str):
        """
        Explain method to visualize anomalies in time series data.
        Optimized for publication (Grayscale compatible, 10pt fonts).
        """
        
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 12
        })

        print("Is Anomaly?:", self.last_value_is_anomaly)
        print("Anomalies:", self.n_anomalies)

        headers = headers.astype(float)
        region_study_list = np.array(region_study_list).astype(str)

        if not self.last_value_is_anomaly:
            return

        if self.normal_reference is None or self.abnormal_reference is None:
            raise ValueError("Error: References are None.")

        # Compute differences
        differences = featurewise_distance(self.abnormal_reference, self.normal_reference, metric=self.dmetric)

        # Create figure
        fig, ax1 = plt.subplots(figsize=(10, 5)) 

        ax1.bar(headers, differences, label=f"Difference (Z: {round(self.z, 2)})", 
                color='orange', alpha=0.6, width=(headers[1]-headers[0]))

        ax1.set_xlabel('Wavelengths (nm)')
        ax1.set_ylabel('Intensity Difference')
        
        
        y_min, y_max = ax1.get_ylim()
        
        for i, rs in enumerate(region_study_list):     
            rs_s = float(rs.split(":")[0])
            rs_f = float(rs.split(":")[1])
            comp = str(rs.split(":")[2])
            
            ax1.axvline(x=rs_s, color='black', linestyle=':', linewidth=1, alpha=0.6)
            ax1.axvline(x=rs_f, color='black', linestyle=':', linewidth=1, alpha=0.6)

            text_pos_y = y_max * (0.9 - (i * 0.05)) 
            ax1.text(rs_s, text_pos_y, f' {comp}', fontsize=8, verticalalignment='top')

        ax1.legend(loc='upper right', frameon=True)

        plt.title(f"Anomaly Explanation (Total Anomalies: {self.n_anomalies})")
        
        plt.tight_layout()
        
        if path and file_name:
            plt.savefig(f"{path}/{file_name}.pdf", format='pdf', bbox_inches='tight')
        
        plt.show()

    def monitor_core_statistics(self):
        # Print monitored stats
        print("Mean:", self.mean)
        print("Standard Deviation:", self.std_dev)
        print("Minimum Distance:", self.min_dist)
        print("P Random Number:", self.p_random_number)
        print("Number of Values:", self.n)
        print("Last Value is Anomaly:", self.last_value_is_anomaly)
        print("Number of Anomalies:", self.n_anomalies)
        print(f"Z Threshold for One tail at {self.alpha}:", self.z_critical_one_tail)
        print("Z Score:", self.z)
    
    def plot_core_statistics(self, path: str, file_name: str):
        # Append monitored stats
        self.means_to_monitor.append(self.mean)
        self.std_devs_to_monitor.append(self.std_dev)
        self.min_dists_to_monitor.append(self.min_dist)
        self.z_scores_to_monitor.append(self.z)
        self.z_thresholds_to_monitor.append(self.z_critical_one_tail)
        self.p_random_number_to_monitor.append(self.p_random_number)

        """
        Plot core statistics: Mean, Min Dist, and Std Dev.
        """
        plt.clf()
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Primary y-axis for Mean and Min Dist
        ax1.plot(self.means_to_monitor, label='Accumulated Mean Dist', color='blue', linestyle='--', marker='o', markersize=2)
        ax1.plot(self.min_dists_to_monitor, label='Min Dist', color='green', marker='s', markersize=2)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Accumulated Mean & Min Dist', fontsize=12)
        ax1.tick_params(axis='y')
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(loc='upper right', fontsize=10)

        # Secondary y-axis for Std Dev
        ax2 = ax1.twinx()
        ax2.plot(self.std_devs_to_monitor, label='Accumulated Std Dev', color='orange', linestyle='--', marker='^', markersize=2)
        ax2.set_ylabel('Accumulated Std Dev', fontsize=12)
        ax2.tick_params(axis='y')
        ax2.legend(loc='lower left', fontsize=10)

        plt.title('Real-time Core Stats: Accumulated Mean Dist, Min Dist & Accumulated Std Dev', fontsize=14, pad=20)
        plt.tight_layout()

        # Save plot
        plot_path1 = os.path.join(path, f"{file_name}_core_stats.pdf")
        plt.savefig(plot_path1, format="pdf", bbox_inches='tight')
        plt.close()
        print(f"Core stats plot saved at {plot_path1}")

        """
        Plot anomaly detection metrics: Z Scores, Z Thresholds, and P Random Number.
        """
        plt.clf()
        fig, ax3 = plt.subplots(figsize=(10, 6))

        # Primary y-axis for Z Scores and Z Thresholds
        ax3.plot(self.z_scores_to_monitor, label='Z Score', color='red', marker='o', markersize=4)
        ax3.plot(self.z_thresholds_to_monitor, label='Z Threshold', color='purple', linestyle=':', marker='s', markersize=2)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Z Scores', fontsize=12)
        ax3.tick_params(axis='y')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(loc='upper right', fontsize=10)

        # Secondary y-axis for P Random Number
        ax4 = ax3.twinx()
        # Create x-axis values (time or index)
        x_values = range(len(self.p_random_number_to_monitor))

        # Plot bars
        ax4.bar(x_values, self.p_random_number_to_monitor, label='P Random Number', color='black', alpha=0.8)
        ax4.set_ylabel('P Random Number', fontsize=12)
        ax4.tick_params(axis='y')
        ax4.legend(loc='lower left', fontsize=10)

        plt.title('Real-time Anomaly Detection: Z Scores, Thresholds & P Random Number', fontsize=14, pad=20)
        plt.tight_layout()

        # Save plot
        plot_path2 = os.path.join(path, f"{file_name}_z_scores_and_p.pdf")
        plt.savefig(plot_path2, format="pdf", bbox_inches='tight')
        plt.close()
        print(f"Z scores plot saved at {plot_path2}")

if __name__ == "__main__":

    from capymoa.stream import NumpyStream
    from capymoa.evaluation import AnomalyDetectionEvaluator
    from data_utils import calculate_roc_pr_auc
    import time
    params = {
    'font.size': 10,           
    'axes.labelsize': 10,      
    'xtick.labelsize': 8,      
    'ytick.labelsize': 8,      
    'legend.fontsize': 8,      
    'figure.titlesize': 12
    }
    plt.rcParams.update(params)
    
    # Get the path to the current script
    current_dir = Path(__file__).resolve().parent
    
    # Go two level up
    current_dir = current_dir.parent.parent

    #DATA_PATH = current_dir / 'datasets' / 'raw' / 'ScenariosV4'
    #DATA_PATH = current_dir / 'datasets' / 'raw' / 'ScenariosV3'
    DATA_PATH = current_dir / 'datasets' / 'raw'
    PATH_PLOT_FILE_NAME_INTERPRETATION = current_dir / 'notebooks' / 'img_anomalies'
    PATH_PLOT_FILE_NAME_SCORE = current_dir / 'notebooks' / 'img_monitoring_score'

    
    COLS_POS_SMIN = 1
    COLS_POS_SMAX = 2049
    
    # List of files
    spectra_files = [f for f in DATA_PATH.iterdir() if f.suffix == '.csv' and not f.name.startswith('0_')]

    # Results dataframe
    summary_data = []

    NUMBER_RUNS = 1
    WINDOW_SIZE = 120
    MODEL = "OnlineBootKNN"
    TRANF = "ZNORM"
    CHUNCK_SIZE = 240
    ENSEMBLE_SIZE = 240
    NO_BOOTSTRAPP = False
    NO_ZSCORE = False
    DMETRIC = "cityblock"
    ALGO = "brute"
    ALPHA = 0.05
    SLEEP_TIME = 0
    #DATASETS_LIST = ["A1_","A2_","A3_","A4_","A5_","A6_","A7_","A8_","A9_"]
    
    DATASETS_LIST = ["A6_"]
    #DATASETS_LIST = ["DA1_", "SA1_", "TA1_","DA2_", "SA2_", "TA2_","DA3_", "SA3_", "TA3_"]
    MIN_Z_SCORE = 4
    REGION_STUDY = ["386.45:393.38:N2", "773.38:780.40:O2","652.47:659.53:H","304.46:311.54:OH","748.38:752.19:Ar"] 

    
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
        print("Name DS: ", file_name)
        print("# of Columns: ", len(result.columns))
        print("# of Columns for Wavelengths: ", len(cols))
        
        stream = NumpyStream(result[cols].values, result["ANOMALY?"].values, dataset_name="PV", feature_names=cols)
        schema = stream.get_schema()    
        evaluator = AnomalyDetectionEvaluator(schema)

        for iter in range(NUMBER_RUNS):

            stream.restart()
            scores = []
            list_auc = []
            row = 0
            learner = OnlineBootKNN(schema=schema, window_size=WINDOW_SIZE, chunk_size=CHUNCK_SIZE,  ensemble_size=ENSEMBLE_SIZE, dmetric=DMETRIC, transf=TRANF, alpha=ALPHA, algorithm=ALGO, no_bootstrapp=NO_BOOTSTRAPP, no_z_score=NO_ZSCORE)
#           learner = OnlineBootKNN(schema=schema)
            while stream.has_more_instances():
        
                time.sleep(SLEEP_TIME)

                instance = stream.next_instance()
                row = row + 1 
                
                print(f'A new instance ({row})...index:', instance.y_label,', label:',instance.y_index)
                print('The new instance:',instance.x)
                
                score = learner.score_instance(instance)

                scores.append(score)
                print(f'Score ({row}):', score)                
 
                evaluator.update(instance.y_index, score)
                auc = evaluator.auc()
                list_auc.append(auc)
                print(f'AUC ({row}):', auc)
                
                plot_file_name = str(file_name.name.split("_")[0])+"_transf_"+TRANF
                
                if learner.z > MIN_Z_SCORE:
                    learner.explain(cols, REGION_STUDY, PATH_PLOT_FILE_NAME_INTERPRETATION, plot_file_name)
                #learner.monitor_core_statistics()
                #learner.plot_core_statistics(PATH_PLOT_FILE_NAME_SCORE, file_name=plot_file_name)
                
                learner.train(instance)

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