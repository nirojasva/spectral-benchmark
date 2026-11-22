import pandas as pd
pd.set_option('mode.use_inf_as_na', True)

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix
from capymoa.instance import Instance
import math
import ast
import re
from scipy.stats import zscore

from vus.metrics import get_metrics
from vus.utils.utility import get_list_anomaly
from sklearn.preprocessing import MinMaxScaler

#INF = np.finfo(np.float64).max
INF = 1.0e+200



def calculate_roc_pr_auc(df, gt_column, score_column, score_direction='direct'):
    """Helper function to calculate ROC AUC, PR AUC, and max F1 Score (threshold-independent)"""
    
    scores = df[score_column].values
    labels = df[gt_column].values

    if score_direction == 'inverse':
        scores = -scores
    elif score_direction == 'direct':
        scores = scores
    else:
        raise ValueError("score direction must be 'direct' or 'inverse' ")

    # Calculate ROC AUC
    roc_auc = roc_auc_score(labels, scores)
    
    # Calculate Precision-Recall AUC
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    # Calculate F1 scores for all thresholds and take the max
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # add small epsilon to avoid division by zero
    max_f1 = f1_scores.max()

    return roc_auc, pr_auc, max_f1

def calculate_performance_metrics(df, gt_column, score_column, t_window_size=None, score_direction='direct'):
    # Calculates performance metrics for anomaly detection.

    # --- 1. Initial Data Preparation ---
    labels = df[gt_column].values
    scores = df[score_column].values

    # Robustness Check: Ensure there are both normal and anomalous points.
    if len(np.unique(labels)) < 2:
        print(f"Warning: Ground truth for '{gt_column}' has only one class. Metrics cannot be calculated.")
        return None

    # --- 2. Handle Score Direction and Offset ---

    # Step 2a: Handle direction first
    if score_direction == 'inverse':
        scores_to_use = -scores
    elif score_direction == 'direct':
        scores_to_use = scores
    else:
        raise ValueError("score_direction must be 'direct' or 'inverse'")
    
    # Step 2b: Calculate Standard Metrics (on original scores) ---
    roc_auc_wtd = roc_auc_score(labels, scores_to_use)

    precision_wtd, recall_wtd, _ = precision_recall_curve(labels, scores_to_use)
    pr_auc_wtd = auc(recall_wtd, precision_wtd)

    # Calculate F1 scores and find the maximum
    f1_scores_wtd = 2 * (precision_wtd * recall_wtd) / (precision_wtd + recall_wtd + 1e-8)
    max_f1_wtd = np.max(f1_scores_wtd)
    
    # Step 3b: Handle slicing second
    if t_window_size is not None:
        final_labels = labels[t_window_size:]
        final_scores = scores_to_use[t_window_size:]
    else:
        # If t_window_size is None, use the full arrays
        final_labels = labels
        final_scores = scores_to_use
    
    if len(final_labels) == 0 or len(np.unique(final_labels)) < 2:
        print(f"Warning: After applying t_window_size, no data or only one class remains. Metrics cannot be calculated.")
        return None
    
    # Step 3c. Calculate Standard Metrics (on shorted scores) ---
    roc_auc = roc_auc_score(final_labels, final_scores)

    precision, recall, thresholds = precision_recall_curve(final_labels, final_scores)
    pr_auc = auc(recall, precision)

    # Calculate F1 scores and find the maximum
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[best_f1_idx]

    # Generate binary predictions based on optimal threshold
    if best_f1_idx < len(thresholds):
        best_threshold = thresholds[best_f1_idx]
    else:
        # Fallback: usually the highest score in the data if we reached the end
        best_threshold = np.max(final_scores)
        
    binary_preds = (final_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(final_labels, binary_preds).ravel()

    # Calculate specific percentages
    pct_detection = recall[best_f1_idx]  # Same as Recall or calculate as: tp / (tp + fn)
    pct_false_positives = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Positive Rate

    # --- 3. Calculate VUS Metrics (on normalized scores) ---
    # Create a new, separate variable for the normalized scores to avoid bugs.
    normalized_scores = MinMaxScaler().fit_transform(final_scores.reshape(-1, 1)).ravel()

    # Robustness Check: Calculate sliding window safely.
    anomaly_lengths = get_list_anomaly(labels)

    slidingWindow = int(np.median(anomaly_lengths)) - 1 

    metrics = get_metrics(normalized_scores, final_labels, metric='all', slidingWindow=slidingWindow)

    # --- 4. Return all metrics in a clear, self-describing dictionary ---
    return roc_auc, pr_auc, max_f1, metrics, roc_auc_wtd, pr_auc_wtd, max_f1_wtd, pct_detection, pct_false_positives, tn, fp, fn, tp, best_threshold

def featurewise_distance(vec1, vec2, metric="cityblock", p=2):
    
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    
    if metric == "cityblock":
        return np.abs(vec1 - vec2)

    elif metric == "minkowski":
        return np.abs(vec1 - vec2) ** p
    
    elif metric == "euclidean":
        return np.abs(vec1 - vec2) ** 2
    else:
        raise ValueError(f"Featurewise distance for '{metric}' is not supported.")
 
def transform_instance(instance:Instance, transf):

    if transf == "MA":
        # Calculate the first-order difference
        t_instance = pd.Series(instance.x).rolling(window=5, min_periods=None).mean().to_numpy()
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance
    if transf == "DIL":
        # Introduce Dilation in a array
        t_instance = instance.x[::5]
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance
    if transf == "FOD":
        # Calculate the Moving Average
        t_instance = np.diff(np.diff(instance.x))
        t_instance = Instance.from_array(instance.schema, t_instance)
        return t_instance
    if transf == "SOD":
        # Calculate the first-order difference
        t_instance = np.diff(np.diff(instance.x))
        t_instance = Instance.from_array(instance.schema, t_instance)
        return t_instance    
    elif transf == "FT":
        t_instance = np.fft.fft(instance.x)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance    
    elif transf == "iFT":
        t_instance = np.fft.ifft(instance.x).real
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance
    elif transf == "SQRT&ZNORM":
        # Calculate Z - Normalized Array
        t_instance = np.power(instance.x, 0.5)
        t_instance = zscore(t_instance, nan_policy='propagate', axis=0)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance
    elif transf == "SQRT":
        t_instance = np.power(instance.x, 0.5)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance
    elif transf == "ZNORM":
        # Calculate Z - Normalized Array
        t_instance = zscore(instance.x, nan_policy='propagate', axis=0)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape( -1))
        return t_instance
    else:
        return instance  # Default: no transformation

def clean_score(score):
    
    error_score = []  # Use a list for better error message formatting
    
    score = ast.literal_eval(score) if isinstance(score, str) and len(score) > 0 else score
    
    # Convert list or array to single value
    if isinstance(score, list):
        if len(score) > 0:  # Ensure the list is not empty
            score = score[0]
            error_score.append("List Output Assigned the First Value.")
            # Check for Inf safely
            if np.isinf(score):
                score = INF
                error_score.append(f"Infinity Output Assigned to {INF}.")
        else:
            score = 0
            error_score.append("Empty List Assigned to 0.")
    
    if isinstance(score, np.ndarray):
        if score.size > 0:  # Ensure the array is not empty
            score = score[0]
            error_score.append("Array Output Assigned the First Value.")
            # Check for Inf safely
            if np.isinf(score):
                score = INF
                error_score.append(f"Infinity Output Assigned to {INF}.")
            
        else:
            score = 0
            error_score.append("Empty Array Assigned to 0.")

    if score is None:
        score = 0
        error_score.append("None Output Assigned to 0.")

    # Check for NaN safely
    if pd.isna(score):
        score = 0
        error_score.append("NaN Output Assigned to 0 (Pandas).")
    
    # Check for Inf safely
    if np.isinf(score):
        score = INF
        error_score.append(f"Infinity Output Assigned to {INF}.")
    
    # Check for NaN safely
    if math.isnan(score):
        score = 0
        error_score.append("NaN Output Assigned to 0 (Math).")
    
    # Handle non-numeric types
    if isinstance(score, float) and (score != score):
        score = 0
        error_score.append("NaN Output Assigned to 0 (Float).")
    
    # Handle non-numeric types
    if isinstance(score, str):
        score = 0
        error_score.append("String Output Assigned to 0.")
    
    score = float(score)  # force invalid values to an Error

    message = " | ".join(error_score)
    
    return score, message

def split_summary_methods(method_window_and_param):
    match = re.match(r"([^_]+)_([^_]+)_\{(.+)\}", method_window_and_param)
    if match:
        method = match.group(1)
        window = match.group(2)
        params = "{" + match.group(3) + "}"
        return method, window, params
    else:
        return method_window_and_param, None, None