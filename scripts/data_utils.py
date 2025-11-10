import pandas as pd
pd.set_option('mode.use_inf_as_na', True)
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from capymoa.instance import Instance
import math
import ast
import re
from scipy.stats import zscore

#INF = np.finfo(np.float64).max
INF = 1.0e+200

def calculate_roc_pr_auc(df, gt_column, score_column):
    """Helper function to calculate ROC AUC, PR AUC, and max F1 Score (threshold-independent)"""
    
    # Calculate ROC AUC
    roc_auc = roc_auc_score(df[gt_column], df[score_column])
    
    # Calculate Precision-Recall AUC
    precision, recall, thresholds = precision_recall_curve(df[gt_column], df[score_column])
    pr_auc = auc(recall, precision)

    # Calculate F1 scores for all thresholds and take the max
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # add small epsilon to avoid division by zero
    max_f1 = f1_scores.max()

    return roc_auc, pr_auc, max_f1

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