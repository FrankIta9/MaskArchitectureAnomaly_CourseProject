"""
Out-of-Distribution Detection Metrics
Standard metrics for anomaly segmentation evaluation
"""

import numpy as np
from sklearn.metrics import roc_curve


def fpr_at_95_tpr(scores, labels):
    """
    Compute False Positive Rate at 95% True Positive Rate.
    
    This is a standard metric for OOD detection. Lower is better.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0 = normal, 1 = anomaly)
        
    Returns:
        FPR at 95% TPR (as a fraction, not percentage)
    """
    # Flatten if needed
    if isinstance(scores, list):
        scores = np.array(scores).flatten()
    if isinstance(labels, list):
        labels = np.array(labels).flatten()
    
    scores = scores.flatten()
    labels = labels.flatten()
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find FPR at TPR >= 0.95
    if len(tpr[tpr >= 0.95]) == 0:
        # If we can't achieve 95% TPR, return 1.0 (worst case)
        return 1.0
    
    fpr_at_95 = fpr[tpr >= 0.95][0]
    
    return fpr_at_95


def auroc(scores, labels):
    """
    Compute Area Under ROC Curve.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0 = normal, 1 = anomaly)
        
    Returns:
        AUROC value (0-1)
    """
    from sklearn.metrics import roc_auc_score
    
    if isinstance(scores, list):
        scores = np.array(scores).flatten()
    if isinstance(labels, list):
        labels = np.array(labels).flatten()
    
    scores = scores.flatten()
    labels = labels.flatten()
    
    return roc_auc_score(labels, scores)


def auprc(scores, labels):
    """
    Compute Area Under Precision-Recall Curve.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0 = normal, 1 = anomaly)
        
    Returns:
        AUPRC value (0-1)
    """
    from sklearn.metrics import average_precision_score
    
    if isinstance(scores, list):
        scores = np.array(scores).flatten()
    if isinstance(labels, list):
        labels = np.array(labels).flatten()
    
    scores = scores.flatten()
    labels = labels.flatten()
    
    return average_precision_score(labels, scores)
