"""
Helper functions for OOD validation during training.
Lightweight validation on OOD datasets (FS_LostFound, fs_static) at epoch end.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from sklearn.metrics import average_precision_score, roc_curve


def fpr_at_95_tpr(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute False Positive Rate at 95% True Positive Rate.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0 = normal, 1 = anomaly)
        
    Returns:
        FPR at 95% TPR (as a fraction, not percentage)
    """
    scores = scores.flatten()
    labels = labels.flatten()
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Find FPR at TPR >= 0.95
    if len(tpr[tpr >= 0.95]) == 0:
        return 1.0  # Worst case
    
    fpr_at_95 = fpr[tpr >= 0.95][0]
    return fpr_at_95


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Area Under Precision-Recall Curve.
    
    Args:
        scores: Anomaly scores (higher = more anomalous)
        labels: Ground truth labels (0 = normal, 1 = anomaly)
        
    Returns:
        AUPRC value (0-1)
    """
    scores = scores.flatten()
    labels = labels.flatten()
    
    return average_precision_score(labels, scores)


def compute_anomaly_map_msp(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute anomaly map using Maximum Softmax Probability (MSP).
    
    Args:
        logits: Per-pixel semantic logits [C, H, W]
        temperature: Temperature scaling factor
        
    Returns:
        Anomaly map [H, W] where higher values = more anomalous
    """
    # Apply temperature scaling
    logits_scaled = logits / temperature
    probs = F.softmax(logits_scaled, dim=0)
    
    # MSP: Score = 1 - max(P(y|x))
    msp = probs.max(dim=0).values
    anomaly_map = 1.0 - msp
    
    return anomaly_map
