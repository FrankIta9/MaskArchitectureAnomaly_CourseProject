# ---------------------------------------------------------------
# Evaluation script for EoMT model on anomaly segmentation datasets
# Computes AuPRC and FPR95 metrics for multiple anomaly datasets
# ---------------------------------------------------------------

import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn.functional as F

# Import EoMT model
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eomt'))
from models.eomt import EoMT
from models.vit import ViTEncoder

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((640, 640), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((640, 640), Image.NEAREST),
])


def load_eomt_model(checkpoint_path: str, num_classes: int = 19, device: str = "cuda"):
    """
    Load EoMT model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_classes: Number of classes (19 for Cityscapes)
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Initialize encoder (DINOv2 backbone)
    from models.vit import ViTEncoder
    encoder = ViTEncoder(
        model_name="dinov2_vitb14",
        img_size=(640, 640),
        ckpt_path=None,
    )
    
    # Initialize EoMT model
    model = EoMT(
        encoder=encoder,
        num_classes=num_classes,
        num_q=100,
        num_blocks=4,
        masked_attn_enabled=True,
    )
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Remove "network." prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("network."):
                new_state_dict[k[8:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using pretrained weights")
    
    model = model.to(device)
    model.eval()
    return model


def compute_anomaly_score_msp(mask_logits, class_logits):
    """
    Compute anomaly score using Maximum Softmax Probability (MSP).
    
    Args:
        mask_logits: Mask logits of shape (batch, num_queries, H, W)
        class_logits: Class logits of shape (batch, num_queries, num_classes + 1)
        
    Returns:
        Anomaly score map of shape (batch, H, W)
    """
    # Convert to per-pixel logits
    mask_probs = mask_logits.sigmoid()  # (batch, num_queries, H, W)
    class_probs = class_logits.softmax(dim=-1)  # (batch, num_queries, num_classes + 1)
    
    # Remove "no object" class (last class)
    class_probs = class_probs[..., :-1]  # (batch, num_queries, num_classes)
    
    # Compute per-pixel class probabilities
    per_pixel_probs = torch.einsum("bqhw, bqc -> bchw", mask_probs, class_probs)
    # per_pixel_probs: (batch, num_classes, H, W)
    
    # Maximum probability per pixel
    max_probs = per_pixel_probs.max(dim=1)[0]  # (batch, H, W)
    
    # Anomaly score = 1 - max_probability
    anomaly_score = 1.0 - max_probs
    
    return anomaly_score


def compute_anomaly_score_max_logit(mask_logits, class_logits):
    """
    Compute anomaly score using Maximum Logit.
    
    Args:
        mask_logits: Mask logits of shape (batch, num_queries, H, W)
        class_logits: Class logits of shape (batch, num_queries, num_classes + 1)
        
    Returns:
        Anomaly score map of shape (batch, H, W)
    """
    # Convert to per-pixel logits
    mask_probs = mask_logits.sigmoid()  # (batch, num_queries, H, W)
    class_logits_clean = class_logits[..., :-1]  # Remove "no object" class
    
    # Compute per-pixel class logits
    per_pixel_logits = torch.einsum("bqhw, bqc -> bchw", mask_probs, class_logits_clean)
    # per_pixel_logits: (batch, num_classes, H, W)
    
    # Maximum logit per pixel
    max_logits = per_pixel_logits.max(dim=1)[0]  # (batch, H, W)
    
    # Normalize to [0, 1] range for anomaly score
    # Use negative of max logit as anomaly score (lower confidence = higher anomaly)
    max_logits_norm = torch.sigmoid(-max_logits)
    
    return max_logits_norm


def compute_anomaly_score_rba(mask_logits, class_logits):
    """
    Compute anomaly score using Rejected by All (RbA) method.
    
    RbA considers a pixel anomalous if it is rejected (low probability) by all queries.
    
    Args:
        mask_logits: Mask logits of shape (batch, num_queries, H, W)
        class_logits: Class logits of shape (batch, num_queries, num_classes + 1)
        
    Returns:
        Anomaly score map of shape (batch, H, W)
    """
    # Mask probabilities
    mask_probs = mask_logits.sigmoid()  # (batch, num_queries, H, W)
    
    # Class probabilities (excluding "no object")
    class_probs = class_logits.softmax(dim=-1)[..., :-1]  # (batch, num_queries, num_classes)
    
    # For each query, compute the maximum class probability
    max_class_probs = class_probs.max(dim=-1)[0]  # (batch, num_queries)
    
    # Weight by mask probability
    query_scores = mask_probs * max_class_probs.unsqueeze(-1).unsqueeze(-1)
    # query_scores: (batch, num_queries, H, W)
    
    # Pixel is anomalous if rejected by all queries (all scores are low)
    # Anomaly score = 1 - max(query_scores) across queries
    max_query_scores = query_scores.max(dim=1)[0]  # (batch, H, W)
    anomaly_score = 1.0 - max_query_scores
    
    return anomaly_score


def evaluate_dataset(
    model,
    image_paths,
    method="msp",
    device="cuda",
):
    """
    Evaluate model on a dataset.
    
    Args:
        model: EoMT model
        image_paths: List of image paths
        method: Anomaly detection method ("msp", "max_logit", "rba")
        device: Device to run inference on
        
    Returns:
        Tuple of (anomaly_scores, ground_truths)
    """
    anomaly_scores_list = []
    ground_truths_list = []
    
    model.eval()
    
    with torch.no_grad():
        for img_path in image_paths:
            # Load and preprocess image
            img = Image.open(img_path).convert("RGB")
            img_tensor = input_transform(img).unsqueeze(0).to(device)
            
            # Forward pass
            mask_logits_per_layer, class_logits_per_layer = model(img_tensor / 255.0)
            
            # Use the final layer predictions
            mask_logits = mask_logits_per_layer[-1]  # (batch, num_queries, H, W)
            class_logits = class_logits_per_layer[-1]  # (batch, num_queries, num_classes + 1)
            
            # Interpolate to original image size
            original_size = img.size[::-1]  # (H, W)
            mask_logits = F.interpolate(
                mask_logits,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            
            # Compute anomaly score
            if method == "msp":
                anomaly_score = compute_anomaly_score_msp(mask_logits, class_logits)
            elif method == "max_logit":
                anomaly_score = compute_anomaly_score_max_logit(mask_logits, class_logits)
            elif method == "rba":
                anomaly_score = compute_anomaly_score_rba(mask_logits, class_logits)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Load ground truth
            gt_path = img_path.replace("images", "labels_masks")
            if "RoadObsticle21" in gt_path:
                gt_path = gt_path.replace("webp", "png")
            elif "fs_static" in gt_path:
                gt_path = gt_path.replace("jpg", "png")
            elif "RoadAnomaly" in gt_path:
                gt_path = gt_path.replace("jpg", "png")
            
            if os.path.exists(gt_path):
                gt_mask = Image.open(gt_path)
                gt_mask = target_transform(gt_mask)
                gt_array = np.array(gt_mask)
                
                # Normalize ground truth to binary (0 = normal, 1 = anomaly)
                if "RoadAnomaly" in gt_path:
                    gt_array = np.where(gt_array == 2, 1, 0)
                elif "LostAndFound" in gt_path:
                    gt_array = np.where(gt_array == 0, 255, gt_array)
                    gt_array = np.where(gt_array == 1, 0, gt_array)
                    gt_array = np.where((gt_array > 1) & (gt_array < 201), 1, gt_array)
                    gt_array = np.where(gt_array == 255, 0, gt_array)
                elif "Streethazard" in gt_path:
                    gt_array = np.where(gt_array == 14, 255, gt_array)
                    gt_array = np.where(gt_array < 20, 0, gt_array)
                    gt_array = np.where(gt_array == 255, 1, gt_array)
                else:
                    # Binary: 0 = normal, 1 = anomaly
                    gt_array = np.where(gt_array > 0, 1, 0)
                
                # Skip if no anomalies in ground truth
                if 1 not in np.unique(gt_array):
                    continue
                
                # Resize anomaly score to match ground truth
                anomaly_score_np = anomaly_score[0].cpu().numpy()
                if anomaly_score_np.shape != gt_array.shape:
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        gt_array.shape[0] / anomaly_score_np.shape[0],
                        gt_array.shape[1] / anomaly_score_np.shape[1],
                    )
                    anomaly_score_np = zoom(anomaly_score_np, zoom_factors, order=1)
                
                anomaly_scores_list.append(anomaly_score_np)
                ground_truths_list.append(gt_array)
    
    return anomaly_scores_list, ground_truths_list


def compute_metrics(anomaly_scores_list, ground_truths_list):
    """
    Compute AuPRC and FPR95 metrics.
    
    Args:
        anomaly_scores_list: List of anomaly score maps
        ground_truths_list: List of ground truth masks
        
    Returns:
        Dictionary with metrics
    """
    # Flatten all scores and ground truths
    all_scores = []
    all_labels = []
    
    for scores, labels in zip(anomaly_scores_list, ground_truths_list):
        all_scores.extend(scores.flatten())
        all_labels.extend(labels.flatten())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute AuPRC
    auprc = average_precision_score(all_labels, all_scores)
    
    # Compute FPR95
    fpr95 = fpr_at_95_tpr(all_scores, all_labels)
    
    return {
        "auprc": auprc * 100.0,
        "fpr95": fpr95 * 100.0,
    }


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to EoMT checkpoint",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Glob pattern for input images (e.g., 'path/to/dataset/images/*.jpg')",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="msp",
        choices=["msp", "max_logit", "rba"],
        help="Anomaly detection method",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.txt",
        help="Output file for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on",
    )
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_eomt_model(args.checkpoint, device=args.device)
    
    # Get image paths
    image_paths = sorted(glob.glob(os.path.expanduser(args.input)))
    print(f"Found {len(image_paths)} images")
    
    # Evaluate
    print(f"Evaluating with method: {args.method}")
    anomaly_scores, ground_truths = evaluate_dataset(
        model,
        image_paths,
        method=args.method,
        device=args.device,
    )
    
    # Compute metrics
    metrics = compute_metrics(anomaly_scores, ground_truths)
    
    # Print and save results
    print(f"\nResults for {args.method}:")
    print(f"AuPRC: {metrics['auprc']:.2f}%")
    print(f"FPR95: {metrics['fpr95']:.2f}%")
    
    with open(args.output, "a") as f:
        f.write(f"\n{args.method} - AuPRC: {metrics['auprc']:.2f}%, FPR95: {metrics['fpr95']:.2f}%\n")
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
