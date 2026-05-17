# utils.py
# all shared helper functions used across the pipeline
# config loading, logging, dataset splitting, loss functions,
# evaluation metrics (Dice, IoU, HD95), volumetric quantification,
# bland-altman analysis, figure saving




# --- standard library imports ---
from pathlib import Path          # handles file/folder paths
from datetime import datetime     # for timestamping log entries
from typing import Dict, List, Tuple  # type hints for function signatures
import csv                        # writing results to CSV
import json                       # writing results to JSON

# --- third party imports ---
import yaml                       # reading config.yaml
import torch                      # main deep learning framework
import torch.nn as nn             # neural network layers and losses
import torch.nn.functional as F   # softmax, one_hot etc
import numpy as np                # numerical array operations
import matplotlib.pyplot as plt   # creating and saving plots

# try importing scipy for HD95 - optional but needed for that metric
# if not installed, HD95 will return nan instead of crashing
try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False




# CONFIG AND UTILITY FUNCTIONS

def load_config(config_path: str = "config.yaml") -> dict:
    # reads config.yaml and returns it as a python dictionary
    # all hyperparameters (batch size, lr, paths etc) stored there
    # default path is config.yaml in root of project
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)  # safe_load reads yaml without executing any code


def ensure_dir(path: str) -> None:
    # creates folder at given path if it doesnt already exist
    # parents=True also creates any missing parent folders
    # exist_ok=True wont throw error if folder already exists
    Path(path).mkdir(parents=True, exist_ok=True)


def append_log(log_path: str, message: str) -> None:
    # appends a timestamped message to the training log file
    # opens in append mode ("a") so previous entries arent overwritten
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # e.g. 2026-05-01 14:32:01
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")




# SAFE DIVISION

def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # adds tiny number (eps) to top and bottom to avoid dividing by zero
    # important when a class is completely absent from a batch
    # eps = 1e-8 is so small it has negligible effect on the result
    return (numerator + eps) / (denominator + eps)




# DATASET SPLITTING

def split_dataset(dataset, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    # splits dataset into train, val, test subsets
    # ratios must add up to 1.0 (e.g. 0.7, 0.15, 0.15)
    # seed=42 makes the split reproducible - same result every run
    total = len(dataset)

    # check ratios add up to 1.0 (allows tiny floating point errors)
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    # work out how many samples go in each split
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size  # remainder goes to test to avoid rounding issues

    # make sure none of the splits end up empty
    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("Dataset too small for requested train/val/test split.")

    # fixed seed so split is identical every run
    generator = torch.Generator().manual_seed(seed)

    # randomly split and return the three subsets
    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )




# EVALUATION METRICS

def multiclass_dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> float:
    # calculates average Dice score across all foreground classes
    # Dice = 2 * overlap / (total predicted + total true)
    # 1.0 = perfect, 0.0 = no overlap

    # convert raw logits to predicted class per pixel (highest score wins)
    pred = torch.argmax(logits, dim=1)

    # start from class 1 (LV) to skip background
    start_class = 1 if ignore_background else 0
    scores = []

    for class_idx in range(start_class, num_classes):
        # binary mask: 1 where this class appears, 0 elsewhere
        pred_c = (pred == class_idx).float()
        target_c = (target == class_idx).float()

        # pixels where prediction and ground truth both say this class
        intersection = (pred_c * target_c).sum()
        denominator = pred_c.sum() + target_c.sum()

        # dice formula
        dice = _safe_divide(2.0 * intersection, denominator)
        scores.append(dice)

    if not scores:
        return 0.0

    # mean dice across all foreground classes, returned as plain float
    return torch.mean(torch.stack(scores)).item()


def multiclass_iou_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> float:
    # calculates average IoU (Jaccard Index) across all foreground classes
    # IoU = overlap / (predicted + true - overlap)
    # similar to Dice but penalises false positives more heavily
    # 1.0 = perfect, 0.0 = no overlap

    pred = torch.argmax(logits, dim=1)

    start_class = 1 if ignore_background else 0
    scores = []

    for class_idx in range(start_class, num_classes):
        pred_c = (pred == class_idx).float()
        target_c = (target == class_idx).float()

        intersection = (pred_c * target_c).sum()
        # union = everything predicted OR truly this class
        union = pred_c.sum() + target_c.sum() - intersection

        iou = _safe_divide(intersection, union)
        scores.append(iou)

    if not scores:
        return 0.0

    return torch.mean(torch.stack(scores)).item()


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> torch.Tensor:
    # dice loss = 1 - dice score
    # uses soft probabilities not hard predictions so gradients can flow during training
    # 0.0 = perfect, 1.0 = completely wrong

    # softmax converts raw logits to class probabilities
    probs = torch.softmax(logits, dim=1)

    # convert integer labels to one-hot e.g. label 2 -> [0,0,1,0,0]
    # permute rearranges from [B,H,W,C] to [B,C,H,W] to match logits shape
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    start_class = 1 if ignore_background else 0
    losses = []

    for class_idx in range(start_class, num_classes):
        prob_c = probs[:, class_idx]             # predicted probability for this class
        target_c = target_one_hot[:, class_idx]  # ground truth for this class

        # soft intersection: predicted probability x ground truth
        intersection = (prob_c * target_c).sum()
        denominator = prob_c.sum() + target_c.sum()

        dice = _safe_divide(2.0 * intersection, denominator)
        losses.append(1.0 - dice)  # loss = 1 - dice

    if not losses:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return torch.mean(torch.stack(losses))


class DiceCrossEntropyLoss(nn.Module):
    # combined loss: Dice loss + Cross-Entropy loss
    # Dice handles class imbalance (most pixels are background)
    # Cross-Entropy good at per-pixel accuracy
    # using both gives more stable training than either alone
    # default weighting is 50/50

    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,   # how much to weight dice loss
        ce_weight: float = 0.5,     # how much to weight cross-entropy loss
        ignore_background: bool = True,
    ):
        super().__init__()  # initialise parent nn.Module
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_background = ignore_background
        self.ce = nn.CrossEntropyLoss()  # pytorch built-in cross-entropy

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # called automatically when you do loss(predictions, targets)
        ce_loss = self.ce(logits, target.long())
        dice_loss = multiclass_dice_loss(
            logits,
            target,
            num_classes=self.num_classes,
            ignore_background=self.ignore_background,
        )
        # weighted sum of both losses
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss




# HAUSDORFF DISTANCE (HD95)

def hd95_binary(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    # 95th percentile Hausdorff Distance between two binary masks
    # measures how far apart the boundaries of predicted and true masks are
    # lower = better, units are pixels

    # how it works:
    # 1. find the surface (boundary) of each mask using erosion
    # 2. compute distance from every surface point in one mask to nearest in the other
    # 3. take 95th percentile (ignores worst 5%)

    if not SCIPY_AVAILABLE:
        return float("nan")  # cant compute without scipy

    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    # both empty = distance is 0
    if not pred_mask.any() and not true_mask.any():
        return 0.0

    # one empty, one not = complete miss
    if pred_mask.any() != true_mask.any():
        return float("inf")

    # XOR with eroded version gives just the boundary pixels
    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    true_surface = true_mask ^ binary_erosion(true_mask)

    # for every pixel, how far is the nearest surface point
    dt_true = distance_transform_edt(~true_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    # distances from predicted surface to nearest true surface point
    pred_to_true = dt_true[pred_surface]
    # distances from true surface to nearest predicted surface point
    true_to_pred = dt_pred[true_surface]

    all_distances = np.concatenate([pred_to_true, true_to_pred])

    if all_distances.size == 0:
        return 0.0

    # 95th percentile ignores the worst 5% of boundary errors
    return float(np.percentile(all_distances, 95))


def multiclass_hd95(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int,
    ignore_background: bool = True,
) -> Dict[int, float]:
    # computes HD95 separately for each foreground class (LV, RV, LA, RA)
    # returns dict mapping class index to HD95 value
    # e.g. {1: 17.13, 2: 5.55, 3: 6.23, 4: 3.49}
    start_class = 1 if ignore_background else 0
    scores = {}

    for class_idx in range(start_class, num_classes):
        pred_c = (pred_mask == class_idx)
        true_c = (true_mask == class_idx)
        scores[class_idx] = hd95_binary(pred_c, true_c)

    return scores


def mean_hd95_from_dicts(hd95_dicts: List[Dict[int, float]]) -> Dict[int, float]:
    # takes list of per-sample HD95 dicts and averages them per class
    # ignores inf and nan values when averaging
    class_scores: Dict[int, List[float]] = {}

    for row in hd95_dicts:
        for class_idx, value in row.items():
            if np.isfinite(value):  # skip inf and nan
                class_scores.setdefault(class_idx, []).append(value)

    # mean per class, or nan if no valid values
    return {
        class_idx: float(np.mean(values)) if values else float("nan")
        for class_idx, values in class_scores.items()
    }




# VOLUMETRIC QUANTIFICATION

def compute_chamber_volumes_ml(
    mask_volume: np.ndarray,
    pixel_spacing: Tuple[float, float],
    slice_thickness: float,
    num_classes: int,
) -> Dict[int, float]:
    # converts 3D segmentation mask into chamber volumes in millilitres
    # mask_volume shape: [D, H, W], each voxel has a class label
    # pixel_spacing: physical size of each pixel in mm
    # slice_thickness: physical thickness of each CT slice in mm

    # method:
    # 1. work out volume of a single voxel in mm3
    # 2. count voxels belonging to each class
    # 3. multiply to get total volume in mm3
    # 4. divide by 1000 to convert to mL (1 mL = 1000 mm3)

    voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

    volumes = {}
    for class_idx in range(1, num_classes):  # skip background (class 0)
        voxel_count = np.sum(mask_volume == class_idx)
        volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0
        volumes[class_idx] = float(volume_ml)

    return volumes




# BLAND-ALTMAN ANALYSIS

def bland_altman_stats(pred_values: List[float], ref_values: List[float]) -> Dict[str, float]:
    # computes bland-altman agreement stats between predicted and reference volumes
    # used to assess whether automated measurements agree with reference standard

    # calculates:
    # bias: mean difference (pred - ref), positive = model over-estimates
    # sd: standard deviation of the differences
    # loa_lower/upper: 95% limits of agreement (bias +/- 1.96 * sd)
    #   range within which 95% of differences will fall

    pred = np.asarray(pred_values, dtype=np.float64)
    ref = np.asarray(ref_values, dtype=np.float64)

    if pred.shape != ref.shape:
        raise ValueError("Predicted and reference arrays must have the same shape.")

    diff = pred - ref               # difference for each patient
    mean_vals = (pred + ref) / 2.0  # mean of both measurements (x axis of BA plot)

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0  # ddof=1 = sample SD
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd

    return {
        "bias": bias,
        "sd": sd,
        "loa_lower": float(loa_lower),
        "loa_upper": float(loa_upper),
        "n": int(len(diff)),
        "mean_of_means": float(np.mean(mean_vals)) if len(mean_vals) > 0 else float("nan"),
    }




# FIGURE SAVING

def save_bland_altman_plot(pred_values: List[float], ref_values: List[float], save_path: str, title: str) -> None:
    # creates and saves a bland-altman plot
    # x axis: mean of predicted and reference volumes per patient
    # y axis: difference between predicted and reference
    # dashed line = bias, dotted lines = 95% limits of agreement

    pred = np.asarray(pred_values, dtype=np.float64)
    ref = np.asarray(ref_values, dtype=np.float64)

    means = (pred + ref) / 2.0  # x axis
    diffs = pred - ref           # y axis

    stats = bland_altman_stats(pred_values, ref_values)

    plt.figure(figsize=(6, 5))
    plt.scatter(means, diffs, alpha=0.8)           # one dot per patient
    plt.axhline(stats["bias"], linestyle="--")     # bias line
    plt.axhline(stats["loa_lower"], linestyle=":") # lower limit of agreement
    plt.axhline(stats["loa_upper"], linestyle=":") # upper limit of agreement
    plt.xlabel("Mean of predicted and reference volume (mL)")
    plt.ylabel("Difference (pred - ref) (mL)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()  # free memory


def save_segmentation_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    save_path: str,
    alpha: float = 0.35,
) -> None:
    # saves side-by-side: raw CT | ground truth overlay | predicted overlay
    # LV=red, RV=green, LA=blue, RA=yellow
    # alpha=0.35 means CT shows through at 65% opacity

    image = np.clip(image, 0.0, 1.0)

    # empty RGB arrays for overlays
    pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    true_rgb = np.zeros((*true_mask.shape, 3), dtype=np.float32)

    # RGB colour per class
    color_map = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # LV = red
        2: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # RV = green
        3: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # LA = blue
        4: np.array([1.0, 1.0, 0.0], dtype=np.float32),  # RA = yellow
    }

    # fill in colour wherever each class appears
    for class_idx, color in color_map.items():
        pred_rgb[pred_mask == class_idx] = color
        true_rgb[true_mask == class_idx] = color

    # convert greyscale to RGB by stacking 3 identical channels
    base = np.stack([image, image, image], axis=-1)

    # blend CT image with colour overlay
    pred_overlay = (1 - alpha) * base + alpha * pred_rgb
    true_overlay = (1 - alpha) * base + alpha * true_rgb

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")           # raw CT
    axes[0].set_title("CT slice")
    axes[1].imshow(np.clip(true_overlay, 0, 1))  # ground truth overlay
    axes[1].set_title("Reference mask")
    axes[2].imshow(np.clip(pred_overlay, 0, 1))  # predicted overlay
    axes[2].set_title("Predicted mask")

    for ax in axes:
        ax.axis("off")  # remove axis ticks

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_json(data: dict, save_path: str) -> None:
    # saves python dict to JSON file
    # indent=2 makes it human readable
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv_rows(rows: List[dict], save_path: str) -> None:
    # saves list of dicts to CSV
    # each dict = one row, keys = column headers
    if not rows:
        return  # dont create empty file

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()    # column names on first line
        writer.writerows(rows)  # all data rows
