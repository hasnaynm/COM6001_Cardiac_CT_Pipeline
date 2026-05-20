from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import csv
import json

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def append_log(log_path: str, message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (numerator + eps) / (denominator + eps)


def split_dataset(dataset, train_ratio: float, val_ratio: float, test_ratio: float, seed: int = 42):
    total = len(dataset)

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    test_size = total - train_size - val_size

    if train_size <= 0 or val_size <= 0 or test_size <= 0:
        raise ValueError("Dataset too small for requested train/val/test split.")

    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )


def multiclass_dice_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> float:
    pred = torch.argmax(logits, dim=1)

    start_class = 1 if ignore_background else 0
    scores = []

    for class_idx in range(start_class, num_classes):
        pred_c = (pred == class_idx).float()
        target_c = (target == class_idx).float()

        intersection = (pred_c * target_c).sum()
        denominator = pred_c.sum() + target_c.sum()
        dice = _safe_divide(2.0 * intersection, denominator)
        scores.append(dice)

    if not scores:
        return 0.0

    return torch.mean(torch.stack(scores)).item()


def multiclass_iou_score(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_background: bool = True,
) -> float:
    pred = torch.argmax(logits, dim=1)

    start_class = 1 if ignore_background else 0
    scores = []

    for class_idx in range(start_class, num_classes):
        pred_c = (pred == class_idx).float()
        target_c = (target == class_idx).float()

        intersection = (pred_c * target_c).sum()
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
    probs = torch.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

    start_class = 1 if ignore_background else 0
    losses = []

    for class_idx in range(start_class, num_classes):
        prob_c = probs[:, class_idx]
        target_c = target_one_hot[:, class_idx]

        intersection = (prob_c * target_c).sum()
        denominator = prob_c.sum() + target_c.sum()

        dice = _safe_divide(2.0 * intersection, denominator)
        losses.append(1.0 - dice)

    if not losses:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return torch.mean(torch.stack(losses))


class DiceCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        ignore_background: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.ignore_background = ignore_background
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target.long())
        dice_loss = multiclass_dice_loss(
            logits,
            target,
            num_classes=self.num_classes,
            ignore_background=self.ignore_background,
        )
        return self.dice_weight * dice_loss + self.ce_weight * ce_loss


def hd95_binary(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    95th percentile Hausdorff Distance for binary masks.
    Returns nan if scipy is unavailable.
    """
    if not SCIPY_AVAILABLE:
        return float("nan")

    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)

    if not pred_mask.any() and not true_mask.any():
        return 0.0
    if pred_mask.any() != true_mask.any():
        return float("inf")

    pred_surface = pred_mask ^ binary_erosion(pred_mask)
    true_surface = true_mask ^ binary_erosion(true_mask)

    dt_true = distance_transform_edt(~true_surface)
    dt_pred = distance_transform_edt(~pred_surface)

    pred_to_true = dt_true[pred_surface]
    true_to_pred = dt_pred[true_surface]

    all_distances = np.concatenate([pred_to_true, true_to_pred])

    if all_distances.size == 0:
        return 0.0

    return float(np.percentile(all_distances, 95))


def multiclass_hd95(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    num_classes: int,
    ignore_background: bool = True,
) -> Dict[int, float]:
    start_class = 1 if ignore_background else 0
    scores = {}

    for class_idx in range(start_class, num_classes):
        pred_c = (pred_mask == class_idx)
        true_c = (true_mask == class_idx)
        scores[class_idx] = hd95_binary(pred_c, true_c)

    return scores


def mean_hd95_from_dicts(hd95_dicts: List[Dict[int, float]]) -> Dict[int, float]:
    class_scores: Dict[int, List[float]] = {}

    for row in hd95_dicts:
        for class_idx, value in row.items():
            if np.isfinite(value):
                class_scores.setdefault(class_idx, []).append(value)

    return {
        class_idx: float(np.mean(values)) if values else float("nan")
        for class_idx, values in class_scores.items()
    }


def compute_chamber_volumes_ml(
    mask_volume: np.ndarray,
    pixel_spacing: Tuple[float, float],
    slice_thickness: float,
    num_classes: int,
) -> Dict[int, float]:
    """
    mask_volume shape: [D, H, W]
    pixel_spacing in mm, slice_thickness in mm
    Returns chamber volumes in mL.
    """
    voxel_volume_mm3 = pixel_spacing[0] * pixel_spacing[1] * slice_thickness

    volumes = {}
    for class_idx in range(1, num_classes):
        voxel_count = np.sum(mask_volume == class_idx)
        volume_ml = (voxel_count * voxel_volume_mm3) / 1000.0
        volumes[class_idx] = float(volume_ml)

    return volumes


def bland_altman_stats(pred_values: List[float], ref_values: List[float]) -> Dict[str, float]:
    pred = np.asarray(pred_values, dtype=np.float64)
    ref = np.asarray(ref_values, dtype=np.float64)

    if pred.shape != ref.shape:
        raise ValueError("Predicted and reference arrays must have the same shape.")

    diff = pred - ref
    mean_vals = (pred + ref) / 2.0

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
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


def save_bland_altman_plot(pred_values: List[float], ref_values: List[float], save_path: str, title: str) -> None:
    pred = np.asarray(pred_values, dtype=np.float64)
    ref = np.asarray(ref_values, dtype=np.float64)

    means = (pred + ref) / 2.0
    diffs = pred - ref

    stats = bland_altman_stats(pred_values, ref_values)

    plt.figure(figsize=(6, 5))
    plt.scatter(means, diffs, alpha=0.8)
    plt.axhline(stats["bias"], linestyle="--")
    plt.axhline(stats["loa_lower"], linestyle=":")
    plt.axhline(stats["loa_upper"], linestyle=":")
    plt.xlabel("Mean of predicted and reference volume (mL)")
    plt.ylabel("Difference (pred - ref) (mL)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_segmentation_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    save_path: str,
    alpha: float = 0.35,
) -> None:
    """
    image: [H, W], assumed normalised 0..1
    pred_mask / true_mask: [H, W] integer labels
    """
    image = np.clip(image, 0.0, 1.0)

    pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.float32)
    true_rgb = np.zeros((*true_mask.shape, 3), dtype=np.float32)

    color_map = {
        1: np.array([1.0, 0.0, 0.0], dtype=np.float32),  # LV
        2: np.array([0.0, 1.0, 0.0], dtype=np.float32),  # RV
        3: np.array([0.0, 0.0, 1.0], dtype=np.float32),  # LA
        4: np.array([1.0, 1.0, 0.0], dtype=np.float32),  # RA
    }

    for class_idx, color in color_map.items():
        pred_rgb[pred_mask == class_idx] = color
        true_rgb[true_mask == class_idx] = color

    base = np.stack([image, image, image], axis=-1)

    pred_overlay = (1 - alpha) * base + alpha * pred_rgb
    true_overlay = (1 - alpha) * base + alpha * true_rgb

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT slice")
    axes[1].imshow(np.clip(true_overlay, 0, 1))
    axes[1].set_title("Reference mask")
    axes[2].imshow(np.clip(pred_overlay, 0, 1))
    axes[2].set_title("Predicted mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_json(data: dict, save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv_rows(rows: List[dict], save_path: str) -> None:
    if not rows:
        return

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)