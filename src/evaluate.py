import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset import DummyCardiacCTDataset
from src.ct_dataset import CardiacCTDataset
from src.model import UNetSmall
from src.utils import (
    load_config,
    ensure_dir,
    multiclass_dice_score,
    multiclass_iou_score,
    multiclass_hd95,
    mean_hd95_from_dicts,
    split_dataset,
    compute_chamber_volumes_ml,
    bland_altman_stats,
    save_bland_altman_plot,
    save_segmentation_overlay,
    save_json,
    save_csv_rows,
)


CLASS_NAMES = {
    1: "LV",
    2: "RV",
    3: "LA",
    4: "RA",
}


def get_dataset(config: dict):
    dataset_mode = config.get("dataset_mode", "dummy")
    data_dir = config["data_dir"]
    image_size = config["image_size"]
    num_input_slices = config["num_input_slices"]
    num_classes = config["num_classes"]

    if dataset_mode == "dummy":
        return DummyCardiacCTDataset(
            data_dir=data_dir,
            num_samples=config.get("num_samples", 20),
            image_size=image_size,
            num_input_slices=num_input_slices,
            num_classes=num_classes,
        )

    if dataset_mode == "real":
        return CardiacCTDataset(
            data_dir=data_dir,
            img_size=image_size,
            num_input_slices=num_input_slices,
            hu_window=tuple(config.get("hu_window", [-150, 250])),
        )

    raise ValueError(f"Invalid dataset mode: {dataset_mode}")


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def evaluate():
    config = load_config()

    output_dir = config["output_dir"]
    eval_dir = os.path.join(output_dir, "evaluation")
    overlay_dir = os.path.join(eval_dir, "overlays")
    ba_dir = os.path.join(eval_dir, "bland_altman")

    ensure_dir(eval_dir)
    ensure_dir(overlay_dir)
    ensure_dir(ba_dir)

    device = torch.device(config["device"])
    num_classes = config["num_classes"]
    num_input_slices = config["num_input_slices"]

    dataset = get_dataset(config)
    _, _, test_dataset = split_dataset(
        dataset=dataset,
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        seed=42,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
    )

    model = UNetSmall(
        in_channels=num_input_slices,
        out_channels=num_classes,
    ).to(device)

    best_model_path = os.path.join(output_dir, "best_unet_small.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found: {best_model_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    total_dice = 0.0
    total_iou = 0.0
    hd95_rows = []

    overlay_limit = config.get("save_overlay_count", 10)
    overlay_counter = 0

    patient_slices = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            batch_gpu = move_batch_to_device(batch, device)

            images = batch_gpu["image"]
            masks = batch_gpu["mask"]

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            total_dice += multiclass_dice_score(logits, masks, num_classes=num_classes)
            total_iou += multiclass_iou_score(logits, masks, num_classes=num_classes)

            preds_np = preds.cpu().numpy()
            masks_np = masks.cpu().numpy()
            images_np = batch["image"].cpu().numpy()

            patient_ids = batch["patient_id"]
            slice_indices = batch["slice_idx"].cpu().numpy()
            pixel_spacings = batch["pixel_spacing"].cpu().numpy()
            slice_thicknesses = batch["slice_thickness"].cpu().numpy()

            for i in range(len(patient_ids)):
                pred_mask = preds_np[i]
                true_mask = masks_np[i]
                centre_image = images_np[i, images_np.shape[1] // 2]

                hd95_dict = multiclass_hd95(
                    pred_mask=pred_mask,
                    true_mask=true_mask,
                    num_classes=num_classes,
                )
                hd95_rows.append(hd95_dict)

                patient_slices[patient_ids[i]].append({
                    "slice_idx": int(slice_indices[i]),
                    "pred_mask": pred_mask,
                    "true_mask": true_mask,
                    "pixel_spacing": tuple(float(x) for x in pixel_spacings[i]),
                    "slice_thickness": float(slice_thicknesses[i]),
                    "image": centre_image,
                })

                if overlay_counter < overlay_limit:
                    save_path = os.path.join(
                        overlay_dir,
                        f"{patient_ids[i]}_slice_{int(slice_indices[i]):04d}.png"
                    )
                    save_segmentation_overlay(
                        image=centre_image,
                        pred_mask=pred_mask,
                        true_mask=true_mask,
                        save_path=save_path,
                    )
                    overlay_counter += 1

    mean_dice = total_dice / len(test_loader)
    mean_iou = total_iou / len(test_loader)
    mean_hd95 = mean_hd95_from_dicts(hd95_rows)

    patient_volume_rows = []
    bland_altman_summary = {}

    per_class_pred = defaultdict(list)
    per_class_ref = defaultdict(list)

    for patient_id, slices in patient_slices.items():
        slices = sorted(slices, key=lambda x: x["slice_idx"])

        pred_volume = np.stack([s["pred_mask"] for s in slices], axis=0)
        true_volume = np.stack([s["true_mask"] for s in slices], axis=0)

        pixel_spacing = slices[0]["pixel_spacing"]
        slice_thickness = slices[0]["slice_thickness"]

        pred_volumes = compute_chamber_volumes_ml(
            mask_volume=pred_volume,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            num_classes=num_classes,
        )
        true_volumes = compute_chamber_volumes_ml(
            mask_volume=true_volume,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
            num_classes=num_classes,
        )

        row = {"patient_id": patient_id}

        for class_idx in range(1, num_classes):
            class_name = CLASS_NAMES.get(class_idx, f"class_{class_idx}")
            pred_key = f"{class_name}_pred_ml"
            ref_key = f"{class_name}_ref_ml"

            row[pred_key] = pred_volumes[class_idx]
            row[ref_key] = true_volumes[class_idx]

            per_class_pred[class_idx].append(pred_volumes[class_idx])
            per_class_ref[class_idx].append(true_volumes[class_idx])

        patient_volume_rows.append(row)

    for class_idx in range(1, num_classes):
        class_name = CLASS_NAMES.get(class_idx, f"class_{class_idx}")
        stats = bland_altman_stats(
            pred_values=per_class_pred[class_idx],
            ref_values=per_class_ref[class_idx],
        )
        bland_altman_summary[class_name] = stats

        plot_path = os.path.join(ba_dir, f"{class_name}_bland_altman.png")
        save_bland_altman_plot(
            pred_values=per_class_pred[class_idx],
            ref_values=per_class_ref[class_idx],
            save_path=plot_path,
            title=f"Bland–Altman: {class_name} volume",
        )

    metrics_summary = {
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "mean_hd95_per_class": {
            CLASS_NAMES.get(k, str(k)): v for k, v in mean_hd95.items()
        },
        "bland_altman": bland_altman_summary,
        "num_test_samples": len(test_dataset),
        "num_test_patients": len(patient_slices),
    }

    save_json(metrics_summary, os.path.join(eval_dir, "metrics_summary.json"))
    save_csv_rows(patient_volume_rows, os.path.join(eval_dir, "patient_volumes.csv"))

    print("Evaluation complete.")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print("Mean HD95 per class:")
    for class_name, value in metrics_summary["mean_hd95_per_class"].items():
        print(f"  {class_name}: {value:.4f}" if np.isfinite(value) else f"  {class_name}: nan")

    print("\nBland–Altman summary:")
    for class_name, stats in bland_altman_summary.items():
        print(
            f"  {class_name}: bias={stats['bias']:.4f}, "
            f"LoA=({stats['loa_lower']:.4f}, {stats['loa_upper']:.4f}), n={stats['n']}"
        )


if __name__ == "__main__":
    evaluate()