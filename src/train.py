import os
import random
from typing import Dict
from src.mmwhs_dataset import MMWHSDataset

import torch
from torch.utils.data import DataLoader, Subset, Dataset

from src.model import UNetDeep
from src.utils import (
    load_config,
    ensure_dir,
    append_log,
    multiclass_dice_score,
    multiclass_iou_score,
    DiceCrossEntropyLoss,
)

NUM_CLASSES = 5
NUM_INPUT_SLICES = 5


class SimpleDummyCardiacCTDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 100,
        img_size: int = 256,
        num_input_slices: int = 5,
        num_classes: int = 5,
    ):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_input_slices = num_input_slices
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        image = torch.rand(self.num_input_slices, self.img_size, self.img_size)
        mask = torch.randint(
            low=0,
            high=self.num_classes,
            size=(self.img_size, self.img_size),
            dtype=torch.long,
        )
        return {
            "image": image,
            "mask": mask,
            "patient_id": f"dummy_{idx // 10}",
            "slice_idx": idx,
            "pixel_spacing": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "slice_thickness": torch.tensor(1.0, dtype=torch.float32),
        }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()
            running_dice += multiclass_dice_score(outputs, masks, NUM_CLASSES)
            running_iou += multiclass_iou_score(outputs, masks, NUM_CLASSES)
            num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}

    return {
        "loss": running_loss / num_batches,
        "dice": running_dice / num_batches,
        "iou": running_iou / num_batches,
    }


def train() -> None:
    config = load_config()

    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = config.get("output_dir", "data/outputs")
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "training_log.txt")
    model_path = os.path.join(output_dir, "model.pt")
    best_model_path = os.path.join(output_dir, "best_unet_deep.pt")

    batch_size = config.get("batch_size", 2)
    num_epochs = config.get("num_epochs", 50)
    learning_rate = config.get("learning_rate", 1e-3)

    use_dummy_data = False

    if use_dummy_data:
        dataset = SimpleDummyCardiacCTDataset(
            num_samples=config.get("num_samples", 100),
            img_size=config.get("img_size", 192),
            num_input_slices=config.get("num_input_slices", NUM_INPUT_SLICES),
            num_classes=config.get("num_classes", NUM_CLASSES),
        )

        total_len = len(dataset)
        train_end = int(0.7 * total_len)
        val_end = int(0.85 * total_len)

        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, total_len))

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    else:
        dataset = MMWHSDataset(
            data_dir=config["data_dir"],
            num_input_slices=config.get("num_input_slices", NUM_INPUT_SLICES),
            img_size=config.get("img_size", 192),
            hu_min=config.get("hu_window", [-150, 250])[0],
            hu_max=config.get("hu_window", [-150, 250])[1],
        )

        total = len(dataset)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)

        train_indices = list(range(0, train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, total))

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)

        print(f"Total slices: {total}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = UNetDeep(
        in_channels=config.get("num_input_slices", NUM_INPUT_SLICES),
        out_channels=config.get("num_classes", NUM_CLASSES),
    ).to(device)

    criterion = DiceCrossEntropyLoss(
        num_classes=config.get("num_classes", NUM_CLASSES)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler - reduces LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stopping_patience = config.get("early_stopping_patience", 10)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = (
            running_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        )

        val_metrics = evaluate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]['lr']

        log_line = (
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        print(log_line)
        append_log(log_path, log_line)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for test evaluation.")

    test_metrics = evaluate(model, test_loader, criterion, device)

    final_line = (
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Dice: {test_metrics['dice']:.4f} | "
        f"Test IoU: {test_metrics['iou']:.4f}"
    )

    print(final_line)
    append_log(log_path, final_line)

    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")