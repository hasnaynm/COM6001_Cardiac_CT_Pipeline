# train.py
# main training script for the UNetDeep model
# handles data loading, training loop, validation, early stopping, saving best model

import os
import random
from typing import Dict
from src.mmwhs_dataset import MMWHSDataset  # real MM-WHS dataset loader

import torch
from torch.utils.data import DataLoader, Subset, Dataset




from src.model import UNetDeep  # the actual model architecture
from src.utils import (
    load_config,           # reads config.yaml
    ensure_dir,            # creates folders if they dont exist
    append_log,            # writes to training log file
    multiclass_dice_score, # dice metric
    multiclass_iou_score,  # iou metric
    DiceCrossEntropyLoss,  # combined loss function
)

# global constants - 5 classes (background + LV, RV, LA, RA), 5 input slices for 2.5D
NUM_CLASSES = 5
NUM_INPUT_SLICES = 5





class SimpleDummyCardiacCTDataset(Dataset):
    # synthetic dataset used early on to test the pipeline before real data was ready
    # generates random tensors as images and random integer labels as masks
    # not used in final training (use_dummy_data = False below)


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
        return self.num_samples  # how many samples in the dataset



    def __getitem__(self, idx: int):
        # random image tensor shape: [slices, H, W]
        image = torch.rand(self.num_input_slices, self.img_size, self.img_size)
        # random mask with integer class labels 0-4
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
            "pixel_spacing": torch.tensor([1.0, 1.0], dtype=torch.float32),  # fake 1mm spacing
            "slice_thickness": torch.tensor(1.0, dtype=torch.float32),        # fake 1mm thickness
        }





def set_seed(seed: int = 42) -> None:
    # fixes random seeds so results are reproducible across runs
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # also seed GPU if available




def evaluate(model, loader, criterion, device) -> Dict[str, float]:
    # runs model on a dataloader and returns average loss, dice, iou
    # used for both validation during training and final test evaluation
    model.eval()  # switch to eval mode (disables dropout, batchnorm behaves differently)

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    with torch.no_grad():  # no gradient calculation needed during evaluation
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            outputs = model(images)           # forward pass
            loss = criterion(outputs, masks)  # compute loss

            running_loss += loss.item()
            running_dice += multiclass_dice_score(outputs, masks, NUM_CLASSES)
            running_iou += multiclass_iou_score(outputs, masks, NUM_CLASSES)
            num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "dice": 0.0, "iou": 0.0}

    # return averages across all batches
    return {
        "loss": running_loss / num_batches,
        "dice": running_dice / num_batches,
        "iou": running_iou / num_batches,
    }




def train() -> None:
    config = load_config()  # load all settings from config.yaml

    seed = config.get("seed", 42)
    set_seed(seed)  # fix seeds for reproducibility

    # use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # set up output paths
    output_dir = config.get("output_dir", "data/outputs")
    ensure_dir(output_dir)
    log_path = os.path.join(output_dir, "training_log.txt")   # where epoch logs go
    model_path = os.path.join(output_dir, "model.pt")          # final model weights
    best_model_path = os.path.join(output_dir, "best_unet_deep.pt")  # best val loss model

    # training hyperparameters from config
    batch_size = config.get("batch_size", 2)
    num_epochs = config.get("num_epochs", 50)
    learning_rate = config.get("learning_rate", 1e-3)

    # toggle between synthetic dummy data and real MM-WHS data
    # set to False for real training
    use_dummy_data = False

    if use_dummy_data:
        # synthetic pipeline test - not used in final model
        dataset = SimpleDummyCardiacCTDataset(
            num_samples=config.get("num_samples", 100),
            img_size=config.get("img_size", 192),
            num_input_slices=config.get("num_input_slices", NUM_INPUT_SLICES),
            num_classes=config.get("num_classes", NUM_CLASSES),
        )

        total_len = len(dataset)
        train_end = int(0.7 * total_len)
        val_end = int(0.85 * total_len)

        # split by index ranges: 70% train, 15% val, 15% test
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
        # real MM-WHS dataset - loads NIfTI volumes and extracts 2.5D slice stacks
        dataset = MMWHSDataset(
            data_dir=config["data_dir"],
            num_input_slices=config.get("num_input_slices", NUM_INPUT_SLICES),
            img_size=config.get("img_size", 192),
            hu_min=config.get("hu_window", [-150, 250])[0],  # HU clip lower bound
            hu_max=config.get("hu_window", [-150, 250])[1],  # HU clip upper bound
        )

        total = len(dataset)
        train_end = int(0.7 * total)
        val_end = int(0.85 * total)

        # 70/15/15 split
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





    # dataloaders - shuffle train data each epoch, dont shuffle val/test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)



    # initialise model and move to GPU/CPU
    model = UNetDeep(
        in_channels=config.get("num_input_slices", NUM_INPUT_SLICES),
        out_channels=config.get("num_classes", NUM_CLASSES),
    ).to(device)




    # combined dice + cross-entropy loss
    criterion = DiceCrossEntropyLoss(num_classes=config.get("num_classes", NUM_CLASSES))



    # adam optimiser - adapts learning rate per parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



    # reduce LR by half if val loss doesnt improve for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',   # minimising loss
        factor=0.5,   # multiply LR by 0.5
        patience=3,   # wait 3 epochs before reducing
    )



    best_val_loss = float("inf")  # track best val loss seen so far
    epochs_no_improve = 0         # counter for early stopping
    early_stopping_patience = config.get("early_stopping_patience", 10)



    # main training loop
    for epoch in range(num_epochs):
        model.train()  # switch back to train mode each epoch
        running_train_loss = 0.0
        num_train_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()         # clear gradients from previous step
            outputs = model(images)       # forward pass
            loss = criterion(outputs, masks)  # compute loss
            loss.backward()               # backprop - compute gradients
            optimizer.step()              # update weights

            running_train_loss += loss.item()
            num_train_batches += 1

        # average train loss for this epoch
        avg_train_loss = running_train_loss / num_train_batches if num_train_batches > 0 else 0.0

        # run validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        # update LR scheduler based on val loss
        scheduler.step(val_metrics["loss"])

        current_lr = optimizer.param_groups[0]['lr']  # get current LR after scheduler step

        # build log string and print + save it
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

        # save model if val loss improved
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)  # save weights only
            print(f"  Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            # stop training if no improvement for early_stopping_patience epochs
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # load best model weights before final test evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model for test evaluation.")

    # run on test set
    test_metrics = evaluate(model, test_loader, criterion, device)

    final_line = (
        f"Test Loss: {test_metrics['loss']:.4f} | "
        f"Test Dice: {test_metrics['dice']:.4f} | "
        f"Test IoU: {test_metrics['iou']:.4f}"
    )




    print(final_line)
    append_log(log_path, final_line)

    # also save final model (may differ from best if training continued after best checkpoint)
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved to {model_path}")
