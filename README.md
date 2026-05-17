# Cardiac CT Chamber Segmentation Pipeline

Deep learning pipeline for automated cardiac chamber segmentation and volumetric quantification from cardiac CT images, developed as part of a BSc (Hons) Computer Science with Artificial Intelligence dissertation at Buckinghamshire New University.

The pipeline implements a 2.5D U-Net architecture (UNetDeep) trained on the [MM-WHS dataset](https://zmiclab.github.io/zxh/0/mmwhs/) to perform multi-class segmentation of the left ventricle (LV), right ventricle (RV), left atrium (LA), and right atrium (RA).

---

## Project Structure

```
cardiac_ct_pipeline/
├── src/
│   ├── model.py            # UNetDeep architecture (4-level encoder-decoder)
│   ├── train.py            # Training loop with early stopping and LR scheduling
│   ├── mmwhs_dataset.py    # MM-WHS NIfTI dataset loader with HU windowing
│   ├── ct_dataset.py       # DICOM dataset loader for real NHS CT scans
│   ├── dataset.py          # Synthetic dummy dataset for pipeline validation
│   ├── evaluate.py         # Evaluation: Dice, IoU, HD95, Bland-Altman, overlays
│   ├── utils.py            # Loss functions, metrics, plotting utilities
│   └── quantify.py         # Volumetric quantification utilities
├── scripts/
│   ├── run_train.py        # Entry point: training
│   ├── run_evaluate.py     # Entry point: evaluation
│   └── run_inference.py    # Entry point: inference on new scans
├── config.yaml             # All hyperparameters and paths
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended (CPU supported but slow for training)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

All settings are controlled via `config.yaml`. Key parameters:

| Parameter | Description |
|---|---|
| `data_dir` | Path to MM-WHS dataset folder |
| `output_dir` | Where models and results are saved |
| `img_size` | Input resolution (192 used in final model) |
| `num_input_slices` | Number of 2.5D context slices (5) |
| `num_classes` | Number of output classes including background (5) |
| `batch_size` | Training batch size |
| `num_epochs` | Maximum training epochs |
| `learning_rate` | Initial learning rate |
| `hu_window` | Hounsfield Unit clipping window [-150, 250] |

---

## Dataset

Training and evaluation use the **Multi-Modality Whole Heart Segmentation (MM-WHS)** dataset (Zhuang et al., 2019), accessed via Kaggle. The dataset is not included in this repository.

Expected folder structure for MM-WHS data:

```
data/
└── mmwhs/
    ├── ct_train_1001_image.nii.gz
    ├── ct_train_1001_label.nii.gz
    ├── ct_train_1002_image.nii.gz
    └── ...
```

Label remapping applied:

| Original value | Structure | Remapped label |
|---|---|---|
| 500 | Left ventricle (LV) | 1 |
| 600 | Right ventricle (RV) | 2 |
| 420 | Left atrium (LA) | 3 |
| 550 | Right atrium (RA) | 4 |
| 0, 205, 820, 850 | Background / other | 0 |

---

## Training

```bash
python scripts/run_train.py
```

The training script:
- Loads MM-WHS NIfTI volumes and extracts 2.5D slice stacks
- Splits data 70% train / 15% val / 15% test
- Trains UNetDeep with combined Dice + Cross-Entropy loss
- Uses ReduceLROnPlateau scheduler and early stopping
- Saves best model to `data/outputs/best_unet_deep.pt`

---

## Evaluation

```bash
python scripts/run_evaluate.py
```

Produces:
- Per-class Dice Similarity Coefficient and IoU
- 95th percentile Hausdorff Distance (HD95) per chamber
- Bland-Altman agreement plots for volumetric quantification
- Segmentation overlay images
- `metrics_summary.json` and `patient_volumes.csv`

**Final model performance (test set):**

| Metric | Overall | LV | RV | LA | RA |
|---|---|---|---|---|---|
| Dice | 0.7972 | 0.5195 | 0.7679 | 0.6164 | 0.7724 |
| HD95 (px) | — | 17.13 | 5.55 | 6.23 | 3.49 |

---

## Inference on NHS DICOM Scans

```bash
python scripts/run_inference.py
```

The inference pipeline:
- Accepts DICOM series (sorted by ImagePositionPatient / SliceLocation)
- Applies HU windowing and 2.5D slice stacking
- Runs the trained UNetDeep model
- Outputs segmentation overlays for qualitative clinical review

NHS scans are not included in this repository. Clinical validation was conducted under NHS DSPT and UK GDPR governance requirements, with data access restricted to the Trust's secure network environment.

---

## Model Architecture

**UNetDeep** — a full-depth 2.5D U-Net with 4 encoder levels:

- Input: 5-slice stack [B, 5, H, W]
- Encoder: 32 → 64 → 128 → 256 → 512 (bottleneck with 0.3 dropout)
- Decoder: transposed convolutions with skip connections
- Output: 5-class segmentation map [B, 5, H, W]

---

## Citation

If referencing the MM-WHS dataset:

> Zhuang, X. et al. (2019) 'Evaluation of algorithms for multi-modality whole heart segmentation: An open-access grand challenge', *Medical Image Analysis*, 58, p.101537.

---

## Author

Hasnayn Mahmood  
BSc (Hons) Computer Science with Artificial Intelligence  
Buckinghamshire New University, 2026  
Supervised by Dr Shahadate Rezvy (Academic) and Dr Tarun Mittal (Clinical, Harefield Hospital)
