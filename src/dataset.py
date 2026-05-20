from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class DummyCardiacCTDataset(Dataset):
    """
    Synthetic 2.5D cardiac CT dataset for pipeline validation.

    Output:
        image: Tensor [C, H, W] where C = num_input_slices
        mask:  Tensor [H, W] with class labels:
               0 = background
               1 = left ventricle
               2 = right ventricle
               3 = left atrium
               4 = right atrium
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        num_samples: int = 20,
        num_input_slices: int = 5,
        num_classes: int = 5,
    ):
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd for centred 2.5D input.")

        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.num_samples = num_samples
        self.num_input_slices = num_input_slices
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def _make_ellipse_mask(self, yy, xx, center_x, center_y, radius_x, radius_y):
        return (
            ((xx - center_x) ** 2) / (radius_x ** 2)
            + ((yy - center_y) ** 2) / (radius_y ** 2)
            <= 1
        )

    def __getitem__(self, idx: int):
        h = w = self.image_size
        yy, xx = np.ogrid[:h, :w]

        # Create base multi-class mask
        mask = np.zeros((h, w), dtype=np.int64)

        # Approximate chamber positions
        cx = w // 2
        cy = h // 2

        jitter_x = np.random.randint(-10, 10)
        jitter_y = np.random.randint(-10, 10)

        # LV
        lv = self._make_ellipse_mask(
            yy, xx,
            cx - 25 + jitter_x, cy + 20 + jitter_y,
            np.random.randint(18, 28), np.random.randint(22, 32)
        )

        # RV
        rv = self._make_ellipse_mask(
            yy, xx,
            cx + 20 + jitter_x, cy + 20 + jitter_y,
            np.random.randint(18, 28), np.random.randint(20, 30)
        )

        # LA
        la = self._make_ellipse_mask(
            yy, xx,
            cx - 20 + jitter_x, cy - 25 + jitter_y,
            np.random.randint(16, 24), np.random.randint(14, 22)
        )

        # RA
        ra = self._make_ellipse_mask(
            yy, xx,
            cx + 20 + jitter_x, cy - 25 + jitter_y,
            np.random.randint(16, 24), np.random.randint(14, 22)
        )

        # Apply labels with simple overwrite order
        mask[lv] = 1
        mask[rv] = 2
        mask[la] = 3
        mask[ra] = 4

        # Build synthetic 2.5D stack
        image_stack = []
        for slice_idx in range(self.num_input_slices):
            noise = np.random.normal(loc=0.08, scale=0.03, size=(h, w)).astype(np.float32)

            # Slight variation across neighbouring slices
            slice_image = noise.copy()

            slice_image += (mask == 1).astype(np.float32) * (0.45 + np.random.uniform(-0.03, 0.03))
            slice_image += (mask == 2).astype(np.float32) * (0.38 + np.random.uniform(-0.03, 0.03))
            slice_image += (mask == 3).astype(np.float32) * (0.52 + np.random.uniform(-0.03, 0.03))
            slice_image += (mask == 4).astype(np.float32) * (0.48 + np.random.uniform(-0.03, 0.03))

            # Mild per-slice intensity drift to imitate inter-slice variation
            slice_image += np.random.uniform(-0.02, 0.02)

            slice_image = np.clip(slice_image, 0.0, 1.0)
            image_stack.append(slice_image)

        image = np.stack(image_stack, axis=0).astype(np.float32)  # [C, H, W]

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "patient_id": f"dummy_{idx:03d}",
            "slice_idx": self.num_input_slices // 2,
            "pixel_spacing": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "slice_thickness": torch.tensor(1.0, dtype=torch.float32),
        }