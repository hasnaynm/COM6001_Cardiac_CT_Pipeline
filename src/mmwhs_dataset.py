from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from PIL import Image


MMWHS_LABEL_MAP = {
    0: 0,
    500: 1,
    600: 2,
    420: 3,
    550: 4,
    205: 0,
    820: 0,
    850: 0,
}


def remap_labels(label_array):
    output = np.zeros_like(label_array, dtype=np.int64)
    for original_val, new_val in MMWHS_LABEL_MAP.items():
        output[label_array == original_val] = new_val
    return output


def resize_volume(vol, img_size):
    """Resize each slice of a volume to img_size x img_size."""
    d = vol.shape[2]
    resized = np.zeros((img_size, img_size, d), dtype=np.float32)
    for z in range(d):
        s = vol[:, :, z].astype(np.float32)
        pil = Image.fromarray(s)
        pil = pil.resize((img_size, img_size), Image.BILINEAR)
        resized[:, :, z] = np.array(pil)
    return resized


class MMWHSDataset(Dataset):
    def __init__(self, data_dir, num_input_slices=5, hu_min=-150.0, hu_max=250.0, img_size=128):
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd.")
        self.data_dir = Path(data_dir)
        self.num_input_slices = num_input_slices
        self.hu_min = hu_min
        self.hu_max = hu_max
        self.img_size = img_size
        self.half = num_input_slices // 2
        self.volumes = []
        self.samples = []
        self._load_all_volumes()

    def _load_all_volumes(self):
        image_paths = sorted(self.data_dir.glob("*_image.nii*"))
        for img_path in image_paths:
            label_path = Path(str(img_path).replace("_image.nii", "_label.nii"))
            if not label_path.exists():
                print(f"Warning: no label found for {img_path.name}, skipping.")
                continue
            print(f"Loading {img_path.name}...")

            img_nib = nib.load(str(img_path))
            img_vol = img_nib.get_fdata().astype(np.float32)
            img_vol = np.clip(img_vol, self.hu_min, self.hu_max)
            img_vol = (img_vol - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)
            img_vol = resize_volume(img_vol, self.img_size)

            lbl_nib = nib.load(str(label_path))
            lbl_vol = lbl_nib.get_fdata().astype(np.float32)
            lbl_vol = remap_labels(lbl_vol.astype(np.int64)).astype(np.float32)
            lbl_vol = resize_volume(lbl_vol, self.img_size)

            pixdim = img_nib.header.get_zooms()
            patient_id = img_path.stem.replace("_image", "")
            vol_idx = len(self.volumes)
            self.volumes.append((img_vol, lbl_vol, pixdim))

            depth = img_vol.shape[2]
            for z in range(self.half, depth - self.half):
                self.samples.append({
                    "vol_idx": vol_idx,
                    "slice_idx": z,
                    "patient_id": patient_id,
                })

        print(f"Loaded {len(self.volumes)} volumes, {len(self.samples)} slices total.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        vol_idx = sample["vol_idx"]
        z = sample["slice_idx"]
        img_vol, lbl_vol, pixdim = self.volumes[vol_idx]

        slices = []
        for offset in range(-self.half, self.half + 1):
            s = img_vol[:, :, z + offset]
            slices.append(s)
        image = np.stack(slices, axis=0).astype(np.float32)

        label = lbl_vol[:, :, z].astype(np.int64)

        pixel_spacing = torch.tensor([float(pixdim[0]), float(pixdim[1])], dtype=torch.float32)
        slice_thickness = torch.tensor(float(pixdim[2]), dtype=torch.float32)

        return {
            "image": torch.FloatTensor(image),
            "mask": torch.LongTensor(label),
            "patient_id": sample["patient_id"],
            "slice_idx": z,
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
        }
