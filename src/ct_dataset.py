import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset


class CardiacCTDataset(Dataset):
    """
    Cardiac CT dataset for 2.5D multi-class segmentation.

    Expected folder structure:
        data_dir/
            patient_001/
                images/
                    *.dcm
                masks/
                    *.npy   OR one volume file such as mask_volume.npy
            patient_002/
                images/
                    *.dcm
                masks/
                    *.npy

    Each training sample is a target slice with neighbouring slices stacked
    as channels, e.g. 5 slices -> tensor shape [5, H, W].

    The target mask is the centre slice mask with shape [H, W].
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int = 256,
        num_input_slices: int = 5,
        hu_window: Tuple[int, int] = (-150, 250),
    ):
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd for centred 2.5D stacking.")

        self.data_dir = data_dir
        self.img_size = img_size
        self.num_input_slices = num_input_slices
        self.half_window = num_input_slices // 2
        self.hu_min, self.hu_max = hu_window

        self.patient_dirs = self.find_patient_dirs()
        if not self.patient_dirs:
            raise ValueError(f"No valid patient folders found in '{data_dir}'.")

        # Build an index of all slice-centred training samples
        self.samples = self.build_sample_index()

    def find_patient_dirs(self) -> List[str]:
        patient_dirs = []

        for entry in os.listdir(self.data_dir):
            patient_path = os.path.join(self.data_dir, entry)
            if not os.path.isdir(patient_path):
                continue

            image_dir = os.path.join(patient_path, "images")
            mask_dir = os.path.join(patient_path, "masks")

            if os.path.isdir(image_dir) and os.path.isdir(mask_dir):
                dicom_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".dcm")]
                if dicom_files:
                    patient_dirs.append(patient_path)

        return sorted(patient_dirs)

    def build_sample_index(self) -> List[Dict]:
        samples = []

        for patient_path in self.patient_dirs:
            image_dir = os.path.join(patient_path, "images")
            dicom_paths = self.get_sorted_dicom_paths(image_dir)

            if len(dicom_paths) < self.num_input_slices:
                continue

        patient_id = os.path.basename(patient_path)

        for centre_idx in range(len(dicom_paths)):
            samples.append({
                "patient_path": patient_path,
                "patient_id": patient_id,
                "centre_idx": centre_idx,
                "dicom_paths": dicom_paths,
            })

        if not samples:
            raise ValueError(
                "No usable samples found. Check folder structure and minimum slice count."
        )

        return samples

    def get_sorted_dicom_paths(self, image_dir: str) -> List[str]:
        dicom_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(".dcm")
        ]

        if not dicom_paths:
            raise ValueError(f"No DICOM files found in {image_dir}")

        # Read minimal metadata for sorting
        dicom_meta = []
        for path in dicom_paths:
            dcm = pydicom.dcmread(path, stop_before_pixels=True)

            if hasattr(dcm, "ImagePositionPatient"):
                z_pos = float(dcm.ImagePositionPatient[2])
            elif hasattr(dcm, "SliceLocation"):
                z_pos = float(dcm.SliceLocation)
            elif hasattr(dcm, "InstanceNumber"):
                z_pos = float(dcm.InstanceNumber)
            else:
                raise ValueError(f"Cannot determine slice order for: {path}")

            dicom_meta.append((path, z_pos))

        dicom_meta.sort(key=lambda x: x[1])
        return [path for path, _ in dicom_meta]

    def __len__(self) -> int:
        return len(self.samples)

    def load_dicom_slice(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        dcm = pydicom.dcmread(file_path)

        image = dcm.pixel_array.astype(np.float32)

        # Convert to Hounsfield Units
        slope = float(getattr(dcm, "RescaleSlope", 1.0))
        intercept = float(getattr(dcm, "RescaleIntercept", 0.0))
        image = image * slope + intercept

        pixel_spacing = getattr(dcm, "PixelSpacing", [1.0, 1.0])
        slice_thickness = float(getattr(dcm, "SliceThickness", 1.0))

        metadata = {
            "pixel_spacing": (float(pixel_spacing[0]), float(pixel_spacing[1])),
            "slice_thickness": slice_thickness,
        }

        return image, metadata

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Cardiac soft-tissue window
        image = np.clip(image, self.hu_min, self.hu_max)
        image = (image - self.hu_min) / (self.hu_max - self.hu_min)

        image = cv2.resize(
            image,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR,
        )

        return image.astype(np.float32)

    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,
        )
        return mask.astype(np.int64)

    def load_mask_slice(self, patient_path: str, slice_idx: int) -> np.ndarray:
        """
        Example mask formats supported:
        1. masks/mask_volume.npy  -> full 3D volume [D, H, W]
        2. masks/{slice_idx:04d}.npy -> per-slice masks
        """
        mask_dir = os.path.join(patient_path, "masks")

        volume_path = os.path.join(mask_dir, "mask_volume.npy")
        if os.path.exists(volume_path):
            mask_volume = np.load(volume_path)
            return mask_volume[slice_idx]

        per_slice_path = os.path.join(mask_dir, f"{slice_idx:04d}.npy")
        if os.path.exists(per_slice_path):
            return np.load(per_slice_path)

        raise FileNotFoundError(
            f"No mask found for patient '{patient_path}' at slice index {slice_idx}"
        )

    def get_slice_indices(self, centre_idx: int, num_slices: int) -> List[int]:
        indices = []
        for offset in range(-self.half_window, self.half_window + 1):
            idx = centre_idx + offset
            idx = max(0, min(idx, num_slices - 1))  # edge padding by repetition
            indices.append(idx)
        return indices

    def __getitem__(self, idx: int):
        sample_info = self.samples[idx]
        patient_path = sample_info["patient_path"]
        patient_id = sample_info["patient_id"]
        centre_idx = sample_info["centre_idx"]
        dicom_paths = sample_info["dicom_paths"]

        slice_indices = self.get_slice_indices(centre_idx, len(dicom_paths))

        stacked_slices = []
        metadata = None

        for slice_idx in slice_indices:
            image, metadata = self.load_dicom_slice(dicom_paths[slice_idx])
            image = self.preprocess_image(image)
            stacked_slices.append(image)

        image_tensor = np.stack(stacked_slices, axis=0)

        target_mask = self.load_mask_slice(patient_path, centre_idx)
        target_mask = self.preprocess_mask(target_mask)

        return {
            "image": torch.tensor(image_tensor, dtype=torch.float32),
            "mask": torch.tensor(target_mask, dtype=torch.long),
            "patient_id": patient_id,
            "slice_idx": centre_idx,
            "pixel_spacing": torch.tensor(metadata["pixel_spacing"], dtype=torch.float32),
            "slice_thickness": torch.tensor(metadata["slice_thickness"], dtype=torch.float32),
    }
    

    def get_patient_ids(self) -> List[str]:
        return sorted(list({sample["patient_id"] for sample in self.samples}))

    def get_indices_for_patients(self, patient_ids: List[str]) -> List[int]:
        patient_id_set = set(patient_ids)
        return [
            idx for idx, sample in enumerate(self.samples)
            if sample["patient_id"] in patient_id_set
    ]