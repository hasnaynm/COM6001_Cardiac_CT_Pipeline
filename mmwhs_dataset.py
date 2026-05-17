# mmwhs_dataset.py
# dataset loader for the MM-WHS (Multi-Modality Whole Heart Segmentation) dataset
# loads NIfTI volumes, applies HU windowing, resizes slices, builds 2.5D sample index


from pathlib import Path
import numpy as np
import nibabel as nib        # for loading NIfTI (.nii/.nii.gz) medical image files
import torch
from torch.utils.data import Dataset
from PIL import Image        # used for resizing individual slices




# label remapping - MM-WHS uses arbitrary integer values for each structure
# we remap them to 0-4 for use as class indices in the model
# anything not in this map (other structures) gets mapped to 0 (background)
MMWHS_LABEL_MAP = {
    0: 0,    # background -> background
    500: 1,  # left ventricle -> class 1
    600: 2,  # right ventricle -> class 2
    420: 3,  # left atrium -> class 3
    550: 4,  # right atrium -> class 4
    205: 0,  # myocardium -> background (not segmenting this)
    820: 0,  # aorta -> background
    850: 0,  # pulmonary artery -> background
}




def remap_labels(label_array):
    # replaces original MM-WHS label values with our 0-4 class indices
    output = np.zeros_like(label_array, dtype=np.int64)  # start with all zeros (background)
    for original_val, new_val in MMWHS_LABEL_MAP.items():
        output[label_array == original_val] = new_val    # replace each value
    return output




def resize_volume(vol, img_size):
    # resizes every 2D slice in a 3D volume to img_size x img_size
    # volume shape expected: [H, W, D] where D = number of slices
    d = vol.shape[2]  # number of slices
    resized = np.zeros((img_size, img_size, d), dtype=np.float32)
    for z in range(d):
        s = vol[:, :, z].astype(np.float32)          # get single slice
        pil = Image.fromarray(s)                      # convert to PIL image for resizing
        pil = pil.resize((img_size, img_size), Image.BILINEAR)  # bilinear interpolation
        resized[:, :, z] = np.array(pil)             # store resized slice back
    return resized





class MMWHSDataset(Dataset):
    # pytorch dataset for MM-WHS NIfTI data
    # loads all volumes into memory at init, then serves 2.5D slice stacks on demand
    # 2.5D = centre slice + neighbouring slices stacked as channels

    def __init__(self, data_dir, num_input_slices=5, hu_min=-150.0, hu_max=250.0, img_size=128):
        if num_input_slices % 2 == 0:
            raise ValueError("num_input_slices must be odd.")  # must be odd so there's a clear centre slice
        self.data_dir = Path(data_dir)
        self.num_input_slices = num_input_slices
        self.hu_min = hu_min       # lower HU clip value (cardiac soft tissue window)
        self.hu_max = hu_max       # upper HU clip value
        self.img_size = img_size
        self.half = num_input_slices // 2  # how many slices above/below centre to include
        self.volumes = []  # stores all loaded volumes as tuples (img_vol, lbl_vol, pixdim)
        self.samples = []  # index of all valid (volume, slice) pairs
        self._load_all_volumes()  # load everything at startup


    def _load_all_volumes(self):
        # finds all image NIfTI files and loads them alongside their label files
        image_paths = sorted(self.data_dir.glob("*_image.nii*"))  # match .nii and .nii.gz

        for img_path in image_paths:
            # expect label file to have same name but with _label instead of _image
            label_path = Path(str(img_path).replace("_image.nii", "_label.nii"))
            if not label_path.exists():
                print(f"Warning: no label found for {img_path.name}, skipping.")
                continue
            print(f"Loading {img_path.name}...")


            # load image volume
            img_nib = nib.load(str(img_path))
            img_vol = img_nib.get_fdata().astype(np.float32)
            img_vol = np.clip(img_vol, self.hu_min, self.hu_max)  # apply HU window
            img_vol = (img_vol - self.hu_min) / (self.hu_max - self.hu_min + 1e-8)  # normalise to 0-1
            img_vol = resize_volume(img_vol, self.img_size)  # resize all slices


            # load label volume
            lbl_nib = nib.load(str(label_path))
            lbl_vol = lbl_nib.get_fdata().astype(np.float32)
            lbl_vol = remap_labels(lbl_vol.astype(np.int64)).astype(np.float32)  # remap to 0-4
            lbl_vol = resize_volume(lbl_vol, self.img_size)


            # get physical pixel dimensions from NIfTI header (used for volume calculation later)
            pixdim = img_nib.header.get_zooms()
            patient_id = img_path.stem.replace("_image", "")
            vol_idx = len(self.volumes)  # index of this volume in self.volumes list
            self.volumes.append((img_vol, lbl_vol, pixdim))


            # build sample index - only include slices where we have enough neighbours
            # skip first and last self.half slices so we can always build a full stack
            depth = img_vol.shape[2]
            for z in range(self.half, depth - self.half):
                self.samples.append({
                    "vol_idx": vol_idx,
                    "slice_idx": z,
                    "patient_id": patient_id,
                })


        print(f"Loaded {len(self.volumes)} volumes, {len(self.samples)} slices total.")



    def __len__(self):
        return len(self.samples)  # total number of 2.5D samples across all volumes



    def __getitem__(self, idx):
        # returns a single 2.5D sample: a stack of slices centred on slice z
        sample = self.samples[idx]
        vol_idx = sample["vol_idx"]
        z = sample["slice_idx"]
        img_vol, lbl_vol, pixdim = self.volumes[vol_idx]


        # build 2.5D input: stack slices from z-half to z+half inclusive
        slices = []
        for offset in range(-self.half, self.half + 1):
            s = img_vol[:, :, z + offset]  # grab neighbouring slice
            slices.append(s)
        image = np.stack(slices, axis=0).astype(np.float32)  # shape: [num_input_slices, H, W]


        # label is just the centre slice
        label = lbl_vol[:, :, z].astype(np.int64)


        # physical spacing from header - needed for volumetric quantification
        pixel_spacing = torch.tensor([float(pixdim[0]), float(pixdim[1])], dtype=torch.float32)
        slice_thickness = torch.tensor(float(pixdim[2]), dtype=torch.float32)



        return {
            "image": torch.FloatTensor(image),   # [num_input_slices, H, W]
            "mask": torch.LongTensor(label),      # [H, W] integer class labels
            "patient_id": sample["patient_id"],
            "slice_idx": z,
            "pixel_spacing": pixel_spacing,
            "slice_thickness": slice_thickness,
        }
