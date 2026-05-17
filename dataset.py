# dataset.py
# synthetic dummy dataset used for early pipeline testing before real data was available
# generates fake CT slices with ellipse-shaped chamber masks
# not used in final training - replaced by mmwhs_dataset.py



from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class DummyCardiacCTDataset(Dataset):
    # generates synthetic 2.5D cardiac CT samples on the fly
    # each sample has a fake CT image stack and a fake segmentation mask
    # chambers are approximated as ellipses in roughly the right positions
    # output:
    #   image: [C, H, W] where C = num_input_slices (5 slices stacked)
    #   mask:  [H, W] integer labels - 0=background, 1=LV, 2=RV, 3=LA, 4=RA



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
        self.num_samples = num_samples        # how many fake samples to generate
        self.num_input_slices = num_input_slices
        self.num_classes = num_classes



    def __len__(self) -> int:
        return self.num_samples



    def _make_ellipse_mask(self, yy, xx, center_x, center_y, radius_x, radius_y):
        # returns a boolean mask - True inside the ellipse, False outside
        # ellipse equation: (x-cx)^2/rx^2 + (y-cy)^2/ry^2 <= 1
        return (
            ((xx - center_x) ** 2) / (radius_x ** 2)
            + ((yy - center_y) ** 2) / (radius_y ** 2)
            <= 1
        )


    def __getitem__(self, idx: int):
        h = w = self.image_size
        yy, xx = np.ogrid[:h, :w]  # coordinate grids for ellipse generation

        # start with all background
        mask = np.zeros((h, w), dtype=np.int64)

        # centre of image - chambers placed relative to this
        cx = w // 2
        cy = h // 2

        # random jitter so chambers arent always in exactly the same place
        jitter_x = np.random.randint(-10, 10)
        jitter_y = np.random.randint(-10, 10)



        # create ellipse masks for each chamber
        # positions and sizes randomised slightly each sample
        lv = self._make_ellipse_mask(
            yy, xx,
            cx - 25 + jitter_x, cy + 20 + jitter_y,    # LV roughly bottom-left of centre
            np.random.randint(18, 28), np.random.randint(22, 32)
        )

        rv = self._make_ellipse_mask(
            yy, xx,
            cx + 20 + jitter_x, cy + 20 + jitter_y,    # RV bottom-right
            np.random.randint(18, 28), np.random.randint(20, 30)
        )

        la = self._make_ellipse_mask(
            yy, xx,
            cx - 20 + jitter_x, cy - 25 + jitter_y,     # LA top-left
            np.random.randint(16, 24), np.random.randint(14, 22)
        )

        ra = self._make_ellipse_mask(
            yy, xx,
            cx + 20 + jitter_x, cy - 25 + jitter_y,    # RA top-right
            np.random.randint(16, 24), np.random.randint(14, 22)
        )


        # assign class labels - later writes overwrite earlier ones where chambers overlap
        mask[lv] = 1
        mask[rv] = 2
        mask[la] = 3
        mask[ra] = 4



        # build 2.5D image stack - one slice per channel
        # each slice has the same mask but slightly different intensity values
        image_stack = []
        for slice_idx in range(self.num_input_slices):
            # gaussian noise background to simulate CT texture
            noise = np.random.normal(loc=0.08, scale=0.03, size=(h, w)).astype(np.float32)
            slice_image = noise.copy()



            # add chamber intensities - each chamber gets a slightly different brightness
            # small random variation per slice imitates how real CT slices differ slightly
            slice_image += (mask == 1).astype(np.float32) * (0.45 + np.random.uniform(-0.03, 0.03))  # LV
            slice_image += (mask == 2).astype(np.float32) * (0.38 + np.random.uniform(-0.03, 0.03))  # RV
            slice_image += (mask == 3).astype(np.float32) * (0.52 + np.random.uniform(-0.03, 0.03))  # LA
            slice_image += (mask == 4).astype(np.float32) * (0.48 + np.random.uniform(-0.03, 0.03))  # RA

            # slight global intensity shift per slice to mimic inter-slice variation
            slice_image += np.random.uniform(-0.02, 0.02)

            slice_image = np.clip(slice_image, 0.0, 1.0)  # keep values in valid range
            image_stack.append(slice_image)


        # stack slices along first axis to get [C, H, W]
        image = np.stack(image_stack, axis=0).astype(np.float32)

        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "patient_id": f"dummy_{idx:03d}",
            "slice_idx": self.num_input_slices // 2,  # centre slice index
            "pixel_spacing": torch.tensor([1.0, 1.0], dtype=torch.float32),  # fake 1mm spacing
            "slice_thickness": torch.tensor(1.0, dtype=torch.float32),        # fake 1mm thickness
        }
