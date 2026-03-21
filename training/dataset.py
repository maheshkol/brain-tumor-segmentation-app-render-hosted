import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        print(f"Looking for BraTS data at: {root_dir}")

        for case in os.listdir(root_dir):
            case_path = os.path.join(root_dir, case)
            if not os.path.isdir(case_path):
                continue

            files = os.listdir(case_path)

            flair = [f for f in files if "_flair" in f and f.endswith(".nii")]
            seg = [f for f in files if "_seg" in f and f.endswith(".nii")]

            if not flair or not seg:
                print(f"⚠️ Skipping {case}: missing flair or seg")
                continue

            self.samples.append({
                "image": os.path.join(case_path, flair[0]),
                "mask": os.path.join(case_path, seg[0])
            })

        print(f"✅ Loaded {len(self.samples)} valid BraTS cases")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img = nib.load(item["image"]).get_fdata()
        mask = nib.load(item["mask"]).get_fdata()

        # Middle slice
        z = img.shape[2] // 2
        img = img[:, :, z]
        mask = mask[:, :, z]

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        mask = (mask > 0).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask
