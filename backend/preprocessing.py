import cv2
import numpy as np
import torch
import nibabel as nib
import os


def preprocess(path: str):
    ext = os.path.splitext(path)[1].lower()

    # -----------------------------
    # Case 1: NIfTI (.nii / .nii.gz)
    # -----------------------------
    if ext in [".nii", ".gz"]:
        nii = nib.load(path)
        img = nii.get_fdata()

        # Take middle slice if 3D
        if img.ndim == 3:
            img = img[:, :, img.shape[2] // 2]

        img = img.astype(np.float32)

    # -----------------------------
    # Case 2: Image (.png / .jpg / .jpeg)
    # -----------------------------
    elif ext in [".png", ".jpg", ".jpeg"]:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Failed to read image file")

        img = img.astype(np.float32)

    else:
        raise ValueError("Unsupported file type")
    
    print("RAW STATS:", img.min(), img.max(), img.mean())


    # -----------------------------
    # Normalize (CRITICAL)
    # -----------------------------
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Resize to model input
    img = cv2.resize(img, (256, 256))

    # Save raw image for overlay
    raw = img.copy()

    # Convert to tensor: [1, 1, H, W]
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    return tensor, raw
