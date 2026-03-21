import os
import numpy as np
import nibabel as nib
from PIL import Image

BASE_DIR = "brainSegData"

INPUT_DIRS = {
    "yes": os.path.join(BASE_DIR, "yes"),
    "no": os.path.join(BASE_DIR, "no"),
}

OUT_DIR = "nifti_data/images"
os.makedirs(OUT_DIR, exist_ok=True)

def png_to_nifti(png_path, out_path):
    img = Image.open(png_path).convert("L")
    arr = np.array(img)

    # Normalize
    if arr.max() > 0:
        arr = arr / arr.max()

    # 2D → 3D (H, W, 1)
    arr = arr[:, :, np.newaxis]

    affine = np.eye(4)
    nii = nib.Nifti1Image(arr.astype(np.float32), affine)
    nib.save(nii, out_path)

count = 0
for label, folder in INPUT_DIRS.items():
    if not os.path.exists(folder):
        print(f"⚠️ Folder not found: {folder}")
        continue

    for fname in sorted(os.listdir(folder)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            in_path = os.path.join(folder, fname)

            out_name = f"{label}_{count:03d}.nii.gz"
            out_path = os.path.join(OUT_DIR, out_name)

            png_to_nifti(in_path, out_path)
            count += 1

print(f"✅ Converted {count} images to NIfTI")
