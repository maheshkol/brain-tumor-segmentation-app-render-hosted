import os
import gzip
import shutil

BRATS_ROOT = r"C:\Users\Mahesh\Documents\brain-tumor-segmentation-app\data\brats"

def is_gzipped(path):
    try:
        with gzip.open(path, "rb") as f:
            f.read(1)
        return True
    except:
        return False

fixed = 0

for case in os.listdir(BRATS_ROOT):
    case_path = os.path.join(BRATS_ROOT, case)
    if not os.path.isdir(case_path):
        continue

    for fname in os.listdir(case_path):
        full = os.path.join(case_path, fname)

        # Files that LIE about being gzipped
        if fname.endswith(".nii.gz") and not is_gzipped(full):
            new_name = fname.replace(".nii.gz", ".nii")
            new_path = os.path.join(case_path, new_name)

            print(f"🔧 Fixing extension: {fname} → {new_name}")
            os.rename(full, new_path)
            fixed += 1

print(f"\n✅ Fixed {fixed} broken BraTS files")
