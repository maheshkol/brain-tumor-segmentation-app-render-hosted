import os

ROOT = "data/brats"

for case in os.listdir(ROOT):
    case_path = os.path.join(ROOT, case)
    if not os.path.isdir(case_path):
        continue

    for fname in os.listdir(case_path):
        old_path = os.path.join(case_path, fname)

        if fname.endswith("_seg"):
            new_path = old_path + ".nii.gz"
        elif "_flair" in fname and not fname.endswith(".nii.gz"):
            new_path = old_path + ".nii.gz"
        elif "_t1" in fname and not fname.endswith(".nii.gz"):
            new_path = old_path + ".nii.gz"
        elif "_t1ce" in fname and not fname.endswith(".nii.gz"):
            new_path = old_path + ".nii.gz"
        elif "_t2" in fname and not fname.endswith(".nii.gz"):
            new_path = old_path + ".nii.gz"
        else:
            continue

        if not os.path.exists(new_path):
            os.rename(old_path, new_path)
            print(f"Renamed: {fname} → {os.path.basename(new_path)}")

print("✅ BraTS file extensions fixed.")
