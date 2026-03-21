import torch
from backend.inference import load_model
from backend.preprocessing import preprocess

# change this to one REAL .nii.gz file
TEST_FILE = "nifti_data/images/no_155.nii.gz"

model = load_model("models/unet_brats_trained.pth")

tensor, raw = preprocess(TEST_FILE)

with torch.no_grad():
    pred = model(tensor)

print("INPUT STATS:", tensor.min().item(), tensor.max().item(), tensor.mean().item())
print("PRED STATS:", pred.min().item(), pred.max().item(), pred.mean().item())

# Check spatial variation
diff = (pred - pred.mean()).abs().mean()
print("MEAN ABS DEVIATION:", diff.item())
