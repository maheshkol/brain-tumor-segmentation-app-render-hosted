import torch
from torch.utils.data import DataLoader
from dataset import BraTSDataset
from model import UNet
from losses import dice_loss
from pathlib import Path

# -------------------------
# Config
# -------------------------
#DATA_ROOT = Path("../data/brats")
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "brats"
print("Resolved BraTS path:", DATA_ROOT)

if not DATA_ROOT.exists():
    raise RuntimeError(f"BraTS dataset not found at: {DATA_ROOT}")
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Looking for BraTS data at: {DATA_ROOT.resolve()}")

# -------------------------
# Dataset (THIS WAS MISSING)
# -------------------------
dataset = BraTSDataset(str(DATA_ROOT))

print(f"✅ Loaded {len(dataset)} valid BraTS cases")

# ❌ Stop early if dataset is empty
if len(dataset) == 0:
    raise RuntimeError("No valid BraTS cases found. Check dataset structure.")

# -------------------------
# DataLoader
# -------------------------
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

# -------------------------
# Model
# -------------------------
model = UNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for step, (img, mask) in enumerate(loader):
        img = img.to(device)
        mask = mask.to(device)

        pred = model(img)
        loss = dice_loss(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if step % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{EPOCHS}] "
                f"Step [{step}/{len(loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_loss = epoch_loss / len(loader)
    print(f"✅ Epoch [{epoch+1}] completed | Avg Dice Loss: {avg_loss:.4f}")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), "unet_brats_trained.pth")
print("💾 Model saved as unet_brats_trained.pth")
