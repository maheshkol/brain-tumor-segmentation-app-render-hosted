from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}
from fastapi import FastAPI, UploadFile, File
import tempfile, base64, cv2
import numpy as np
import torch
from pathlib import Path
import shutil

from .preprocessing import preprocess
from .inference import load_model

app = FastAPI(title="Brain Tumor Segmentation API")

# Load model once at startup
model = load_model("models/unet_brats_trained.pth")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / file.filename

        # Save uploaded file (preserves .nii / .nii.gz)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Preprocess
        tensor, raw = preprocess(str(file_path))

        # Inference
        with torch.no_grad():
            pred = model(tensor)
            print(
                "PRED STATS:",
                pred.min().item(),
                pred.max().item(),
                pred.mean().item()
            )
            #mask = (pred > 0.5).float().cpu().numpy()[0, 0]
            mask = pred.cpu().numpy()[0, 0]


    # ---------- FIXED OVERLAY LOGIC ----------

    # Normalize raw image to uint8
    if raw.dtype != np.uint8:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX)
        raw = raw.astype(np.uint8)

    # Ensure raw image is 3-channel BGR
    if len(raw.shape) == 2:
        raw_color = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        raw_color = raw

    # Ensure mask is uint8 and same size
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (raw_color.shape[1], raw_color.shape[0]))

    # Create red overlay mask
    mask_color = np.zeros_like(raw_color)
    mask_color[:, :, 2] = mask  # Red channel

    # Overlay tumor mask on MRI
    overlay = cv2.addWeighted(
        raw_color, 0.7,
        mask_color, 0.3,
        0
    )

    # Encode outputs
    _, m_buf = cv2.imencode(".png", mask)
    _, o_buf = cv2.imencode(".png", overlay)

    return {
        "mask": base64.b64encode(m_buf).decode("utf-8"),
        "overlay": base64.b64encode(o_buf).decode("utf-8")
    }