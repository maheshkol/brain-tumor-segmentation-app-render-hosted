from cv2.gapi import mask
from fastapi import FastAPI, UploadFile, File
import tempfile, base64, cv2
import numpy as np
import torch
from pathlib import Path
import shutil
import os
import requests 
PORT = int(os.environ.get("PORT", 8000))

from .preprocessing import preprocess
from .inference import load_model, generate_gradcam

app = FastAPI(title="Brain Tumor Segmentation API")

model = None  # global reference

# ✅ Get correct base path (VERY IMPORTANT for Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # backend folder
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "unet_brats_trained.pth")

MODEL_URL = "https://huggingface.co/maheshkol/brain-tumor-unet/resolve/main/unet_brats_trained.pth"

@app.get("/")
def home():
    return {
        "app": "Brain Tumor Segmentation API",
        "status": "running",
        "endpoint": "/predict"
    }

@app.on_event("startup")
def startup_event():
    global model

    # ✅ Ensure models folder exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ------------------------------
    # Download model if not exists
    # ------------------------------
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from HuggingFace...")
        print("Saving to:", MODEL_PATH)

        response = requests.get(MODEL_URL, stream=True)

        # 🔥 Check response
        print("Status Code:", response.status_code)

        if response.status_code != 200:
            raise RuntimeError(f"❌ Failed to download model. Status: {response.status_code}")

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("✅ Model downloaded successfully")

    # ------------------------------
    # Verify file exists
    # ------------------------------
    print("Checking model path:", MODEL_PATH)
    print("File exists?", os.path.exists(MODEL_PATH))

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model still missing at: {MODEL_PATH}")

    # ------------------------------
    # Load model
    # ------------------------------
    print("📦 Loading model...")
    model = load_model(MODEL_PATH)
    model.eval()

    print("✅ Model loaded successfully")

@app.post("/predict")
#async def predict(file: UploadFile = File(...)):
async def predict(file: UploadFile = File(...), gradcam: bool = False):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        tensor, raw = preprocess(str(file_path))

        with torch.no_grad():
            pred = model(tensor)
            mask = pred.cpu().numpy()[0, 0]

    # Normalize raw image
    if raw.dtype != np.uint8:
        raw = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX)
        raw = raw.astype(np.uint8)

    if len(raw.shape) == 2:
        raw_color = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    else:
        raw_color = raw

    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (raw_color.shape[1], raw_color.shape[0]))

# 🔥 HEATMAP CODE (replace old mask_color logic)
    #heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    confidence = pred.cpu().numpy()[0, 0]
    confidence = (confidence * 255).astype(np.uint8)
    confidence = cv2.resize(confidence, (raw_color.shape[1], raw_color.shape[0]))

    confidence_map = cv2.applyColorMap(confidence, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(raw_color, 0.6, confidence_map, 0.4, 0)
    gradcam_overlay = None

    if gradcam:
        gradcam_overlay = generate_gradcam(model, tensor, raw_color)


    _, m_buf = cv2.imencode(".png", mask)
    _, o_buf = cv2.imencode(".png", overlay)
    _, c_buf = cv2.imencode(".png", confidence_map) 
    #_, g_buf = cv2.imencode(".png", gradcam_overlay)

    response = {
        "mask": base64.b64encode(m_buf).decode("utf-8"),
        "overlay": base64.b64encode(o_buf).decode("utf-8"),
        "confidence": base64.b64encode(c_buf).decode("utf-8"),
    
    }

   
    if gradcam_overlay is not None and gradcam_overlay.size != 0:
        _, g_buf = cv2.imencode(".png", gradcam_overlay)
        response["gradcam"] = base64.b64encode(g_buf).decode("utf-8")


    return response

