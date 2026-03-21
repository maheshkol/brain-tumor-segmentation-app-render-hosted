import torch
from pathlib import Path
from .model import UNet
import cv2
import numpy as np


# ------------------------------
# LOAD MODEL
# ------------------------------
def load_model(model_name: str):
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / model_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at: {model_path}")

    model = UNet()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ------------------------------
# 🔥 GRAD-CAM (IMPROVED)
# ------------------------------
def generate_gradcam(model, input_tensor, raw_image):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # 🎯 Best layer (deepest features)
    target_layer = model.bottleneck.conv

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    # 🔥 Enable gradients
    input_tensor = input_tensor.requires_grad_(True)

    # Forward pass
    output = model(input_tensor)
    prob = torch.sigmoid(output)

    # 🔥 Focus on tumor region
    target = (prob > 0.5).float()
    loss = (prob * target).sum()

    model.zero_grad()
    loss.backward()

    grads = gradients[0][0].cpu().data.numpy()
    acts = activations[0][0].cpu().data.numpy()

    # Channel importance
    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    # ReLU
    cam = np.maximum(cam, 0)

    # 🔥 Improved normalization
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

    # Resize to original image
    cam = cv2.resize(cam, (raw_image.shape[1], raw_image.shape[0]))
    cam = (cam * 255).astype(np.uint8)

    # Heatmap overlay
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_image, 0.6, heatmap, 0.4, 0)

    handle_f.remove()
    handle_b.remove()

    return overlay


# ------------------------------
# 🚀 CONFIDENCE HEATMAP
# ------------------------------
def generate_confidence_map(output, raw_image):
    prob = torch.sigmoid(output).cpu().numpy()[0, 0]

    # 🔥 Better normalization
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)

    prob = cv2.resize(prob, (raw_image.shape[1], raw_image.shape[0]))
    prob = (prob * 255).astype(np.uint8)

    heatmap = cv2.applyColorMap(prob, cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(raw_image, 0.6, heatmap, 0.4, 0)

    return overlay