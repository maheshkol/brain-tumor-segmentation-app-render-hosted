import base64
import json
import cv2
import numpy as np

# 1️⃣ Load API response JSON (copy-paste from Swagger or curl)
with open("response.json", "r") as f:
    data = json.load(f)

# 2️⃣ Decode mask
mask_bytes = base64.b64decode(data["mask"])
mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
mask_img = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

cv2.imwrite("pred_mask.png", mask_img)

# 3️⃣ Decode overlay
overlay_bytes = base64.b64decode(data["overlay"])
overlay_np = np.frombuffer(overlay_bytes, dtype=np.uint8)
overlay_img = cv2.imdecode(overlay_np, cv2.IMREAD_COLOR)

cv2.imwrite("pred_overlay.png", overlay_img)

print("Saved: pred_mask.png and pred_overlay.png")
