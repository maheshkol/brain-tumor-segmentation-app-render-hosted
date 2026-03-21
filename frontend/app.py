import streamlit as st
import requests
import base64
from PIL import Image
import io
import nibabel as nib
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet

# ------------------------------
# CONFIG
# ------------------------------
#BACKEND_URL = "http://127.0.0.1:8000/predict"
BACKEND_URL = "https://brain-tumor-segmentation-app.onrender.com/predict"

st.set_page_config(
    page_title="Brain Tumor Segmentation AI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0B0F1A;
    color: #EAEAEA;
}

h1, h2, h3 {
    color: #FFFFFF;
}

/* Buttons */
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    padding: 10px;
}

/* Metric card */
[data-testid="stMetric"] {
    background-color: #111827;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------
# HEADER
# ------------------------------
st.title("🧠 Brain Tumor Segmentation using Deep Learning")

st.markdown("""
### 🏥 AI Radiology Dashboard  
**Model:** U-Net | **Dataset:** BraTS | **Mode:** Inference             
This application uses a **UNet deep learning model trained on the BraTS dataset**
to automatically **segment brain tumors from MRI scans (FLAIR images)**.

⚠️ **Disclaimer**  
This tool is for **educational and research purposes only**.  
It is **not a medical diagnostic system**.
""")

st.divider()

# ------------------------------
# SIDEBAR
# ------------------------------
st.sidebar.header("📌 About the Model")
st.sidebar.markdown("""
- **Architecture**: UNet  
- **Training Data**: BraTS  
- **Input**: MRI FLAIR (.nii / .nii.gz)  
- **Output**: Tumor mask + overlay  
- **Loss**: Dice Loss  
""")

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "📂 Upload MRI (.nii or .nii.gz)",
    type=["nii", "gz"]
)

# ------------------------------
# HELPERS
# ------------------------------
def decode_base64_image(b64_string):
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def generate_pdf_report(area_cm2, mask_img, overlay_img):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    doc = SimpleDocTemplate(tmp.name)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("Brain Tumor Segmentation Report", styles['Title']))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Tumor Area: {area_cm2:.2f} cm²", styles['Normal']))
    elements.append(Spacer(1, 10))

    # Save images temporarily
    mask_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    overlay_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    mask_img.save(mask_path)
    overlay_img.save(overlay_path)

    elements.append(Paragraph("Segmentation Mask", styles['Heading2']))
    elements.append(RLImage(mask_path, width=300, height=300))

    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Overlay", styles['Heading2']))
    elements.append(RLImage(overlay_path, width=300, height=300))

    doc.build(elements)

    return tmp.name


def show_legend():
    st.markdown("### 🎨 Heatmap Legend")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("🔵 **Low Importance / Low Confidence**")

    with col2:
        st.markdown("🟢 **Medium**")

    with col3:
        st.markdown("🔴 **High Importance / Tumor Region**")


# ------------------------------
# MAIN LOGIC
# ------------------------------
if uploaded_file is not None:

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    nii = nib.load(tmp_path)
    volume = nii.get_fdata()

    slice_idx = st.slider(
        "📍 Select MRI Slice",
        0, volume.shape[2] - 1,
        volume.shape[2] // 2
    )

    slice_img = volume[:, :, slice_idx]
    slice_img = (slice_img - slice_img.min()) / (slice_img.max() - slice_img.min())

    st.subheader("🧠 MRI Slice Viewer")
    st.image(slice_img, width="stretch")

    # Grad-CAM toggle
    show_gradcam = st.checkbox("🔥 Enable Grad-CAM")

    # Run prediction
    if st.button("🚀 Run Segmentation"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/octet-stream"
            )
        }

        with st.spinner("🔬 Running tumor segmentation..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    files=files,
                    params={"gradcam": show_gradcam},
                    timeout=300
                )
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Is FastAPI running?")
                st.stop()

        if response.status_code != 200:
            st.error("❌ Backend error")
            st.code(response.text)
            st.stop()

        # ✅ SAFE DATA
        data = response.json()

        # ------------------------------
        # Decode Images
        # ------------------------------
        mask_img = decode_base64_image(data.get("mask"))
        overlay_img = decode_base64_image(data.get("overlay"))

        gradcam_img = None
        if show_gradcam and "gradcam" in data:
            gradcam_img = decode_base64_image(data["gradcam"])

        confidence_img = None
        if "confidence" in data:
            confidence_img = decode_base64_image(data["confidence"])

        # ------------------------------
        # TABS UI
        # ------------------------------
        tabs = st.tabs(["🎯 Mask", "🔍 Overlay", "🔥 Grad-CAM", "🚀 Confidence"])

        # TAB 1 — MASK
        with tabs[0]:
            st.subheader("Tumor Segmentation Mask")

            if mask_img:
                st.image(mask_img, width="stretch")

                st.download_button(
                    "⬇️ Download Mask",
                    pil_to_bytes(mask_img),
                    file_name="mask.png"
                )
            else:
                st.warning("Mask not available")

        # TAB 2 — OVERLAY
        with tabs[1]:
            st.subheader("Tumor Overlay on MRI")

            if overlay_img:
                st.image(overlay_img, width="stretch")

                alpha = st.slider("🔄 Compare Original ↔ Overlay", 0.0, 1.0, 0.5)

                #orig_img = Image.fromarray((slice_img * 255).astype(np.uint8)).convert("RGB")
                #blend = Image.blend(orig_img, overlay_img, alpha)
                orig_img = Image.fromarray((slice_img * 255).astype(np.uint8)).convert("RGB")

                # ✅ Ensure same size
                overlay_resized = overlay_img.resize(orig_img.size)

                # ✅ Ensure same mode
                overlay_resized = overlay_resized.convert("RGB")

                # ✅ Now safe to blend
                blend = Image.blend(orig_img, overlay_resized, alpha)

                st.image(blend, caption="Comparison View", width="stretch")

                st.download_button(
                    "⬇️ Download Overlay",
                    pil_to_bytes(overlay_img),
                    file_name="overlay.png"
                )
            else:
                st.warning("Overlay not available")

        # TAB 3 — GRAD-CAM
        with tabs[2]:
            if gradcam_img:
                st.subheader("Grad-CAM Visualization")
                st.image(gradcam_img, width="stretch")
                show_legend()

                st.download_button(
                    "⬇️ Download Grad-CAM",
                    pil_to_bytes(gradcam_img),
                    file_name="gradcam.png"
                )
            else:
                st.info("Enable Grad-CAM to view")

        # TAB 4 — CONFIDENCE
        with tabs[3]:
            if confidence_img:
                st.subheader("Confidence Heatmap")
                st.image(confidence_img, width="stretch")

                st.download_button(
                    "⬇️ Download Confidence Map",
                    pil_to_bytes(confidence_img),
                    file_name="confidence.png"
                )
            else:
                st.info("Confidence map not available")

        # ------------------------------
        # Tumor Area
        # ------------------------------
        mask_np = np.array(mask_img.convert("L"))
        tumor_pixels = np.sum(mask_np > 0)

        tumor_area_cm2 = tumor_pixels / 100.0

        st.metric(
            label="🧠 Estimated Tumor Area",
            value=f"{tumor_area_cm2:.2f} cm²"
        )

        show_legend()

        st.divider()

        # ------------------------------
        # PDF REPORT
        # ------------------------------
        pdf_path = generate_pdf_report(
            tumor_area_cm2,
            mask_img,
            overlay_img
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download Diagnostic Report (PDF)",
                f,
                file_name="brain_tumor_report.pdf"
            )

        # ------------------------------
        # Interpretation
        # ------------------------------
        st.subheader("📊 Interpretation")

        st.markdown("""
        **How to read the results:**
        - **White area** → Tumor region
        - **Overlay** → Tumor on MRI
        - **Grad-CAM** → Model attention

        ⚠️ Always consult a radiologist.
        """)

        st.success("✅ Segmentation completed successfully")

else:
    st.info("👈 Upload an MRI scan from the sidebar to begin.")
