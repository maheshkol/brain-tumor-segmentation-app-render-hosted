🧠 Brain Tumor Segmentation App using MRI 

(NMR-based Imaging) 

 

📌 Project Overview 

This project implements an end-to-end deep learning application for automatic brain tumor segmentation from MRI scans, which are generated using Nuclear Magnetic Resonance (NMR) principles. 

The system uses U-Net and Attention U-Net deep learning models to perform semantic segmentation of tumor regions and provides a web-based interface for visualization and analysis. 

The application is designed to demonstrate: 

Medical imaging understanding 

Deep learning–based segmentation 

Full-stack ML system design (frontend + backend + model) 

🔬 Relation Between NMR and MRI 

Nuclear Magnetic Resonance (NMR) is a physical phenomenon where nuclei absorb and emit radiofrequency energy in a magnetic field. 

Magnetic Resonance Imaging (MRI) applies NMR principles to generate high-resolution images of soft tissues. 

Brain tumors alter tissue relaxation properties (T1, T2, FLAIR), making them detectable in MRI. 

Deep learning models learn these intensity and spatial patterns produced by NMR-based imaging. 

 
🎯 Problem Statement 

Manual brain tumor segmentation: 

Is time-consuming for radiologists 

Is subjective and error-prone 

Struggles with small or irregular tumors 

This project automates the segmentation process using convolutional neural networks, improving accuracy, consistency, and speed. 

 

🗂 Dataset 

BraTS (Brain Tumor Segmentation) Dataset 

Multi-modal MRI scans: 

T1 

T1ce (contrast-enhanced) 

T2 

FLAIR 

Ground truth masks: 

Whole Tumor (WT) 

Tumor Core (TC) 

Enhancing Tumor (ET) 

Format: .nii.gz (NIfTI) 

⚠️ This repository does not include the full BraTS dataset. 
Sample MRI files can be placed inside data/sample_mri/. 

 
🏗 System Architecture 

 

User (Browser) 

   ↓ 

Streamlit Frontend 

   ↓  REST API 

FastAPI Backend 

   ↓ 

Preprocessing Pipeline 

   ↓ 

U-Net / Attention U-Net Model 

   ↓ 

Postprocessing + Visualization 

 

 🧠 How the Project Works (Step-by-Step) 

1️⃣ MRI Upload 

User uploads an MRI file (.nii or .nii.gz) via the web interface. 

2️⃣ Backend Processing 

The backend extracts a representative slice from the MRI volume. 

Image normalization (Z-score normalization). 

Resizing to model-compatible dimensions. 

3️⃣ Model Inference 

Preprocessed image is passed to: 

U-Net or 

Attention U-Net 

The model outputs a probability mask for tumor regions. 

4️⃣ Postprocessing 

Probability map is thresholded. 

Binary tumor mask is generated. 

Mask is overlaid on the original MRI slice. 

5️⃣ Visualization 

Frontend displays: 

Predicted tumor mask 

Tumor overlay on MRI image 

 


📁 Project Structure 

 

brain-tumor-segmentation-app/ 

│ 

├── frontend/              # Streamlit UI 

│   ├── app.py 

│   └── ui_utils.py 

│ 

├── backend/               # FastAPI backend 

│   ├── main.py 

│   ├── inference.py 

│   ├── preprocessing.py 

│   └── metrics.py 

│ 

├── models/                # Trained model weights 

│   ├── unet.pth 

│   └── attention_unet.pth 

│ 

├── data/ 

│   └── sample_mri/ 

│ 

├── notebooks/             # Exploration & experiments 

│   └── exploration.ipynb 

│ 

├── docker/                # Docker configuration 

│   ├── Dockerfile.backend 

│   └── Dockerfile.frontend 

│ 

├── requirements.txt 

├── README.md 

└── LICENSE 

 
🧑‍💻 Technologies Used 

🔹 Programming Language 

Python 3.9+ 

🔹 Deep Learning 

PyTorch 

U-Net 

Attention U-Net 

🔹 Medical Imaging 

NiBabel (NIfTI MRI handling) 

OpenCV 

NumPy 

🔹 Backend 

FastAPI 

Uvicorn 

🔹 Frontend 

Streamlit 

Pillow 

Matplotlib 

🔹 DevOps / Deployment 

Docker 

GitHub 

 

⚙️ Installation & Setup 

1️⃣ Clone Repository 

git clone https://github.com/your-username/brain-tumor-segmentation-app.git 

cd brain-tumor-segmentation-app 

 
2️⃣ Install Dependencies 

pip install -r requirements.txt 

 

▶️ Running the Application (Local) 

🔹 Start Backend (FastAPI) 

 

uvicorn backend.main:app –reload 

 

Backend runs at: 

 

http://localhost:8000 

 

🔹 Start Frontend (Streamlit) 

 

Open a new terminal: 

 

streamlit run frontend/app.py 

 

Frontend runs at: 

 

http://localhost:8501 

 

 

 🧪 Sample Workflow 

Open the Streamlit app in browser 

Upload an MRI .nii or .nii.gz file 

Backend processes the image 

Model predicts tumor mask 

Segmentation results are displayed 

 

📊 Evaluation Metrics 

Dice Score 
Measures overlap between prediction and ground truth. 

Intersection over Union (IoU) 
Measures region-level accuracy. 

Metrics are implemented and can be enabled when ground truth masks are available. 

🔒 Data Privacy & Ethics 

No patient-identifying information is stored 

Designed for research and educational purposes 

Not intended for clinical diagnosis 

 

🚀 Future Enhancements 

Multi-modal MRI fusion (T1 + T2 + FLAIR) 

3D U-Net implementation 

Real-time Dice/IoU computation 

Grad-CAM explainability 

Cloud deployment (AWS/GCP) 

PACS system integration 

 

 