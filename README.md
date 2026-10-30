# 🧠 NeuroLens AI — Parkinson's Disease Detection via Explainable AI

A deep learning-based system that analyzes 3D brain MRI scans and detects **Parkinson's Disease** using ResNet18 with **Grad-CAM explainability**.

---

## 📌 Project Overview

| Item | Details |
|------|---------|
| **Dataset** | PPMI (Parkinson's Progression Markers Initiative) |
| **Model** | ResNet18 (Transfer Learning, 2.5D slice triplet input) |
| **Input** | 3D Brain MRI (NIfTI `.nii.gz` format) |
| **Output** | PD / HC classification + Confidence + Heatmap |
| **Framework** | PyTorch + FastAPI + SimpleITK |

---

## 🗂️ Project Structure

```
sem5_edi/
├── app.py                        # FastAPI web application
├── dataset_manifest.csv          # PPMI subject metadata
├── src/
│   ├── model.py                  # ResNet18 model architecture
│   ├── dataset.py                # MRISliceDataset + patient-level splits
│   ├── preprocess_mri.py         # Full MRI preprocessing pipeline
│   ├── train.py                  # Training loop + evaluation
│   └── generate_all_midsem_plots.py  # 8 evaluation plots generator
└── data/
    ├── dataset_splits.json       # Train/Val/Test subject splits
    ├── midsem_test_metrics.csv   # Evaluation metrics
    └── plots/                    # Generated evaluation plots
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Parth-kulkarni300/sem5_edi.git
cd sem5_edi

# 2. Install dependencies
pip install torch torchvision fastapi uvicorn SimpleITK nibabel numpy pandas matplotlib scikit-learn

# 3. Run the web app
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Then open `http://127.0.0.1:8000` in your browser.

---

## 🏥 MRI Preprocessing Pipeline (5 Steps)

1. **DICOM → NIfTI** — Convert raw DICOM scans to NIfTI format
2. **RAS Reorientation** — Standardize anatomical orientation (Right-Anterior-Superior)
3. **1.0mm Isotropic Resampling** — Normalize voxel spacing across all subjects
4. **Skull Stripping** — Remove skull using Otsu threshold + connected components
5. **Z-score Normalization** — Normalize intensity using brain tissue statistics only

Run preprocessing on your PPMI data:
```bash
python src/preprocess_mri.py
```

---

## 🤖 Model Training

```bash
python src/train.py
```

- **Architecture:** ResNet18 with frozen backbone (Transfer Learning from ImageNet)
- **Input:** 2.5D — 3 adjacent axial slices stacked as RGB (224×224×3)
- **Optimizer:** AdamW (lr=5e-4, weight_decay=1e-2)
- **Scheduler:** Cosine Annealing LR
- **Loss:** CrossEntropyLoss with class weights (PD is penalized more for misses)
- **Evaluation:** Volume-level multi-triplet voting (5 triplets per subject)

---

## 📊 Evaluation Metrics (Test Set, τ = 0.35)

| Metric | Score |
|--------|-------|
| Accuracy | 44.8% |
| Precision | 45.5% |
| Recall (Sensitivity) | 71.4% |
| Specificity | 20.0% |
| F1 Score | 0.556 |
| ROC-AUC | 0.486 |

---

## 📈 Generate All Evaluation Plots

```bash
python src/generate_all_midsem_plots.py
```

Generates 8 plots in `data/plots/`:
- ① Training/Validation Loss
- ② Training/Validation Balanced Accuracy
- ③ ROC Curve + ROC-AUC
- ④ Confusion Matrix
- ⑥ Probability Distribution Histogram
- ⑧ Threshold vs Sensitivity/Specificity Trade-off

---

## 👥 Team Contributions

| Member | Role | Modules |
|--------|------|---------|
| Group A — Person 1 | Data Acquisition & Conversion | `preprocess_mri.py` (DICOM→NIfTI, RAS) |
| Group A — Person 2 | Preprocessing Pipeline | `preprocess_mri.py` (Skull Strip, Normalization) |
| Group B — Person 3 | Dataset Splitting & Loader | `dataset.py` (splits, 2.5D triplets) |
| Group B — Person 4 | Model Training & Web App | `model.py`, `train.py`, `app.py` |

---

## 📝 Note on Model Weights & Dataset

> ⚠️ The trained model (`data/best_resnet18.pth`) and raw MRI NIfTI files are **NOT included** in this repository due to file size limits.
> Contact the team for access to the pretrained weights or download the PPMI dataset from [ppmi-info.org](https://www.ppmi-info.org).
