# 🧠 MRI Scan Quality Assessment & Pattern Analysis System for Parkinson's Disease

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A complete two-module pipeline for automated MRI analysis:
1. **Quality Assessment** — evaluates MRI scan quality before analysis
2. **Pattern Analysis** — detects Parkinson's Disease using fMRI functional connectivity

---

## 🖥️ Demo

![Pipeline Demo](outputs/demo_screenshot.png)

**Pipeline Accuracy: 77.1% on 83 subjects**

---

## 🏗️ System Architecture
```
Raw MRI (.nii.gz)
      │
      ▼
┌─────────────────────┐
│  MODULE 1           │
│  Quality Assessment │──── Grade: A/B/C/FAIL
│  SNR, CNR,          │──── Score: 0-100
│  Sharpness, FOV,    │
│  Uniformity,        │
│  Ghosting           │
└────────┬────────────┘
         │ PASS
         ▼
┌─────────────────────┐
│  MODULE 2           │
│  Pattern Analysis   │──── Prediction: PD / HC
│  fMRI Connectivity  │──── Probability: 0-100%
│  15-parcel brain    │──── Confidence: High/Med/Low
│  FC + mALFF + ReHo  │
└─────────────────────┘
```

---

## 📁 Project Structure
```
mri_quality_pattern_parkinson/
├── src/
│   ├── build_dataset.py          # Build label CSV from raw data
│   ├── preprocess.py             # T1 MRI preprocessing pipeline
│   ├── eda.py                    # Exploratory data analysis
│   ├── extract_slices.py         # Extract 2D slices from 3D MRI
│   ├── build_ntua_dataset.py     # NTUA dataset pipeline
│   ├── quality_assessment.py     # Standalone QA module
│   ├── fmri_connectivity.py      # fMRI feature extraction (v1)
│   ├── fmri_v2.py                # fMRI feature extraction (v2)
│   ├── fmri_neurocon.py          # NEUROCON-specific classifier
│   ├── train_production.py       # Train production model
│   ├── check_fmri.py             # fMRI file validation
│   ├── fix_taowu_fmri.py         # TaoWu format fix utility
│   └── pipeline.py               # Unified inference pipeline
│
├── data/
│   └── labels/
│       ├── dataset.csv           # Subject metadata + T1 paths
│       ├── fmri_dataset.csv      # fMRI file registry
│       ├── slices_dataset.csv    # 2D slice registry
│       └── ntua_slices.csv       # NTUA slice registry
│
├── models/                       # Saved models (not in repo)
│   ├── production_classifier.pkl
│   ├── production_parcellator.pkl
│   └── production_config.json
│
├── outputs/                      # Generated outputs (not in repo)
│
├── app.py                        # Streamlit web application
├── requirements.txt
└── README.md
```

---

## 📊 Datasets

| Dataset | Subjects | HC | PD | Modality | Source |
|---------|----------|----|----|----------|--------|
| NEUROCON | 43 | 16 | 27 | T1 + fMRI | [NITRC](https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html) |
| TaoWu | 40 | 20 | 20 | T1 + fMRI | [NITRC](https://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html) |
| NTUA | 77 | 23 | 54 | MRI + DaT | [GitHub](https://github.com/ails-lab/ntua-parkinson-dataset) |
| **Total** | **160** | **59** | **101** | | |

> **Note:** Datasets are not included in this repository due to size.
> Download instructions below.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/mri-parkinsons-analysis.git
cd mri-parkinsons-analysis
```

### 2. Create virtual environment
```bash
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyTorch with CUDA (if you have NVIDIA GPU)
```bash
# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## 📥 Dataset Download

### NEUROCON + TaoWu (free, no registration)
```bash
pip install awscli
aws s3 sync s3://fcp-indi/data/Projects/NEUROCON/ neurocon_raw/ --no-sign-request
aws s3 sync s3://fcp-indi/data/Projects/TaoWu/ taowu_raw/ --no-sign-request
```

### NTUA Dataset
```bash
git clone https://github.com/ails-lab/ntua-parkinson-dataset.git
# On Windows, run the extraction script to handle special characters:
python extract_ntua.py
```

---

## 🚀 Quick Start

### Step 1 — Build dataset
```bash
python src/build_dataset.py
```

### Step 2 — Preprocess T1 MRI scans
```bash
python src/preprocess.py
```

### Step 3 — Fix TaoWu fMRI format
```bash
python src/fix_taowu_fmri.py
```

### Step 4 — Train production model
```bash
python src/train_production.py
```

### Step 5 — Run full pipeline on all subjects
```bash
python src/pipeline.py
```

### Step 6 — Launch Streamlit app
```bash
streamlit run app.py
```

---

## 📈 Results

### Quality Assessment
| Metric | Description | Threshold |
|--------|-------------|-----------|
| SNR | Signal-to-Noise Ratio | >15 = good |
| CNR | Contrast-to-Noise Ratio | >5 = good |
| Sharpness | Edge gradient magnitude | >60 = good |
| FOV Coverage | Brain volume coverage | >70% = good |
| Uniformity | Intensity uniformity | >50 = good |
| Ghosting | Artifact level | <0.1 = good |

**Grade Distribution (83 subjects):**
- Grade A: majority of scans
- Grade B: some scans
- FAIL: 0 scans

### Pattern Analysis (fMRI Connectivity)
| Metric | Value |
|--------|-------|
| Dataset | NEUROCON (43 subjects) |
| Features | 135 (FC + mALFF + ReHo) |
| Classifier | Gradient Boosting |
| CV Accuracy | 64.7% ± 8.9% |
| Sensitivity | 74.1% (PD detection) |
| Pipeline Accuracy | **77.1%** (83 subjects) |

---

## 🔬 Methods

### Quality Assessment
Computes 6 metrics from raw T1 MRI:
- **SNR** — mean signal / std of background noise
- **CNR** — |mean(WM) - mean(GM)| / std(background)
- **Sharpness** — mean gradient magnitude (motion proxy)
- **FOV Coverage** — estimated brain volume vs expected
- **Uniformity** — 100 - coefficient of variation × 100
- **Ghosting** — std(outside brain) / mean(inside brain)

### fMRI Connectivity Pipeline
1. Brain masking (30th percentile threshold)
2. Linear detrending + z-score normalization
3. Bandpass filtering (0.01–0.10 Hz)
4. Data-driven parcellation (k-means, 15 parcels)
5. Functional connectivity matrix (Fisher Z-transformed Pearson r)
6. mALFF (amplitude of low-frequency fluctuations)
7. ReHo proxy (regional homogeneity)
8. Gradient Boosting classification (SelectKBest top-40 features)

---

## 🖥️ Streamlit App Features

- **Upload & Analyze** — upload T1 + fMRI → instant quality + PD prediction
- **Dataset Results** — visualize all 83 subjects with grade distribution
- **Interactive FC matrix** — plotly heatmap of brain connectivity
- **BOLD timeseries** — resting-state signal visualization
- **PD Risk gauge** — intuitive probability display

---

## ⚠️ Disclaimer

This system is for **research purposes only** and is **not intended for clinical diagnosis**.
All predictions should be interpreted by qualified medical professionals.

---

## 👤 Author

**Arya** — Computer Engineering (AI & Data Science), MIT-WPU Pune

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 📚 References

1. NEUROCON/TaoWu datasets — NITRC INDI Parkinson's Initiative
2. NTUA Parkinson Dataset — ails-lab, NTUA
3. Esteban et al. (2017) — MRIQC: Advancing the automatic prediction of image quality in MRI
4. Skidmore et al. — Reliability analysis of the resting state can sensitively and specifically identify the presence of Parkinson disease