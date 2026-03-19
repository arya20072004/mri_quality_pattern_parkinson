#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
BASE     = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH = os.path.join(BASE, "data", "labels", "fmri_dataset.csv")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# fMRI parameters
NEUROCON_TR = 3.48   # seconds
TAOWU_TR    = 2.00   # approximately same
LOW_FREQ    = 0.01   # Hz — resting state band
HIGH_FREQ   = 0.10   # Hz
# ────────────────────────────────────────────────────────────────


# ── STEP 1: Define Brain ROIs (Regions of Interest) ──────────────
# We use simple anatomical ROIs based on MNI coordinates
# mapped to our fMRI voxel space (64x64x27 or 64x64x28)
# These target PD-relevant networks: basal ganglia, motor, default mode

def get_rois(z_slices):
    """
    Define ROI masks in voxel space.
    fMRI is 64x64xZ — we define spherical ROIs around key regions.
    Coordinates are in voxel space (not MNI).
    """
    rois = {
        # Basal Ganglia (most affected in PD)
        "left_putamen":       (22, 32, int(z_slices*0.45)),
        "right_putamen":      (42, 32, int(z_slices*0.45)),
        "left_caudate":       (24, 38, int(z_slices*0.55)),
        "right_caudate":      (40, 38, int(z_slices*0.55)),

        # Motor cortex
        "left_motor":         (20, 20, int(z_slices*0.80)),
        "right_motor":        (44, 20, int(z_slices*0.80)),

        # Default Mode Network
        "medial_prefrontal":  (32, 42, int(z_slices*0.75)),
        "posterior_cingulate":(32, 26, int(z_slices*0.60)),
        "left_angular":       (18, 22, int(z_slices*0.70)),
        "right_angular":      (46, 22, int(z_slices*0.70)),

        # Cerebellum
        "left_cerebellum":    (22, 18, int(z_slices*0.15)),
        "right_cerebellum":   (42, 18, int(z_slices*0.15)),

        # Thalamus
        "left_thalamus":      (26, 30, int(z_slices*0.50)),
        "right_thalamus":     (38, 30, int(z_slices*0.50)),
    }
    return rois


def extract_roi_timeseries(fmri_data, roi_center, radius=3):
    """
    Extract mean timeseries from a spherical ROI.
    fmri_data: (X, Y, Z, T)
    roi_center: (x, y, z) in voxel coordinates
    """
    x0, y0, z0 = roi_center
    X, Y, Z, T = fmri_data.shape

    mask = np.zeros((X, Y, Z), dtype=bool)
    for x in range(max(0, x0-radius), min(X, x0+radius+1)):
        for y in range(max(0, y0-radius), min(Y, y0+radius+1)):
            for z in range(max(0, z0-radius), min(Z, z0+radius+1)):
                if (x-x0)**2 + (y-y0)**2 + (z-z0)**2 <= radius**2:
                    mask[x, y, z] = True

    # Mean signal across ROI voxels over time
    roi_voxels = fmri_data[mask]  # (n_voxels, T)
    if roi_voxels.shape[0] == 0:
        return np.zeros(T)
    return roi_voxels.mean(axis=0)  # (T,)


def bandpass_filter(timeseries, tr, low=0.01, high=0.10):
    """Apply bandpass filter to isolate resting-state frequencies."""
    nyq    = 0.5 / tr
    low_n  = low  / nyq
    high_n = high / nyq
    low_n  = max(0.001, min(low_n,  0.99))
    high_n = max(0.001, min(high_n, 0.99))
    if low_n >= high_n:
        return timeseries
    b, a = signal.butter(4, [low_n, high_n], btype='band')
    return signal.filtfilt(b, a, timeseries)


def extract_connectivity_features(fmri_path, tr, subject):
    """
    Full pipeline: load fMRI → extract ROI timeseries →
    bandpass filter → compute connectivity matrix → flatten upper triangle
    """
    img      = nib.load(fmri_path)
    data     = img.get_fdata().astype(np.float32)
    Z        = data.shape[2]

    # Detrend (remove linear drift)
    from scipy.signal import detrend
    data = detrend(data, axis=3)

    # Normalize each voxel timeseries
    mean = data.mean(axis=3, keepdims=True)
    std  = data.std(axis=3, keepdims=True) + 1e-8
    data = (data - mean) / std

    rois      = get_rois(Z)
    roi_names = list(rois.keys())
    n_rois    = len(roi_names)

    # Extract and filter timeseries for each ROI
    timeseries = {}
    for name, center in rois.items():
        ts = extract_roi_timeseries(data, center, radius=3)
        ts = bandpass_filter(ts, tr, LOW_FREQ, HIGH_FREQ)
        timeseries[name] = ts

    # Compute functional connectivity (Pearson correlation matrix)
    fc_matrix = np.zeros((n_rois, n_rois))
    for i, roi_i in enumerate(roi_names):
        for j, roi_j in enumerate(roi_names):
            if i == j:
                fc_matrix[i, j] = 1.0
            elif i < j:
                r, _ = pearsonr(timeseries[roi_i], timeseries[roi_j])
                fc_matrix[i, j] = r
                fc_matrix[j, i] = r

    # Fisher Z-transform for normalization
    fc_z = np.arctanh(np.clip(fc_matrix, -0.999, 0.999))

    # Extract upper triangle as feature vector (excludes diagonal)
    upper_idx = np.triu_indices(n_rois, k=1)
    features  = fc_z[upper_idx]  # n_rois*(n_rois-1)/2 = 91 features

    return features, fc_matrix, roi_names


# ── STEP 2: Extract Features for All Subjects ────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Extracting fMRI connectivity features for {len(df)} subjects...\n")

X_list, y_list, subjects_list = [], [], []
fc_matrices = {}

for _, row in df.iterrows():
    subj     = row["subject"]
    label    = int(row["label"])
    fmri_path= row["fmri_path"]
    dataset  = row["dataset"]
    tr       = NEUROCON_TR if dataset == "neurocon" else TAOWU_TR

    try:
        features, fc_mat, roi_names = extract_connectivity_features(
            fmri_path, tr, subj)
        X_list.append(features)
        y_list.append(label)
        subjects_list.append(subj)
        fc_matrices[subj] = fc_mat
        print(f"  [OK] {subj:30s} | label={label} | features={features.shape[0]}")
    except Exception as e:
        print(f"  [ERROR] {subj} → {e}")

X = np.array(X_list)
y = np.array(y_list)
print(f"\nFeature matrix: {X.shape}  (subjects × connectivity_features)")
print(f"Controls: {(y==0).sum()} | Patients: {(y==1).sum()}\n")


# ── STEP 3: Visualize Mean FC Matrix ─────────────────────────────
mean_fc_pd = np.mean([fc_matrices[s] for s, l in
                      zip(subjects_list, y_list) if l == 1], axis=0)
mean_fc_hc = np.mean([fc_matrices[s] for s, l in
                      zip(subjects_list, y_list) if l == 0], axis=0)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Functional Connectivity Matrices", fontsize=13)

im0 = axes[0].imshow(mean_fc_hc, cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_title("Mean FC — Healthy Controls")
axes[0].set_xticks(range(len(roi_names)))
axes[0].set_yticks(range(len(roi_names)))
axes[0].set_xticklabels(roi_names, rotation=90, fontsize=7)
axes[0].set_yticklabels(roi_names, fontsize=7)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(mean_fc_pd, cmap="RdBu_r", vmin=-1, vmax=1)
axes[1].set_title("Mean FC — PD Patients")
axes[1].set_xticks(range(len(roi_names)))
axes[1].set_yticks(range(len(roi_names)))
axes[1].set_xticklabels(roi_names, rotation=90, fontsize=7)
axes[1].set_yticklabels(roi_names, fontsize=7)
plt.colorbar(im1, ax=axes[1])

diff = mean_fc_pd - mean_fc_hc
im2  = axes[2].imshow(diff, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
axes[2].set_title("Difference (PD - HC)")
axes[2].set_xticks(range(len(roi_names)))
axes[2].set_yticks(range(len(roi_names)))
axes[2].set_xticklabels(roi_names, rotation=90, fontsize=7)
axes[2].set_yticklabels(roi_names, fontsize=7)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fc_matrices.png"), dpi=150)
plt.close()
print("FC matrix visualization saved → outputs/fc_matrices.png\n")


# ── STEP 4: Train Classifiers with Cross-Validation ──────────────
print("="*55)
print("CLASSIFICATION WITH FUNCTIONAL CONNECTIVITY FEATURES")
print("="*55)

# Feature names for interpretability
n_rois    = len(roi_names)
feat_names = []
for i in range(n_rois):
    for j in range(i+1, n_rois):
        feat_names.append(f"{roi_names[i]}↔{roi_names[j]}")

# Models to compare
classifiers = {
    "SVM (RBF)":        Pipeline([
                            ("scaler",  StandardScaler()),
                            ("select",  SelectKBest(f_classif, k=30)),
                            ("clf",     SVC(kernel="rbf", C=1.0,
                                           probability=True, random_state=42))
                        ]),
    "Random Forest":    Pipeline([
                            ("scaler",  StandardScaler()),
                            ("select",  SelectKBest(f_classif, k=30)),
                            ("clf",     RandomForestClassifier(
                                            n_estimators=200, max_depth=4,
                                            random_state=42))
                        ]),
    "Gradient Boosting":Pipeline([
                            ("scaler",  StandardScaler()),
                            ("select",  SelectKBest(f_classif, k=30)),
                            ("clf",     GradientBoostingClassifier(
                                            n_estimators=100, max_depth=3,
                                            learning_rate=0.05, random_state=42))
                        ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, clf in classifiers.items():
    acc_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    results[name] = {"acc": acc_scores, "auc": auc_scores}
    print(f"\n{name}:")
    print(f"  Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"  AUC-ROC:  {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"  Per-fold acc: {[f'{s:.3f}' for s in acc_scores]}")

# ── STEP 5: Best Model — Full Evaluation ─────────────────────────
print(f"\n{'='*55}")
print("DETAILED EVALUATION — BEST MODEL")
print(f"{'='*55}\n")

# Find best model by AUC
best_name = max(results, key=lambda k: results[k]["auc"].mean())
print(f"Best model: {best_name}\n")

best_clf = classifiers[best_name]
best_clf.fit(X, y)

# Leave-one-out style detailed report
from sklearn.model_selection import cross_val_predict
y_pred  = cross_val_predict(best_clf, X, y, cv=cv)
y_proba = cross_val_predict(best_clf, X, y, cv=cv, method="predict_proba")[:,1]

print(classification_report(y, y_pred, target_names=["Control","Patient"]))
print(f"AUC-ROC: {roc_auc_score(y, y_proba):.4f}")

cm = confusion_matrix(y, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# Save feature importance (Random Forest)
if "Random Forest" in classifiers:
    rf_pipe = classifiers["Random Forest"]
    rf_pipe.fit(X, y)
    selector    = rf_pipe.named_steps["select"]
    rf          = rf_pipe.named_steps["clf"]
    selected_idx= selector.get_support(indices=True)
    selected_names = [feat_names[i] for i in selected_idx]
    importances = rf.feature_importances_

    top_idx  = np.argsort(importances)[::-1][:15]
    top_feats= [selected_names[i] for i in top_idx]
    top_vals = importances[top_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_feats[::-1], top_vals[::-1], color="steelblue")
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Connectivity Features (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"\nTop 5 most discriminative connections:")
    for i in range(5):
        print(f"  {top_feats[i]:50s} importance={top_vals[i]:.4f}")

import joblib
joblib.dump(best_clf, os.path.join(BASE, "models", "fmri_classifier.pkl"))
print(f"\nModel saved → models/fmri_classifier.pkl")

# In[ ]:



