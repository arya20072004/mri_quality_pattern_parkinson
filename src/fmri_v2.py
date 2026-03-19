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
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
BASE     = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH = os.path.join(BASE, "data", "labels", "fmri_dataset.csv")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
LOW_FREQ  = 0.01
HIGH_FREQ = 0.10
N_PARCELS = 20   # data-driven brain parcels
# ────────────────────────────────────────────────────────────────


def bandpass_filter(ts, tr, low=0.01, high=0.10):
    nyq   = 0.5 / tr
    lo, hi = low/nyq, high/nyq
    lo = max(0.001, min(lo, 0.49))
    hi = max(0.001, min(hi, 0.49))
    if lo >= hi: return ts
    b, a = signal.butter(4, [lo, hi], btype='band')
    return signal.filtfilt(b, a, ts)


def extract_features(fmri_path, tr, n_parcels=20):
    img  = nib.load(fmri_path)
    data = img.get_fdata().astype(np.float32)
    X, Y, Z, T = data.shape

    # ── Brain mask: voxels with sufficient signal ────────────────
    mean_vol  = data.mean(axis=3)
    threshold = np.percentile(mean_vol[mean_vol > 0], 30)
    brain_mask = mean_vol > threshold   # (X, Y, Z)
    n_voxels   = brain_mask.sum()

    # ── Extract brain voxel timeseries ───────────────────────────
    voxel_ts = data[brain_mask]   # (n_voxels, T)

    # Detrend + normalize each voxel
    from scipy.signal import detrend as sp_detrend
    voxel_ts = sp_detrend(voxel_ts, axis=1)
    std = voxel_ts.std(axis=1, keepdims=True) + 1e-8
    voxel_ts = voxel_ts / std

    # Bandpass filter each voxel
    filtered = np.zeros_like(voxel_ts)
    for i in range(n_voxels):
        filtered[i] = bandpass_filter(voxel_ts[i], tr, LOW_FREQ, HIGH_FREQ)

    # ── Data-driven parcellation via k-means on voxel coords ─────
    # Group voxels by spatial location into N_PARCELS regions
    # This is scanner-space parcellation — no MNI needed
    from sklearn.cluster import MiniBatchKMeans
    coords = np.array(np.where(brain_mask)).T   # (n_voxels, 3)

    # Normalize coords to 0-1
    coords_norm = coords / coords.max(axis=0)

    kmeans  = MiniBatchKMeans(n_clusters=n_parcels, random_state=42,
                               n_init=3, max_iter=100)
    parcel_labels = kmeans.fit_predict(coords_norm)

    # Mean timeseries per parcel
    parcel_ts = np.zeros((n_parcels, T))
    for p in range(n_parcels):
        idx = parcel_labels == p
        if idx.sum() > 0:
            parcel_ts[p] = filtered[idx].mean(axis=0)

    # ── Functional connectivity matrix ───────────────────────────
    fc = np.corrcoef(parcel_ts)   # (n_parcels, n_parcels)
    fc = np.arctanh(np.clip(fc, -0.999, 0.999))   # Fisher Z

    # Upper triangle features
    idx_upper = np.triu_indices(n_parcels, k=1)
    fc_feats  = fc[idx_upper]   # n_parcels*(n_parcels-1)/2 features

    # ── Additional features: regional BOLD variance ──────────────
    # Variance of each parcel's timeseries (measure of activity)
    variance_feats = parcel_ts.var(axis=1)

    # ── Mean amplitude of low-frequency fluctuations (mALFF) ─────
    # Power in resting-state band normalized by total power
    alff_feats = np.zeros(n_parcels)
    for p in range(n_parcels):
        ts    = parcel_ts[p]
        freqs = np.fft.rfftfreq(T, d=tr)
        power = np.abs(np.fft.rfft(ts))**2
        rs_mask = (freqs >= LOW_FREQ) & (freqs <= HIGH_FREQ)
        if power.sum() > 0:
            alff_feats[p] = power[rs_mask].sum() / power.sum()

    all_features = np.concatenate([fc_feats, variance_feats, alff_feats])
    return all_features


# ── EXTRACT FEATURES ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Extracting features for {len(df)} subjects...\n")

X_list, y_list, subj_list = [], [], []

for _, row in df.iterrows():
    subj  = row["subject"]
    label = int(row["label"])
    tr    = float(row["tr"])

    try:
        feats = extract_features(row["fmri_path"], tr, N_PARCELS)
        X_list.append(feats)
        y_list.append(label)
        subj_list.append(subj)
        print(f"  [OK] {subj:30s} | label={label} | features={feats.shape[0]}")
    except Exception as e:
        print(f"  [ERROR] {subj} → {e}")

X = np.array(X_list)
y = np.array(y_list)
print(f"\nFeature matrix: {X.shape}")
print(f"Controls: {(y==0).sum()} | Patients: {(y==1).sum()}\n")

# Save features for reuse
np.save(os.path.join(BASE, "data", "fmri_features_X.npy"), X)
np.save(os.path.join(BASE, "data", "fmri_features_y.npy"), y)
print("Features saved to data/fmri_features_X.npy\n")

# ── CLASSIFY ─────────────────────────────────────────────────────
print("="*55)
print("CLASSIFICATION")
print("="*55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classifiers = {
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=50)),
        ("clf",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=50)),
        ("clf",    RandomForestClassifier(n_estimators=500, max_depth=5,
                                          min_samples_leaf=3, random_state=42)),
    ]),
    "SVM (Linear)": Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=50)),
        ("clf",    SVC(kernel="linear", C=0.1,
                       probability=True, random_state=42)),
    ]),
}

results = {}
for name, clf in classifiers.items():
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    results[name] = {"acc": acc, "auc": auc}
    print(f"\n{name}:")
    print(f"  Accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"  AUC-ROC:  {auc.mean():.3f} ± {auc.std():.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in acc]}")

# ── BEST MODEL DETAILED REPORT ────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"].mean())
print(f"\n{'='*55}")
print(f"BEST MODEL: {best_name}")
print(f"{'='*55}\n")

best_clf = classifiers[best_name]
y_pred  = cross_val_predict(best_clf, X, y, cv=cv)
y_proba = cross_val_predict(best_clf, X, y, cv=cv, method="predict_proba")[:,1]

print(classification_report(y, y_pred, target_names=["Control","Patient"]))
print(f"AUC-ROC: {roc_auc_score(y, y_proba):.4f}")

cm = confusion_matrix(y, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, y_proba)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color="steelblue", lw=2,
         label=f"AUC = {roc_auc_score(y, y_proba):.3f}")
plt.plot([0,1],[0,1],"k--",lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve — {best_name} (fMRI Connectivity)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "fmri_roc.png"), dpi=150)
plt.close()

import joblib
best_clf.fit(X, y)
joblib.dump(best_clf, os.path.join(BASE, "models", "fmri_classifier_v2.pkl"))
print(f"\nModel saved → models/fmri_classifier_v2.pkl")
print(f"ROC curve  → outputs/fmri_roc.png")

# In[ ]:



