#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy.signal import detrend as sp_detrend
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib, warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
BASE      = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH  = os.path.join(BASE, "data", "labels", "fmri_dataset.csv")
MODEL_DIR = os.path.join(BASE, "models")
TR        = 3.48
LOW_F     = 0.01
HIGH_F    = 0.10
N_PARCELS = 15
# ────────────────────────────────────────────────────────────────


def bandpass(ts, tr, lo=0.01, hi=0.10):
    nyq = 0.5 / tr
    l, h = lo/nyq, hi/nyq
    l = max(0.001, min(l, 0.49))
    h = max(0.001, min(h, 0.49))
    if l >= h: return ts
    b, a = signal.butter(4, [l, h], btype='band')
    return signal.filtfilt(b, a, ts)


def extract_features(fmri_path, n_parcels=15, km_model=None):
    """
    Extract fMRI connectivity features.
    If km_model is provided, use existing parcellation (for inference).
    Returns features and km_model.
    """
    img  = nib.load(fmri_path)
    data = img.get_fdata().astype(np.float32)
    X, Y, Z, T = data.shape

    # Brain mask
    mean_vol   = data.mean(axis=3)
    thresh     = np.percentile(mean_vol[mean_vol > 0], 30)
    brain_mask = mean_vol > thresh
    n_vox      = brain_mask.sum()

    # Voxel timeseries
    vts = data[brain_mask]
    vts = sp_detrend(vts, axis=1)
    vts = vts / (vts.std(axis=1, keepdims=True) + 1e-8)
    filtered = np.array([bandpass(vts[i], TR, LOW_F, HIGH_F)
                         for i in range(n_vox)])

    # Parcellation
    coords     = np.array(np.where(brain_mask)).T.astype(float)
    coords    /= coords.max(axis=0) + 1e-8

    if km_model is None:
        km_model = MiniBatchKMeans(n_clusters=n_parcels, random_state=42,
                                   n_init=5, max_iter=200)
        labels = km_model.fit_predict(coords)
    else:
        labels = km_model.predict(coords)

    # Parcel timeseries
    parcel_ts = np.zeros((n_parcels, T))
    for p in range(n_parcels):
        idx = labels == p
        if idx.sum() > 0:
            parcel_ts[p] = filtered[idx].mean(axis=0)

    # FC features
    fc      = np.corrcoef(parcel_ts)
    fc      = np.arctanh(np.clip(fc, -0.999, 0.999))
    fc_feats= fc[np.triu_indices(n_parcels, k=1)]

    # mALFF
    alff = np.zeros(n_parcels)
    for p in range(n_parcels):
        freqs = np.fft.rfftfreq(T, d=TR)
        power = np.abs(np.fft.rfft(parcel_ts[p]))**2
        rs    = (freqs >= LOW_F) & (freqs <= HIGH_F)
        if power.sum() > 0:
            alff[p] = power[rs].sum() / power.sum()

    # ReHo proxy
    reho = np.zeros(n_parcels)
    for p in range(n_parcels):
        idx = labels == p
        if idx.sum() > 1:
            c       = np.corrcoef(filtered[idx])
            reho[p] = c[np.triu_indices(len(c), k=1)].mean()

    return np.concatenate([fc_feats, alff, reho]), km_model


# ── EXTRACT ALL NEUROCON FEATURES ────────────────────────────────
df       = pd.read_csv(CSV_PATH)
df_neuro = df[df["dataset"] == "neurocon"].reset_index(drop=True)

print(f"Training production model on ALL {len(df_neuro)} NEUROCON subjects\n")

X_list, y_list = [], []
km_global = None   # shared parcellation across all subjects

for _, row in df_neuro.iterrows():
    subj  = row["subject"]
    label = int(row["label"])
    try:
        feats, km_global = extract_features(
            row["fmri_path"], N_PARCELS, km_global)
        X_list.append(feats)
        y_list.append(label)
        print(f"  [OK] {subj} | label={label}")
    except Exception as e:
        print(f"  [ERROR] {subj} → {e}")

X = np.array(X_list)
y = np.array(y_list)
print(f"\nFeature matrix: {X.shape}")

# ── CROSS-VALIDATE FIRST (honest estimate) ───────────────────────
print("\nFinal cross-validation estimate:")
pipe = Pipeline([
    ("sc",  StandardScaler()),
    ("sel", SelectKBest(f_classif, k=40)),
    ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=2,
                learning_rate=0.05, random_state=42)),
])
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc  = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
auc  = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(f"  Accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
print(f"  AUC-ROC:  {auc.mean():.3f} ± {auc.std():.3f}")

# ── TRAIN ON ALL DATA ─────────────────────────────────────────────
print("\nFitting production model on all 43 subjects...")
pipe.fit(X, y)

# ── SAVE EVERYTHING NEEDED FOR INFERENCE ─────────────────────────
joblib.dump(pipe,      os.path.join(MODEL_DIR, "production_classifier.pkl"))
joblib.dump(km_global, os.path.join(MODEL_DIR, "production_parcellator.pkl"))

# Save feature config
config = {
    "n_parcels":  N_PARCELS,
    "tr":         TR,
    "low_f":      LOW_F,
    "high_f":     HIGH_F,
    "n_features": X.shape[1],
}
import json
with open(os.path.join(MODEL_DIR, "production_config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*50}")
print(f"Production model saved:")
print(f"  models/production_classifier.pkl")
print(f"  models/production_parcellator.pkl")
print(f"  models/production_config.json")
print(f"{'='*50}")

# In[ ]:



