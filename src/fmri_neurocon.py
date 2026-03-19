#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import signal
from scipy.signal import detrend as sp_detrend
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import warnings, joblib
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────
BASE     = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH = os.path.join(BASE, "data", "labels", "fmri_dataset.csv")
OUT_DIR  = os.path.join(BASE, "outputs")
MODEL_DIR= os.path.join(BASE, "models")
TR       = 3.48
LOW_F    = 0.01
HIGH_F   = 0.10
N_PARCELS= 15
# ────────────────────────────────────────────────────────────────


def bandpass(ts, tr, lo=0.01, hi=0.10):
    nyq  = 0.5 / tr
    l, h = lo/nyq, hi/nyq
    l    = max(0.001, min(l, 0.49))
    h    = max(0.001, min(h, 0.49))
    if l >= h: return ts
    b, a = signal.butter(4, [l, h], btype='band')
    return signal.filtfilt(b, a, ts)


def extract_features(fmri_path, n_parcels=15):
    img  = nib.load(fmri_path)
    data = img.get_fdata().astype(np.float32)
    X, Y, Z, T = data.shape

    # Brain mask
    mean_vol   = data.mean(axis=3)
    thresh     = np.percentile(mean_vol[mean_vol > 0], 30)
    brain_mask = mean_vol > thresh
    n_vox      = brain_mask.sum()

    # Extract + preprocess voxel timeseries
    vts = data[brain_mask]                         # (n_vox, T)
    vts = sp_detrend(vts, axis=1)                  # linear detrend
    vts = vts / (vts.std(axis=1, keepdims=True) + 1e-8)  # normalize

    # Bandpass filter
    filtered = np.array([bandpass(vts[i], TR, LOW_F, HIGH_F)
                         for i in range(n_vox)])

    # Spatial parcellation (k-means on voxel coordinates)
    coords      = np.array(np.where(brain_mask)).T.astype(float)
    coords     /= coords.max(axis=0) + 1e-8
    km          = MiniBatchKMeans(n_clusters=n_parcels, random_state=42,
                                  n_init=5, max_iter=200)
    labels      = km.fit_predict(coords)

    # Parcel mean timeseries
    parcel_ts = np.zeros((n_parcels, T))
    for p in range(n_parcels):
        idx = labels == p
        if idx.sum() > 0:
            parcel_ts[p] = filtered[idx].mean(axis=0)

    # Functional connectivity (Fisher Z-transformed Pearson r)
    fc = np.corrcoef(parcel_ts)
    fc = np.arctanh(np.clip(fc, -0.999, 0.999))
    fc_feats = fc[np.triu_indices(n_parcels, k=1)]  # upper triangle

    # mALFF: amplitude of low-freq fluctuations per parcel
    alff = np.zeros(n_parcels)
    for p in range(n_parcels):
        freqs = np.fft.rfftfreq(T, d=TR)
        power = np.abs(np.fft.rfft(parcel_ts[p]))**2
        rs    = (freqs >= LOW_F) & (freqs <= HIGH_F)
        if power.sum() > 0:
            alff[p] = power[rs].sum() / power.sum()

    # Regional homogeneity proxy: mean correlation within parcel
    reho = np.zeros(n_parcels)
    for p in range(n_parcels):
        idx = labels == p
        if idx.sum() > 1:
            sub_ts = filtered[idx]
            c      = np.corrcoef(sub_ts)
            reho[p]= c[np.triu_indices(len(c), k=1)].mean()

    return np.concatenate([fc_feats, alff, reho])


# ── LOAD NEUROCON ONLY ───────────────────────────────────────────
df       = pd.read_csv(CSV_PATH)
df_neuro = df[df["dataset"] == "neurocon"].reset_index(drop=True)

print(f"NEUROCON fMRI subjects: {len(df_neuro)}")
print(f"Controls: {(df_neuro['label']==0).sum()} | "
      f"Patients: {(df_neuro['label']==1).sum()}\n")
print(f"Extracting features (TR={TR}s, {N_PARCELS} parcels)...\n")

X_list, y_list, subj_list = [], [], []

for _, row in df_neuro.iterrows():
    subj  = row["subject"]
    label = int(row["label"])
    try:
        feats = extract_features(row["fmri_path"], N_PARCELS)
        X_list.append(feats)
        y_list.append(label)
        subj_list.append(subj)
        print(f"  [OK] {subj:30s} | label={label} | "
              f"features={feats.shape[0]}")
    except Exception as e:
        print(f"  [ERROR] {subj} → {e}")

X = np.array(X_list)
y = np.array(y_list)
print(f"\nFeature matrix: {X.shape}")
print(f"Controls: {(y==0).sum()} | Patients: {(y==1).sum()}\n")


# ── CLASSIFY ─────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

pipelines = {
    "SVM Linear C=0.01": Pipeline([
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=40)),
        ("clf", SVC(kernel="linear", C=0.01,
                    probability=True, random_state=42)),
    ]),
    "SVM Linear C=0.1": Pipeline([
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=40)),
        ("clf", SVC(kernel="linear", C=0.1,
                    probability=True, random_state=42)),
    ]),
    "SVM RBF": Pipeline([
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=40)),
        ("clf", SVC(kernel="rbf", C=1, gamma="scale",
                    probability=True, random_state=42)),
    ]),
    "Random Forest": Pipeline([
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=40)),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=3,
                                       min_samples_leaf=3,
                                       random_state=42)),
    ]),
    "Gradient Boosting": Pipeline([
        ("sc", StandardScaler()),
        ("sel", SelectKBest(f_classif, k=40)),
        ("clf", GradientBoostingClassifier(n_estimators=100,
                                           max_depth=2,
                                           learning_rate=0.05,
                                           random_state=42)),
    ]),
}

print("="*55)
print("5-FOLD CROSS VALIDATION RESULTS")
print("="*55)

results = {}
for name, pipe in pipelines.items():
    acc = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    results[name] = {"acc": acc, "auc": auc, "pipe": pipe}
    print(f"\n{name}:")
    print(f"  Accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"  AUC-ROC:  {auc.mean():.3f} ± {auc.std():.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in acc]}")

# ── BEST MODEL ───────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["auc"].mean())
best_pipe = results[best_name]["pipe"]

print(f"\n{'='*55}")
print(f"BEST MODEL: {best_name}")
print(f"  AUC: {results[best_name]['auc'].mean():.3f} ± "
      f"{results[best_name]['auc'].std():.3f}")
print(f"{'='*55}\n")

y_pred  = cross_val_predict(best_pipe, X, y, cv=cv)
y_proba = cross_val_predict(best_pipe, X, y, cv=cv,
                             method="predict_proba")[:,1]

print(classification_report(y, y_pred,
                             target_names=["Control","Patient"]))
print(f"AUC-ROC: {roc_auc_score(y, y_proba):.4f}")

cm = confusion_matrix(y, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
print(f"  Sensitivity: {cm[1,1]/(cm[1,1]+cm[1,0]):.3f}")
print(f"  Specificity: {cm[0,0]/(cm[0,0]+cm[0,1]):.3f}")

# Per-subject probability table
print(f"\nPer-subject predictions:")
for i, subj in enumerate(subj_list):
    status = "✓" if y_pred[i] == y[i] else "✗"
    print(f"  {status} {subj:30s} | true={y[i]} "
          f"pred={y_pred[i]} prob={y_proba[i]:.3f}")

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y, y_proba)
auc_val     = roc_auc_score(y, y_proba)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color="steelblue", lw=2,
         label=f"AUC = {auc_val:.3f}")
plt.fill_between(fpr, tpr, alpha=0.1, color="steelblue")
plt.plot([0,1],[0,1],"k--",lw=1,label="Random")
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title(f"ROC Curve — {best_name}\nNEUROCON fMRI Connectivity")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig(os.path.join(OUT_DIR, "neurocon_fmri_roc.png"), dpi=150)
plt.close()

# Save model
best_pipe.fit(X, y)
joblib.dump(best_pipe,
    os.path.join(MODEL_DIR, "neurocon_fmri_classifier.pkl"))
np.save(os.path.join(BASE,"data","neurocon_X.npy"), X)
np.save(os.path.join(BASE,"data","neurocon_y.npy"), y)

print(f"\nModel saved → models/neurocon_fmri_classifier.pkl")
print(f"ROC curve  → outputs/neurocon_fmri_roc.png")

# In[ ]:



