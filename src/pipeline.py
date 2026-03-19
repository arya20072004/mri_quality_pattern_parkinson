#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""
MRI Quality Assessment + Parkinson's Disease Prediction Pipeline
================================================================
Input:  Path to a raw T1 MRI (.nii.gz) and optionally fMRI (.nii.gz)
Output: Quality report + PD prediction with confidence
"""

import os
import json
import numpy as np
import nibabel as nib
import joblib
from scipy import signal
from scipy.signal import detrend as sp_detrend
from sklearn.cluster import MiniBatchKMeans

# ── CONFIG ──────────────────────────────────────────────────────
BASE      = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR   = os.path.join(BASE, "outputs", "pipeline_results")
os.makedirs(OUT_DIR, exist_ok=True)

# Load production models
CLASSIFIER   = joblib.load(os.path.join(MODEL_DIR, "production_classifier.pkl"))
PARCELLATOR  = joblib.load(os.path.join(MODEL_DIR, "production_parcellator.pkl"))
with open(os.path.join(MODEL_DIR, "production_config.json")) as f:
    CONFIG = json.load(f)

TR        = CONFIG["tr"]
LOW_F     = CONFIG["low_f"]
HIGH_F    = CONFIG["high_f"]
N_PARCELS = CONFIG["n_parcels"]
# ────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════════════════════════
# MODULE 1: QUALITY ASSESSMENT
# ════════════════════════════════════════════════════════════════

def compute_snr(arr):
    flat = arr.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) == 0: return 0.0
    signal_r = flat[flat > np.percentile(flat, 80)]
    noise_r  = flat[flat < np.percentile(flat, 10)]
    if len(noise_r) == 0 or np.std(noise_r) == 0: return 0.0
    return float(np.mean(signal_r) / (np.std(noise_r) + 1e-8))

def compute_cnr(arr):
    flat = arr.flatten()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if len(flat) == 0: return 0.0
    p25, p50, p75 = np.percentile(flat, [25, 50, 75])
    gm  = flat[(flat >= p25) & (flat < p50)]
    wm  = flat[flat >= p75]
    bg  = flat[flat < np.percentile(flat, 5)]
    if len(gm) == 0 or len(wm) == 0 or len(bg) == 0: return 0.0
    return float(abs(np.mean(wm) - np.mean(gm)) / (np.std(bg) + 1e-8))

def compute_sharpness(arr):
    gx = np.diff(arr, axis=0)
    gy = np.diff(arr, axis=1)
    gz = np.diff(arr, axis=2)
    grad = (np.mean(np.abs(gx)) + np.mean(np.abs(gy)) +
            np.mean(np.abs(gz))) / 3.0
    return float(min(100.0, grad * 50))

def compute_ghosting(arr):
    mid        = arr[:, :, arr.shape[2]//2]
    brain_mask = mid > np.percentile(mid, 60)
    outside    = mid[~brain_mask]
    inside     = mid[brain_mask]
    return float(np.std(outside) / (np.mean(inside) + 1e-8))

def compute_fov(arr, voxel_size):
    brain    = arr > np.percentile(arr, 20)
    edge     = (brain[0].any() or brain[-1].any() or
                brain[:,0,:].any() or brain[:,-1,:].any() or
                brain[:,:,0].any() or brain[:,:,-1].any())
    vol      = brain.sum() * np.prod(voxel_size)
    coverage = min(100.0, vol / 12000.0)
    if edge: coverage *= 0.8
    return float(coverage)

def compute_uniformity(arr):
    brain = arr[arr > np.percentile(arr, 30)]
    cv    = np.std(brain) / (np.mean(brain) + 1e-8)
    return float(max(0.0, 100.0 - cv * 100))

def quality_grade(snr, cnr, sharp, ghost, fov, unif):
    score  = min(30, snr/3)
    score += min(20, cnr/2)
    score += sharp * 0.25
    score += fov   * 0.15
    score += unif  * 0.10
    score -= ghost * 20
    score  = max(0, min(100, score))
    if score >= 75:   grade = "A"
    elif score >= 55: grade = "B"
    elif score >= 35: grade = "C"
    else:             grade = "FAIL"
    return round(score, 1), grade

def assess_quality(t1_path):
    """Run full quality assessment on T1 MRI."""
    img        = nib.load(t1_path)
    arr        = img.get_fdata().astype(np.float32)
    voxel_size = img.header.get_zooms()[:3]

    snr   = compute_snr(arr)
    cnr   = compute_cnr(arr)
    sharp = compute_sharpness(arr)
    ghost = compute_ghosting(arr)
    fov   = compute_fov(arr, voxel_size)
    unif  = compute_uniformity(arr)
    score, grade = quality_grade(snr, cnr, sharp, ghost, fov, unif)

    return {
        "snr":          round(snr, 3),
        "cnr":          round(cnr, 3),
        "sharpness":    round(sharp, 3),
        "ghosting":     round(ghost, 4),
        "fov_coverage": round(fov, 3),
        "uniformity":   round(unif, 3),
        "quality_score":score,
        "grade":        grade,
        "shape":        list(arr.shape),
        "voxel_size":   [round(v, 3) for v in voxel_size],
    }


# ════════════════════════════════════════════════════════════════
# MODULE 2: PD PREDICTION FROM fMRI
# ════════════════════════════════════════════════════════════════

def bandpass(ts, tr, lo, hi):
    nyq = 0.5 / tr
    l, h = lo/nyq, hi/nyq
    l = max(0.001, min(l, 0.49))
    h = max(0.001, min(h, 0.49))
    if l >= h: return ts
    b, a = signal.butter(4, [l, h], btype='band')
    return signal.filtfilt(b, a, ts)

def extract_fmri_features(fmri_path):
    """Extract connectivity features using production parcellator."""
    img  = nib.load(fmri_path)
    data = img.get_fdata().astype(np.float32)
    X, Y, Z, T = data.shape

    mean_vol   = data.mean(axis=3)
    thresh     = np.percentile(mean_vol[mean_vol > 0], 30)
    brain_mask = mean_vol > thresh
    n_vox      = brain_mask.sum()

    vts = data[brain_mask]
    vts = sp_detrend(vts, axis=1)
    vts = vts / (vts.std(axis=1, keepdims=True) + 1e-8)
    filtered = np.array([bandpass(vts[i], TR, LOW_F, HIGH_F)
                         for i in range(n_vox)])

    coords  = np.array(np.where(brain_mask)).T.astype(float)
    coords /= coords.max(axis=0) + 1e-8
    labels  = PARCELLATOR.predict(coords)

    parcel_ts = np.zeros((N_PARCELS, T))
    for p in range(N_PARCELS):
        idx = labels == p
        if idx.sum() > 0:
            parcel_ts[p] = filtered[idx].mean(axis=0)

    fc      = np.corrcoef(parcel_ts)
    fc      = np.arctanh(np.clip(fc, -0.999, 0.999))
    fc_feats= fc[np.triu_indices(N_PARCELS, k=1)]

    alff = np.zeros(N_PARCELS)
    for p in range(N_PARCELS):
        freqs = np.fft.rfftfreq(T, d=TR)
        power = np.abs(np.fft.rfft(parcel_ts[p]))**2
        rs    = (freqs >= LOW_F) & (freqs <= HIGH_F)
        if power.sum() > 0:
            alff[p] = power[rs].sum() / power.sum()

    reho = np.zeros(N_PARCELS)
    for p in range(N_PARCELS):
        idx = labels == p
        if idx.sum() > 1:
            c       = np.corrcoef(filtered[idx])
            reho[p] = c[np.triu_indices(len(c), k=1)].mean()

    return np.concatenate([fc_feats, alff, reho])

def predict_pd(fmri_path):
    """Run PD prediction from fMRI."""
    feats = extract_fmri_features(fmri_path)
    prob  = CLASSIFIER.predict_proba(feats.reshape(1, -1))[0]
    pred  = int(CLASSIFIER.predict(feats.reshape(1, -1))[0])
    return {
        "prediction":    "Parkinson's Disease" if pred == 1 else "Healthy Control",
        "label":         pred,
        "pd_probability":  round(float(prob[1]), 4),
        "hc_probability":  round(float(prob[0]), 4),
        "confidence":    "High" if max(prob) > 0.8 else
                         "Medium" if max(prob) > 0.6 else "Low",
    }


# ════════════════════════════════════════════════════════════════
# UNIFIED PIPELINE
# ════════════════════════════════════════════════════════════════

def run_pipeline(subject_id, t1_path, fmri_path=None):
    """
    Full pipeline:
    1. Quality Assessment on T1
    2. If quality passes → PD prediction from fMRI
    3. Return combined report
    """
    print(f"\n{'='*60}")
    print(f"  MRI ANALYSIS PIPELINE — {subject_id}")
    print(f"{'='*60}")

    report = {"subject": subject_id, "t1_path": t1_path}

    # ── MODULE 1: Quality Assessment ─────────────────────────────
    print("\n[1/2] Running Quality Assessment...")
    qa = assess_quality(t1_path)
    report["quality"] = qa

    grade_colors = {"A":"✅","B":"✅","C":"⚠️","FAIL":"❌"}
    icon = grade_colors.get(qa["grade"], "?")
    print(f"      Grade:   {icon} {qa['grade']}  "
          f"(Score: {qa['quality_score']}/100)")
    print(f"      SNR:     {qa['snr']:.2f}")
    print(f"      CNR:     {qa['cnr']:.2f}")
    print(f"      Shape:   {qa['shape']}")

    # ── Quality Gate ─────────────────────────────────────────────
    if qa["grade"] == "FAIL":
        report["prediction"] = None
        report["status"] = "REJECTED — Scan quality too low for analysis"
        print(f"\n      ❌ SCAN REJECTED — Quality insufficient")
        print(f"         Recommendation: Re-acquire MRI scan")
        return report

    # ── MODULE 2: PD Prediction ───────────────────────────────────
    if fmri_path is None:
        report["prediction"] = None
        report["status"] = "QUALITY PASSED — No fMRI provided for PD prediction"
        print(f"\n      ✅ Quality passed but no fMRI provided")
        return report

    print(f"\n[2/2] Running PD Prediction from fMRI...")
    try:
        pred = predict_pd(fmri_path)
        report["prediction"] = pred
        report["status"] = "COMPLETE"

        risk_icon = "🔴" if pred["label"] == 1 else "🟢"
        print(f"      {risk_icon} Prediction: {pred['prediction']}")
        print(f"         PD Probability:  {pred['pd_probability']:.1%}")
        print(f"         HC Probability:  {pred['hc_probability']:.1%}")
        print(f"         Confidence:      {pred['confidence']}")
    except Exception as e:
        report["prediction"] = None
        report["status"] = f"Prediction failed: {e}"
        print(f"      ⚠️  Prediction error: {e}")

    # ── Final Summary ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Quality:    Grade {qa['grade']} ({qa['quality_score']}/100)")
    if report.get("prediction"):
        print(f"  Diagnosis:  {pred['prediction']}")
        print(f"  PD Risk:    {pred['pd_probability']:.1%} "
              f"({pred['confidence']} confidence)")
    print(f"{'='*60}\n")

    return report


# ════════════════════════════════════════════════════════════════
# RUN ON ALL 83 SUBJECTS
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pandas as pd

    t1_csv   = os.path.join(BASE, "data", "labels", "dataset.csv")
    fmri_csv = os.path.join(BASE, "data", "labels", "fmri_dataset.csv")

    t1_df   = pd.read_csv(t1_csv)
    fmri_df = pd.read_csv(fmri_csv).set_index("subject")

    all_reports = []
    stats = {"total":0, "passed":0, "failed_qa":0,
             "predicted_pd":0, "predicted_hc":0, "no_fmri":0}

    print(f"Running pipeline on {len(t1_df)} subjects...\n")

    for _, row in t1_df.iterrows():
        subj     = row["subject"]
        t1_path  = row["t1_path"]
        true_lab = row["label"]

        # Get fMRI path if available
        fmri_path = None
        if subj in fmri_df.index:
            fp = fmri_df.loc[subj, "fmri_path"]
            if pd.notna(fp) and os.path.exists(str(fp)):
                fmri_path = str(fp)

        report = run_pipeline(subj, t1_path, fmri_path)
        report["true_label"] = true_lab
        all_reports.append(report)
        stats["total"] += 1

        qa_grade = report["quality"]["grade"]
        if qa_grade == "FAIL":
            stats["failed_qa"] += 1
        else:
            stats["passed"] += 1
            if report.get("prediction"):
                if report["prediction"]["label"] == 1:
                    stats["predicted_pd"] += 1
                else:
                    stats["predicted_hc"] += 1
            else:
                stats["no_fmri"] += 1

    # ── Summary Report ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIPELINE SUMMARY — ALL {stats['total']} SUBJECTS")
    print(f"{'='*60}")
    print(f"  Quality passed:    {stats['passed']}")
    print(f"  Quality failed:    {stats['failed_qa']}")
    print(f"  Predicted PD:      {stats['predicted_pd']}")
    print(f"  Predicted HC:      {stats['predicted_hc']}")
    print(f"  No fMRI available: {stats['no_fmri']}")

    # Accuracy on NEUROCON subjects that have fMRI predictions
    correct = sum(
        1 for r in all_reports
        if r.get("prediction") and
        r["prediction"]["label"] == r["true_label"]
    )
    predicted = sum(1 for r in all_reports if r.get("prediction"))
    if predicted > 0:
        print(f"\n  Pipeline Accuracy:  {correct}/{predicted} = "
              f"{correct/predicted:.1%}")
    print(f"{'='*60}")

    # Save JSON report
    import json
    out = os.path.join(OUT_DIR, "pipeline_results.json")
    with open(out, "w") as f:
        json.dump(all_reports, f, indent=2, default=str)
    print(f"\n  Full report saved → {out}")

# In[ ]:



