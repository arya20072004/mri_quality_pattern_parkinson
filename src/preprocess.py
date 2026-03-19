#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────
BASE        = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH    = os.path.join(BASE, "data", "labels", "dataset.csv")
OUTPUT_DIR  = os.path.join(BASE, "data", "processed")
TARGET_SIZE = (128, 128, 128)   # All scans will be resized to this
TARGET_SPACING = (1.5, 1.5, 1.5)  # mm per voxel
# ────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_volume(sitk_img, new_spacing):
    """Resample image to a target voxel spacing."""
    orig_spacing = sitk_img.GetSpacing()
    orig_size    = sitk_img.GetSize()

    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(sitk_img)


def resize_volume(arr, target_size):
    """Resize 3D numpy array to target_size using zoom."""
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target_size, arr.shape)]
    return zoom(arr, factors, order=1)


def normalize(arr):
    """Z-score normalization — zero mean, unit std."""
    arr = arr.astype(np.float32)
    mean = arr.mean()
    std  = arr.std()
    if std == 0:
        return arr - mean
    return (arr - mean) / std


def preprocess_scan(t1_path, output_path):
    # Load with SimpleITK
    sitk_img = sitk.ReadImage(str(t1_path))

    # Step 1: Resample to uniform spacing
    sitk_img = resample_volume(sitk_img, TARGET_SPACING)

    # Step 2: Convert to numpy
    arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
    # sitk gives (Z, Y, X) → keep as is, resize handles it

    # Step 3: Resize to fixed shape
    arr = resize_volume(arr, TARGET_SIZE)

    # Step 4: Normalize
    arr = normalize(arr)

    # Step 5: Save as .npy (fast to load during training)
    np.save(output_path, arr)
    return arr.shape


# ── MAIN ────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
processed_paths = []
failed = []

print(f"Processing {len(df)} scans → target shape {TARGET_SIZE}\n")

for idx, row in df.iterrows():
    subject    = row["subject"]
    dataset    = row["dataset"]
    t1_path    = row["t1_path"]

    out_fname  = f"{dataset}_{subject}.npy"
    out_path   = os.path.join(OUTPUT_DIR, out_fname)

    if os.path.exists(out_path):
        print(f"  [SKIP - already done] {subject}")
        processed_paths.append(out_path)
        continue

    try:
        shape = preprocess_scan(t1_path, out_path)
        print(f"  [OK] {subject} | {dataset} | saved shape={shape}")
        processed_paths.append(out_path)
    except Exception as e:
        print(f"  [ERROR] {subject} → {e}")
        failed.append(subject)
        processed_paths.append(None)

# Update CSV with processed paths
df["processed_path"] = processed_paths
df.to_csv(CSV_PATH, index=False)

print("\n" + "="*50)
print(f"Done: {len(df) - len(failed)}/{len(df)} scans preprocessed")
print(f"Failed: {len(failed)} → {failed}")
print(f"CSV updated: {CSV_PATH}")
print("="*50)

# In[ ]:



