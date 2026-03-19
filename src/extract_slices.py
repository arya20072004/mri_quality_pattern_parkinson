#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────────
BASE       = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH   = os.path.join(BASE, "data", "labels", "dataset.csv")
SLICES_DIR = os.path.join(BASE, "data", "slices")
N_SLICES   = 30   # slices per plane (center 30 of 128)
# ────────────────────────────────────────────────────────────────

os.makedirs(SLICES_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)
df = df[df["processed_path"].notna()].reset_index(drop=True)

slice_records = []

for _, row in df.iterrows():
    arr     = np.load(row["processed_path"])   # (128,128,128)
    subject = row["subject"]
    dataset = row["dataset"]
    label   = int(row["label"])
    mid     = arr.shape[0] // 2                # 64

    # Extract center N_SLICES from each plane
    planes = {
        "axial":    [arr[:, :, mid - N_SLICES//2 + i] for i in range(N_SLICES)],
        "coronal":  [arr[:, mid - N_SLICES//2 + i, :] for i in range(N_SLICES)],
        "sagittal": [arr[mid - N_SLICES//2 + i, :, :] for i in range(N_SLICES)],
    }

    for plane_name, slices in planes.items():
        for i, sl in enumerate(slices):
            # Normalize to 0-255
            sl = sl.astype(np.float32)
            sl_min, sl_max = sl.min(), sl.max()
            if sl_max - sl_min > 0:
                sl = (sl - sl_min) / (sl_max - sl_min) * 255.0
            sl = sl.astype(np.uint8)

            # Convert to RGB (ResNet expects 3 channels)
            img = Image.fromarray(sl).convert("RGB")

            # Save
            fname = f"{dataset}_{subject}_{plane_name}_{i:03d}.png"
            fpath = os.path.join(SLICES_DIR, fname)
            img.save(fpath)

            slice_records.append({
                "subject":  subject,
                "dataset":  dataset,
                "label":    label,
                "plane":    plane_name,
                "slice_idx": i,
                "path":     fpath,
            })

    print(f"  [OK] {subject} | {dataset} | {N_SLICES*3} slices saved")

slices_df = pd.DataFrame(slice_records)
slices_csv = os.path.join(BASE, "data", "labels", "slices_dataset.csv")
slices_df.to_csv(slices_csv, index=False)

print(f"\n{'='*50}")
print(f"Total slices: {len(slices_df)}")
print(f"Controls:     {(slices_df['label']==0).sum()}")
print(f"Patients:     {(slices_df['label']==1).sum()}")
print(f"CSV saved  →  {slices_csv}")
print(f"{'='*50}")

# In[ ]:



