#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import pandas as pd
from pathlib import Path
from PIL import Image

# ── CONFIG ──────────────────────────────────────────────────────
BASE      = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
NTUA_DIR  = os.path.join(BASE, "ntua-parkinson-dataset")
OUT_DIR   = os.path.join(BASE, "data", "slices_ntua")
os.makedirs(OUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────

GROUPS = {
    "PD Patients":     1,   # label = 1
    "Non PD Patients": 0,   # label = 0
}

records = []

for group_folder, label in GROUPS.items():
    group_path = os.path.join(NTUA_DIR, group_folder)
    subjects   = [s for s in os.listdir(group_path)
                  if os.path.isdir(os.path.join(group_path, s))]

    print(f"\n{group_folder} ({len(subjects)} subjects, label={label})")

    for subject in sorted(subjects):
        mri_dir = os.path.join(group_path, subject, "1.MRI")

        if not os.path.exists(mri_dir):
            print(f"  [SKIP] {subject} — no 1.MRI folder")
            continue

        # Get all PNG files in MRI folder
        png_files = sorted([
            f for f in os.listdir(mri_dir)
            if f.lower().endswith('.png')
        ])

        if len(png_files) == 0:
            print(f"  [SKIP] {subject} — no PNG files")
            continue

        # Copy PNGs to our slices directory with standard naming
        copied = 0
        for i, png in enumerate(png_files):
            src  = os.path.join(mri_dir, png)
            dst_name = f"ntua_{group_folder.replace(' ','_')}_{subject}_{i:03d}.png"
            dst  = os.path.join(OUT_DIR, dst_name)

            try:
                # Verify image is valid + convert to RGB
                img = Image.open(src).convert("RGB")
                img.save(dst)

                records.append({
                    "subject":  f"ntua_{subject}_{group_folder.replace(' ','_')}",
                    "dataset":  "ntua",
                    "label":    label,
                    "label_name": "control" if label == 0 else "patient",
                    "path":     dst,
                    "source":   src,
                })
                copied += 1
            except Exception as e:
                pass  # skip corrupt images

        print(f"  [OK] {subject} | {copied} slices")

# Save CSV
df = pd.DataFrame(records)
csv_path = os.path.join(BASE, "data", "labels", "ntua_slices.csv")
df.to_csv(csv_path, index=False)

print(f"\n{'='*55}")
print(f"NTUA subjects processed:")
print(f"  PD (label=1):      {(df['label']==1).nunique()} ... {(df['label']==1).sum()} slices")
print(f"  Control (label=0): {(df['label']==0).nunique()} ... {(df['label']==0).sum()} slices")
print(f"  Total slices:      {len(df)}")
print(f"  CSV saved → {csv_path}")
print(f"{'='*55}")

# In[ ]:



