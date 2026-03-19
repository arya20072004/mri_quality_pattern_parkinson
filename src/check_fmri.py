#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import nibabel as nib
import pandas as pd

BASE     = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH = os.path.join(BASE, "data", "labels", "dataset.csv")

df = pd.read_csv(CSV_PATH)

print("Checking fMRI files...\n")

fmri_records = []
for _, row in df.iterrows():
    subj    = row["subject"]
    dataset = row["dataset"]
    label   = row["label"]

    # fMRI is in func/ folder next to anat/
    t1_path  = row["t1_path"]
    func_dir = t1_path.replace("anat", "func")
    func_dir = os.path.dirname(func_dir)

    # Find bold files
    bold_files = []
    if os.path.exists(func_dir):
        bold_files = sorted([
            os.path.join(func_dir, f)
            for f in os.listdir(func_dir)
            if f.endswith("_bold.nii.gz")
        ])

    if not bold_files:
        print(f"  [MISSING] {subj}")
        continue

    # Check first bold file
    try:
        img   = nib.load(bold_files[0])
        shape = img.shape   # should be (X, Y, Z, timepoints)
        tr    = img.header.get_zooms()[3]   # TR in seconds
        print(f"  [OK] {subj:30s} | shape={shape} | TR={tr:.2f}s | {len(bold_files)} run(s)")
        fmri_records.append({
            "subject":    subj,
            "dataset":    dataset,
            "label":      label,
            "fmri_path":  bold_files[0],
            "shape":      str(shape),
            "tr":         tr,
            "n_runs":     len(bold_files),
        })
    except Exception as e:
        print(f"  [ERROR] {subj} → {e}")

fmri_df = pd.DataFrame(fmri_records)
fmri_csv = os.path.join(BASE, "data", "labels", "fmri_dataset.csv")
fmri_df.to_csv(fmri_csv, index=False)

print(f"\n{'='*55}")
print(f"fMRI scans found:  {len(fmri_df)}")
print(f"Controls:          {(fmri_df['label']==0).sum()}")
print(f"Patients:          {(fmri_df['label']==1).sum()}")
print(f"Unique shapes:     {fmri_df['shape'].unique()}")
print(f"CSV saved → {fmri_csv}")
print(f"{'='*55}")

# In[ ]:



