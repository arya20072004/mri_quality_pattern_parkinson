#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────
BASE     = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH = os.path.join(BASE, "data", "labels", "dataset.csv")
OUT_DIR  = os.path.join(BASE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
# ────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

# ── 1. Class Balance ─────────────────────────────────────────────
print("=== CLASS BALANCE ===")
print(df.groupby(["dataset", "label_name"]).size().to_string())
print(f"\nTotal Controls: {(df['label']==0).sum()}")
print(f"Total Patients: {(df['label']==1).sum()}")

# ── 2. Visualize one control + one patient (3 planes each) ───────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
fig.suptitle("MRI Scan Verification: Control vs Patient", fontsize=16, fontweight='bold')

samples = {
    "Control (HC)": df[df["label"] == 0].iloc[0],
    "Patient (PD)": df[df["label"] == 1].iloc[0],
}

for row_idx, (title, sample) in enumerate(samples.items()):
    arr = np.load(sample["processed_path"])
    mid = [s // 2 for s in arr.shape]

    planes = {
        "Axial (top-down)":    arr[:, :, mid[2]],
        "Coronal (front)":     arr[:, mid[1], :],
        "Sagittal (side)":     arr[mid[0], :, :],
    }

    for col_idx, (plane_name, slice_data) in enumerate(planes.items()):
        ax = axes[row_idx][col_idx]
        ax.imshow(slice_data.T, cmap="gray", origin="lower")
        ax.set_title(f"{title}\n{plane_name}", fontsize=9)
        ax.axis("off")

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "eda_scan_check.png")
plt.savefig(save_path, dpi=150)
plt.show()
print(f"\nScan visualization saved → {save_path}")

# ── 3. Intensity Distribution ────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
fig2.suptitle("Intensity Distribution: Control vs Patient", fontsize=13)

for ax, label_val, label_name, color in zip(
    axes2, [0, 1], ["Control (HC)", "Patient (PD)"], ["steelblue", "tomato"]
):
    group = df[df["label"] == label_val]
    all_vals = []
    for _, row in group.iterrows():
        arr = np.load(row["processed_path"]).flatten()
        all_vals.append(arr[::50])   # sample every 50th voxel for speed

    all_vals = np.concatenate(all_vals)
    ax.hist(all_vals, bins=80, color=color, alpha=0.8, edgecolor='none')
    ax.set_title(label_name)
    ax.set_xlabel("Intensity (z-scored)")
    ax.set_ylabel("Voxel count")
    ax.axvline(0, color='black', linestyle='--', linewidth=1, label='mean=0')
    ax.legend()

plt.tight_layout()
save_path2 = os.path.join(OUT_DIR, "eda_intensity.png")
plt.savefig(save_path2, dpi=150)
plt.show()
print(f"Intensity plot saved → {save_path2}")

# In[ ]:



