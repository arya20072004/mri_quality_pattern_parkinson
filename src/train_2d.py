#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────
BASE       = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH   = os.path.join(BASE, "data", "labels", "slices_dataset.csv")
MODEL_DIR  = os.path.join(BASE, "models")
OUT_DIR    = os.path.join(BASE, "outputs")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 64
EPOCHS     = 40
LR         = 5e-5
SEED       = 42
# ────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# ── SUBJECT-LEVEL SPLIT (no leakage) ────────────────────────────
df       = pd.read_csv(CSV_PATH)
subjects = df["subject"].unique()
labels   = [df[df["subject"]==s]["label"].iloc[0] for s in subjects]

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(gss.split(subjects, groups=subjects))
tv_subjects, test_subjects = subjects[tv_idx], subjects[te_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.176, random_state=SEED)
tr_idx, va_idx = next(gss2.split(tv_subjects, groups=tv_subjects))
train_subjects = tv_subjects[tr_idx]
val_subjects   = tv_subjects[va_idx]

train_df = df[df["subject"].isin(train_subjects)].reset_index(drop=True)
val_df   = df[df["subject"].isin(val_subjects)].reset_index(drop=True)
test_df  = df[df["subject"].isin(test_subjects)].reset_index(drop=True)

print(f"Train: {len(train_subjects)} subjects | {len(train_df)} slices")
print(f"Val:   {len(val_subjects)} subjects | {len(val_df)} slices")
print(f"Test:  {len(test_subjects)} subjects | {len(test_df)} slices\n")

# ── TRANSFORMS ──────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),  # cutout
])

val_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── DATASET ─────────────────────────────────────────────────────
class SliceDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        return self.transform(img), torch.tensor(int(row["label"]), dtype=torch.long)

train_loader = DataLoader(SliceDataset(train_df, train_tf),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(SliceDataset(val_df,   val_tf),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(SliceDataset(test_df,  val_tf),
                          batch_size=BATCH, shuffle=False, num_workers=0)

# ── MODEL: Freeze ALL except final FC ───────────────────────────
# With small data, only fine-tune the classifier head
# Backbone acts as a fixed feature extractor
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Freeze EVERYTHING
for param in model.parameters():
    param.requires_grad = False

# Only train the final classifier
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 2)
)
# fc is unfrozen by default since it's newly created
model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,}  (backbone frozen)\n")

counts  = train_df["label"].value_counts().sort_index().values
weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', patience=5, factor=0.5, verbose=True)

# ── TRAINING ────────────────────────────────────────────────────
best_val_acc = 0.0
no_improve   = 0
history      = {"tl":[], "vl":[], "ta":[], "va":[]}

print("Training (frozen backbone — only head trains)...\n")

for epoch in range(1, EPOCHS + 1):

    # — PHASE 1: Unfreeze layer4 after epoch 10 for fine-tuning —
    if epoch == 11:
        print("\n>>> Unfreezing layer4 for fine-tuning...\n")
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR * 0.1, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5, verbose=True)

    # Train
    model.train()
    tl, tc, tt = 0, 0, 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tl += loss.item() * X.size(0)
        tc += (out.argmax(1) == y).sum().item()
        tt += X.size(0)

    # Validate
    model.eval()
    vl, vc, vt = 0, 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out  = model(X)
            loss = loss_fn(out, y)
            vl += loss.item() * X.size(0)
            vc += (out.argmax(1) == y).sum().item()
            vt += X.size(0)

    ta, va = tc/tt, vc/vt
    tl_avg, vl_avg = tl/tt, vl/vt
    history["tl"].append(tl_avg); history["vl"].append(vl_avg)
    history["ta"].append(ta);     history["va"].append(va)

    scheduler.step(va)

    gap = ta - va  # overfitting indicator
    if va > best_val_acc:
        best_val_acc = va
        no_improve   = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "resnet18_best.pth"))
        tag = "  ← best"
    else:
        no_improve += 1
        tag = ""

    print(f"Ep {epoch:02d}/{EPOCHS} | "
          f"Train {ta:.3f} | Val {va:.3f} | Gap {gap:+.3f}{tag}")

    if no_improve >= 12:
        print(f"\nEarly stop at epoch {epoch}")
        break

print(f"\nBest Val Acc: {best_val_acc:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("ResNet18 Training (Frozen Backbone → Gradual Unfreeze)", fontsize=12)
axes[0].plot(history["tl"], label="Train"); axes[0].plot(history["vl"], label="Val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history["ta"], label="Train"); axes[1].plot(history["va"], label="Val")
axes[1].set_ylim(0, 1); axes[1].set_title("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "resnet18_v2_curves.png"), dpi=150)
plt.close()

# ── SUBJECT-LEVEL TEST EVALUATION ───────────────────────────────
print("\n=== TEST RESULTS (subject-level majority vote) ===")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "resnet18_best.pth")))
model.eval()

test_df2             = test_df.copy().reset_index(drop=True)
test_df2["pred"]     = -1
test_df2["prob1"]    = 0.0

all_preds, all_probs = [], []
with torch.no_grad():
    for X, _ in test_loader:
        out   = model(X.to(DEVICE))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs)

test_df2["pred"]  = all_preds
test_df2["prob1"] = all_probs

subject_results = []
for subj in test_subjects:
    rows     = test_df2[test_df2["subject"] == subj]
    true_lab = rows["label"].iloc[0]
    maj_vote = int(rows["pred"].mode()[0])
    mean_prob= rows["prob1"].mean()
    subject_results.append({"subject": subj, "true": true_lab,
                             "pred": maj_vote, "prob": mean_prob})

res_df = pd.DataFrame(subject_results)
print(res_df[["subject","true","pred","prob"]].to_string(index=False))

y_true = res_df["true"].tolist()
y_pred = res_df["pred"].tolist()
y_prob = res_df["prob"].tolist()

print(f"\n{classification_report(y_true, y_pred, target_names=['Control','Patient'])}")
try:
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")
except: pass

cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_2d_model.pth"))
print(f"\nFinal model saved → models/final_2d_model.pth")

# In[ ]:



