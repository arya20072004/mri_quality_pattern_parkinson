#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


BASE       = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
NEURO_CSV  = os.path.join(BASE, "data", "labels", "slices_dataset.csv")
NTUA_CSV   = os.path.join(BASE, "data", "labels", "ntua_slices.csv")
MODEL_DIR  = os.path.join(BASE, "models")
OUT_DIR    = os.path.join(BASE, "outputs")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH      = 64
EPOCHS     = 25
LR         = 1e-4
SEED       = 42
MAX_SLICES_PER_SUBJECT = 60   # cap — prevents any subject dominating

# In[3]:


torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"Device: {DEVICE}")

# In[4]:


neuro_df = pd.read_csv(NEURO_CSV)[["subject","label","path"]]
ntua_df  = pd.read_csv(NTUA_CSV)[["subject","label","path"]]
df       = pd.concat([neuro_df, ntua_df], ignore_index=True)

# Cap slices per subject — sample evenly spaced slices
def cap_subject_slices(df, max_slices, seed=42):
    rng    = np.random.default_rng(seed)
    frames = []
    for subj, grp in df.groupby("subject"):
        if len(grp) > max_slices:
            # Evenly spaced sampling (not random) — preserves scan coverage
            indices = np.linspace(0, len(grp)-1, max_slices, dtype=int)
            grp     = grp.iloc[indices]
        frames.append(grp)
    return pd.concat(frames, ignore_index=True)

df_capped = cap_subject_slices(df, MAX_SLICES_PER_SUBJECT)

print(f"\nAfter capping at {MAX_SLICES_PER_SUBJECT} slices/subject:")
print(f"  Total subjects: {df_capped['subject'].nunique()}")
print(f"  Total slices:   {len(df_capped)}")
print(f"  PD slices:      {(df_capped['label']==1).sum()}")
print(f"  Control slices: {(df_capped['label']==0).sum()}")


# In[5]:


subjects = df_capped["subject"].unique()

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tv_idx, te_idx = next(gss.split(subjects, groups=subjects))
tv_subjects, test_subjects = subjects[tv_idx], subjects[te_idx]

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
tr_idx, va_idx = next(gss2.split(tv_subjects, groups=tv_subjects))
train_subjects = tv_subjects[tr_idx]
val_subjects   = tv_subjects[va_idx]

train_df = df_capped[df_capped["subject"].isin(train_subjects)].reset_index(drop=True)
val_df   = df_capped[df_capped["subject"].isin(val_subjects)].reset_index(drop=True)
test_df  = df_capped[df_capped["subject"].isin(test_subjects)].reset_index(drop=True)

# For test evaluation use ALL slices (not capped) for better majority vote
test_df_full = df[df["subject"].isin(test_subjects)].reset_index(drop=True)

print(f"\nSplit:")
print(f"  Train: {len(train_subjects)} subjects | {len(train_df)} slices")
print(f"  Val:   {len(val_subjects)} subjects | {len(val_df)} slices")
print(f"  Test:  {len(test_subjects)} subjects | {len(test_df_full)} slices (full)\n")


# In[6]:


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.4, scale=(0.02, 0.2)),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# In[7]:


class SliceDataset(Dataset):
    def __init__(self, df, transform):
        self.df        = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["path"]).convert("RGB")
        except:
            img = Image.new("RGB", (224, 224), 0)
        return self.transform(img), torch.tensor(int(row["label"]), dtype=torch.long)

train_loader = DataLoader(SliceDataset(train_df, train_tf),
                          batch_size=BATCH, shuffle=True,  num_workers=0)
val_loader   = DataLoader(SliceDataset(val_df,   val_tf),
                          batch_size=BATCH, shuffle=False, num_workers=0)
test_loader  = DataLoader(SliceDataset(test_df_full, val_tf),
                          batch_size=BATCH, shuffle=False, num_workers=0)


# In[8]:


from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Freeze all backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.classifier[1].in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(128, 2)
)
model = model.to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable:,} (backbone fully frozen)\n")

counts  = train_df["label"].value_counts().sort_index().values
weights = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
loss_fn = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


# In[9]:


best_val_acc = 0.0
no_improve   = 0
history      = {"tl":[], "vl":[], "ta":[], "va":[]}
print("Training (EfficientNet-B0, frozen backbone)...\n")

for epoch in range(1, EPOCHS + 1):

    # Unfreeze last block at epoch 10
    if epoch == 11:
        print("\n>>> Unfreezing EfficientNet last block...\n")
        for name, param in model.named_parameters():
            if "features.8" in name or "classifier" in name:
                param.requires_grad = True
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR * 0.2, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS - 10)

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

    scheduler.step()
    ta, va = tc/tt, vc/vt
    history["tl"].append(tl/tt); history["vl"].append(vl/vt)
    history["ta"].append(ta);    history["va"].append(va)

    if va > best_val_acc:
        best_val_acc = va
        no_improve   = 0
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "efficientnet_best.pth"))
        tag = "  ← best"
    else:
        no_improve += 1
        tag = ""

    print(f"Ep {epoch:02d}/{EPOCHS} | Train {ta:.3f} | Val {va:.3f} | Gap {ta-va:+.3f}{tag}")

    if no_improve >= 10:
        print(f"\nEarly stop at epoch {epoch}")
        break

print(f"\nBest Val Acc: {best_val_acc:.3f}")


# In[11]:


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("EfficientNet-B0 — Combined Dataset", fontsize=12)
axes[0].plot(history["tl"], label="Train"); axes[0].plot(history["vl"], label="Val")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(history["ta"], label="Train"); axes[1].plot(history["va"], label="Val")
axes[1].set_ylim(0, 1); axes[1].set_title("Accuracy")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "efficientnet_training.png"), dpi=150)
plt.close()

# In[12]:


print("\n=== TEST RESULTS (subject-level majority vote) ===")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "efficientnet_best.pth")))
model.eval()

test_df_full2          = test_df_full.copy().reset_index(drop=True)
all_preds, all_probs   = [], []
with torch.no_grad():
    for X, _ in test_loader:
        out   = model(X.to(DEVICE))
        probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_probs.extend(probs)

test_df_full2["pred"]  = all_preds
test_df_full2["prob1"] = all_probs

subject_results = []
for subj in test_subjects:
    rows = test_df_full2[test_df_full2["subject"] == subj]
    if len(rows) == 0: continue
    subject_results.append({
        "subject":  subj,
        "true":     rows["label"].iloc[0],
        "pred":     int(rows["pred"].mode()[0]),
        "prob":     rows["prob1"].mean(),
        "n_slices": len(rows),
    })

res_df = pd.DataFrame(subject_results)
print(res_df[["subject","true","pred","prob","n_slices"]].to_string(index=False))

y_true = res_df["true"].tolist()
y_pred = res_df["pred"].tolist()
y_prob = res_df["prob"].tolist()

print(f"\n{classification_report(y_true, y_pred, target_names=['Control','Patient'])}")
try:
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")
except: pass

cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_efficientnet_model.pth"))
print(f"\nFinal model saved → models/final_efficientnet_model.pth")

# In[ ]:



