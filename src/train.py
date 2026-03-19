#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ── CONFIG ──────────────────────────────────────────────────────
BASE      = r"C:\Users\aryab\Coding\mri_quality_pattern_parkinson"
CSV_PATH  = os.path.join(BASE, "data", "labels", "dataset.csv")
MODEL_DIR = os.path.join(BASE, "models")
OUT_DIR   = os.path.join(BASE, "outputs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE = 4
EPOCHS     = 60
LR         = 3e-4
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
# ────────────────────────────────────────────────────────────────

print(f"Using device: {DEVICE}")
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── DATASET ──────────────────────────────────────────────────────
class MRIDataset(Dataset):
    def __init__(self, df, augment=False):
        self.df      = df.reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        arr   = np.load(row["processed_path"]).astype(np.float32)
        label = int(row["label"])

        if self.augment:
            if np.random.rand() > 0.5: arr = np.flip(arr, axis=0).copy()
            if np.random.rand() > 0.5: arr = np.flip(arr, axis=1).copy()
            if np.random.rand() > 0.5: arr = np.flip(arr, axis=2).copy()
            # Random intensity scaling
            arr *= np.random.uniform(0.9, 1.1)
            # Small gaussian noise
            arr += np.random.normal(0, 0.02, arr.shape).astype(np.float32)

        arr = arr[np.newaxis, ...]  # (1, 128, 128, 128)
        return torch.tensor(arr), torch.tensor(label, dtype=torch.long)


# ── SMALL MODEL (fits small dataset) ─────────────────────────────
class SmallPD3DCNN(nn.Module):
    """
    ~200K parameters — much better suited for 83 subjects.
    Uses Global Average Pooling instead of huge FC layers.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),              # 64³

            # Block 2
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),              # 32³

            # Block 3
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),              # 16³

            # Block 4
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),      # → (B, 64, 1, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # → (B, 64)
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── TRAIN / EVAL FUNCTIONS ───────────────────────────────────────
def run_epoch(model, loader, optimizer, loss_fn, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if training:
                optimizer.zero_grad()
            out  = model(X)
            loss = loss_fn(out, y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct    += (out.argmax(1) == y).sum().item()
            total      += X.size(0)
    return total_loss / total, correct / total


# ── DATA ─────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df["processed_path"].notna()].reset_index(drop=True)

# Hold out a fixed test set (never touched during training)
train_val_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=SEED)

print(f"Test set (held out): {len(test_df)} subjects")
print(f"Train+Val pool:      {len(train_val_df)} subjects\n")

# ── 5-FOLD CROSS VALIDATION ──────────────────────────────────────
# With only 83 subjects, k-fold gives a much more reliable estimate
# than a single train/val split

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_val_df, train_val_df["label"]), 1):

    print(f"{'='*55}")
    print(f"FOLD {fold}/5")
    print(f"{'='*55}")

    fold_train = train_val_df.iloc[train_idx]
    fold_val   = train_val_df.iloc[val_idx]

    print(f"  Train: {len(fold_train)} | Val: {len(fold_val)}")

    train_loader = DataLoader(MRIDataset(fold_train, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(MRIDataset(fold_val, augment=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Fresh model each fold
    model = SmallPD3DCNN().to(DEVICE)

    counts   = fold_train["label"].value_counts().sort_index().values
    weights  = torch.tensor(1.0 / counts, dtype=torch.float32).to(DEVICE)
    loss_fn  = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc  = 0.0
    best_model_path = os.path.join(MODEL_DIR, f"fold{fold}_best.pth")
    history = {"tl": [], "vl": [], "ta": [], "va": []}

    for epoch in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, train_loader, optimizer, loss_fn, training=True)
        vl, va = run_epoch(model, val_loader,   optimizer, loss_fn, training=False)
        scheduler.step()

        history["tl"].append(tl); history["vl"].append(vl)
        history["ta"].append(ta); history["va"].append(va)

        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), best_model_path)
            tag = "  ← best"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:02d}/{EPOCHS} | "
                  f"Train {ta:.3f} | Val {va:.3f}{tag}")

    fold_results.append(best_val_acc)
    print(f"  Fold {fold} best val acc: {best_val_acc:.3f}\n")

    # Plot this fold
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Fold {fold} Training", fontsize=12)
    ax[0].plot(history["tl"], label="Train"); ax[0].plot(history["vl"], label="Val")
    ax[0].set_title("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(history["ta"], label="Train"); ax[1].plot(history["va"], label="Val")
    ax[1].set_title("Accuracy"); ax[1].set_ylim(0, 1)
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"fold{fold}_curves.png"), dpi=120)
    plt.close()

print(f"\n{'='*55}")
print(f"Cross-validation results:")
for i, acc in enumerate(fold_results, 1):
    print(f"  Fold {i}: {acc:.3f}")
print(f"  Mean: {np.mean(fold_results):.3f} ± {np.std(fold_results):.3f}")
print(f"{'='*55}\n")

# ── FINAL TEST EVALUATION (best fold model) ───────────────────────
best_fold = int(np.argmax(fold_results)) + 1
print(f"Using Fold {best_fold} model for final test evaluation...\n")

model = SmallPD3DCNN().to(DEVICE)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"fold{best_fold}_best.pth")))
model.eval()

test_loader = DataLoader(MRIDataset(test_df, augment=False),
                         batch_size=1, shuffle=False, num_workers=0)

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        out  = model(X.to(DEVICE))
        prob = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
        pred = out.argmax(1).cpu().numpy()
        all_preds.extend(pred)
        all_labels.extend(y.numpy())
        all_probs.extend(prob)

print("=== FINAL TEST RESULTS ===")
print(classification_report(all_labels, all_preds,
                             target_names=["Control", "Patient"]))
try:
    print(f"AUC-ROC: {roc_auc_score(all_labels, all_probs):.4f}")
except:
    print("AUC-ROC: not computable (too few test samples)")

cm = confusion_matrix(all_labels, all_preds)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# In[ ]:



