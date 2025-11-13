#!/usr/bin/env python3
"""
train_importance.py  (H,W-aware)

Train an image classifier to predict "important (1)" vs "unimportant (0)".

Inputs:
- CSVs from the preprocessing step:
    labels_train.csv, labels_val.csv, labels_test.csv
  Each row: image_path,label,pdf_base,page_num,is_crop,crop_id

Features:
- Augmentations for training
- Three architectures: cnn | vit | hybrid
- Optional weighted sampling (oversample crops or positives)
- Metric logging and best-checkpoint saving

Examples:
python train_importance.py \
  --data_root /OUTPUT_ROOT \
  --arch cnn \
  --epochs 12 \
  --batch_size 32 \
  --size_h 512 --size_w 360 \
  --save_dir ./runs/exp_cnn_512x360

python train_importance.py \
  --data_root /OUTPUT_ROOT \
  --arch hybrid \
  --epochs 12 \
  --batch_size 24 \
  --size_h 512 --size_w 360 \
  --save_dir ./runs/exp_hybrid_512x360

python train_importance.py \
  --data_root /OUTPUT_ROOT \
  --arch vit \
  --epochs 12 \
  --batch_size 16 \
  --size_h 512 --size_w 360 \
  --save_dir ./runs/exp_vit_512x360
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image
import csv
import time
from collections import Counter
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from datetime import datetime

# --------------------
# Letterbox (keep aspect, pad to HxW with white)
# --------------------
def letterbox(im: Image.Image, size_hw: Tuple[int,int]) -> Image.Image:
    H, W = size_hw
    im = im.convert("RGB")
    # Fit inside (W,H) while preserving aspect
    im.thumbnail((W, H), Image.BICUBIC)
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    x = (W - im.width) // 2
    y = (H - im.height) // 2
    bg.paste(im, (x, y))
    return bg

# --------------------
# Dataset
# --------------------
class ReportDataset(Dataset):
    def __init__(self, csv_path: Path,
                 split: str = "train",
                 target_h: int = 512, target_w: int = 360,
                 normalize: bool = True,
                 augment: bool = True):
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)

        self.split = split
        self.target_h = target_h
        self.target_w = target_w
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        self.do_aug = augment and split == "train"
        # Mild, doc-safe augs (no heavy crops that could erase layout)
        self.jitter = T.ColorJitter(brightness=0.15, contrast=0.15)
        self.rotate = T.RandomRotation(degrees=2, fill=(255,255,255))  # keep white corners

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        p = r["image_path"]
        y = int(r["label"])
        is_crop = int(r.get("is_crop", 0))

        img = Image.open(p).convert("RGB")
        # 1) Letterbox to exact target size (no distortion)
        img = letterbox(img, (self.target_h, self.target_w))

        # 2) Optional small augs
        if self.do_aug:
            # very light: tiny rotation + mild brightness/contrast
            img = self.rotate(img)
            img = self.jitter(img)

        # 3) To tensor + normalize
        img = T.ToTensor()(img)
        img = self.normalize(img)
        return img, y, is_crop

def build_sampler(rows: List[Dict], oversample_crops: bool = False,
                  crop_weight: float = 1.0,
                  pos_weight: float = 1.0):
    """
    Weighted sampler:
    - oversample_crops: extra weight to is_crop==1
    - crop_weight: multiplier for crop samples
    - pos_weight: multiplier for positives (label==1)
    """
    if not oversample_crops and pos_weight == 1.0 and crop_weight == 1.0:
        return None

    weights = []
    for r in rows:
        w = 1.0
        if pos_weight != 1.0 and int(r["label"]) == 1:
            w *= float(pos_weight)
        if oversample_crops and int(r.get("is_crop", 0)) == 1:
            w *= float(crop_weight)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

# --------------------
# Models
# --------------------
class SimpleCNN(nn.Module):
    """Compact CNN baseline; input size can be any HxW due to AdaptiveAvgPool."""
    def __init__(self, num_classes=2):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((7,7)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*7*7, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.feat(x)
        x = self.head(x)
        return x

class ViTClassifier(nn.Module):
    """
    Dynamic ViT:
    - No fixed img_size; builds positional embeddings on the fly.
    - Requires H and W to be multiples of patch_size (letterbox ensures this if you choose multiples).
    """
    def __init__(self, patch_size=16, dim=256, depth=4, heads=8, mlp_ratio=4,
                 num_classes=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = None  # will be (1, 1+N, dim)

        enc = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim*mlp_ratio,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        ps = self.patch_size
        assert H % ps == 0 and W % ps == 0, "H and W must be multiples of patch size"
        # Patchify
        x = x.unfold(2, ps, ps).unfold(3, ps, ps)          # B,C,H/ps,W/ps,ps,ps
        x = x.permute(0,2,3,1,4,5).contiguous()            # B,H/ps,W/ps,C,ps,ps
        x = x.view(B, -1, C*ps*ps)                         # B,N,patch_dim
        x = self.patch_embed(x)                            # B,N,dim

        N = x.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != (1 + N):
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, x.size(2), device=x.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                     # B,1+N,dim
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])                             # CLS token
        return self.head(x)

class HybridCNNTransformer(nn.Module):
    """CNN backbone -> tokens -> Transformer encoder -> classifier."""
    def __init__(self, num_classes=2, cnn_channels=256, token_dim=256, depth=2, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, cnn_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.proj = nn.Linear(cnn_channels, token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos_embed = None
        enc = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=token_dim*mlp_ratio,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.norm = nn.LayerNorm(token_dim)
        self.head = nn.Linear(token_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.size(0)
        feat = self.cnn(x)                                  # B,C,H',W'
        B, C, Hp, Wp = feat.shape
        tokens = feat.permute(0,2,3,1).contiguous().view(B, Hp*Wp, C)  # B,N,C
        tokens = self.proj(tokens)                          # B,N,D

        N = tokens.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != (1 + N):
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, tokens.size(2), device=tokens.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)                 # B,1+N,D
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)

def build_model(arch: str, num_classes: int = 2):
    if arch == "cnn":
        return SimpleCNN(num_classes=num_classes)
    elif arch == "vit":
        return ViTClassifier(num_classes=num_classes)  # dynamic; no img_size arg
    elif arch == "hybrid":
        return HybridCNNTransformer(num_classes=num_classes, cnn_channels=256, token_dim=256, depth=2, heads=8)
    else:
        raise ValueError(f"Unknown arch: {arch}")

# --------------------
# Training Utils
# --------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def load_rows(csv_path: Path) -> List[Dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for imgs, ys, _is_crop in loader:
        imgs, ys = imgs.to(device), ys.to(device)
        logits = model(imgs)
        loss = criterion(logits, ys)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        b = ys.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(logits, ys) * b
        n += b
    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for imgs, ys, _is_crop in loader:
        imgs, ys = imgs.to(device), ys.to(device)
        logits = model(imgs)
        loss = criterion(logits, ys)
        b = ys.size(0)
        total_loss += loss.item() * b
        total_acc += accuracy(logits, ys) * b
        n += b
    return total_loss / n, total_acc / n

@torch.no_grad()
def collect_logits_and_labels(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for imgs, ys, _ in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(ys.clone().cpu())
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_logits, all_labels

def compute_full_metrics(logits, labels):
    # logits -> probs for class 1
    probs = torch.from_numpy(logits).softmax(dim=1).numpy()[:, 1]
    preds = (logits.argmax(axis=1)).astype(int)

    # basic
    acc = (preds == labels).mean().item() if hasattr((preds == labels).mean(), 'item') else float((preds == labels).mean())

    # AUC (guard against single-class)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float('nan')

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()  # as list for JSON

    return {
        "acc": float(acc),
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm
    }

# --------------------
# Main
# --------------------
def train_model(
    data_root,
    save_dir,
    *,
    arch: str = "cnn",
    size_h: int = 512,
    size_w: int = 360,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    pos_weight: float = 1.0,
    oversample_crops: bool = False,
    crop_weight: float = 1.0,
) -> Dict:
    """Train the model programmatically and return metrics."""

    logs: List[str] = []

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        logs.append(formatted)
        print(formatted)

    if size_h % 16 != 0 or size_w % 16 != 0:
        msg = f"size_h and size_w must be multiples of 16; got {size_h}x{size_w}"
        log(f"[ERROR] {msg}")
        return {"success": False, "message": msg, "logs": logs}

    data_root = Path(data_root).expanduser().resolve()
    save_dir = Path(save_dir).expanduser().resolve()

    train_csv = data_root / "labels_train.csv"
    val_csv   = data_root / "labels_val.csv"
    test_csv  = data_root / "labels_test.csv"
    if not train_csv.exists():
        msg = f"Missing {train_csv}"
        log(f"[ERROR] {msg}")
        return {"success": False, "message": msg, "logs": logs}

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Device: {device}")
    log(f"Training arch={arch}, size={size_h}x{size_w}, epochs={epochs}, batch_size={batch_size}")

    ds_train = ReportDataset(train_csv, split="train", target_h=size_h, target_w=size_w,
                             normalize=True, augment=True)
    ds_val   = ReportDataset(val_csv,   split="val",   target_h=size_h, target_w=size_w,
                             normalize=True, augment=False)
    ds_test  = ReportDataset(test_csv,  split="test",  target_h=size_h, target_w=size_w,
                             normalize=True, augment=False)

    rows_train = load_rows(train_csv)
    sampler = None
    if oversample_crops or pos_weight != 1.0 or crop_weight != 1.0:
        sampler = build_sampler(
            rows_train,
            oversample_crops=oversample_crops,
            crop_weight=crop_weight,
            pos_weight=pos_weight,
        )

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=(sampler is None),
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=max(64, batch_size), shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=max(64, batch_size), shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(arch, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history: List[Dict] = []
    best_val = -1.0
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, device, criterion)
        va_loss, va_acc = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        log(f"[{epoch:03d}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f} | {dt:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "duration_sec": dt,
        })

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "arch": arch,
                "size_h": size_h,
                "size_w": size_w,
                "state_dict": model.state_dict(),
                "val_acc": va_acc,
            }, save_dir / "best.pt")
            log(f"  -> saved best.pt (val_acc={va_acc:.4f})")

    ckpt_path = save_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    te_loss, te_acc = evaluate(model, test_loader, device, criterion)
    logits_val, labels_val = collect_logits_and_labels(model, val_loader, device)
    val_metrics = compute_full_metrics(logits_val, labels_val)
    logits_test, labels_test = collect_logits_and_labels(model, test_loader, device)
    test_metrics = compute_full_metrics(logits_test, labels_test)

    results = {
        "size_h": int(size_h),
        "size_w": int(size_w),
        "arch": arch,
        "val_acc_checkpoint": float(ckpt["val_acc"]),
        "val_metrics": val_metrics,
        "test_loss": float(te_loss),
        "test_acc": float(te_acc),
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"[TEST] loss {te_loss:.4f} acc {te_acc:.4f} | AUC {test_metrics['auc']:.4f} | F1 {test_metrics['f1']:.4f}")
    log(f"[DONE] Saved best checkpoint and metrics to: {save_dir}/results.json")

    return {
        "success": True,
        "message": "Training complete",
        "logs": logs,
        "history": history,
        "results": results,
        "best_checkpoint": str(ckpt_path),
        "results_path": str(save_dir / "results.json"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="Folder containing labels_*.csv and split folders")
    ap.add_argument("--arch", type=str, default="cnn", choices=["cnn", "vit", "hybrid"])
    ap.add_argument("--size_h", type=int, default=512, help="Target input height")
    ap.add_argument("--size_w", type=int, default=360, help="Target input width")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_dir", type=str, default="./runs/exp")
    ap.add_argument("--pos_weight", type=float, default=1.0, help=">1.0 to upweight positives in sampler")
    ap.add_argument("--oversample_crops", type=str, default="false", help="true/false to oversample cropped positives")
    ap.add_argument("--crop_weight", type=float, default=1.0, help="Weight multiplier for crops when oversampling")
    args = ap.parse_args()

    result = train_model(
        data_root=args.data_root,
        save_dir=args.save_dir,
        arch=args.arch,
        size_h=args.size_h,
        size_w=args.size_w,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        pos_weight=args.pos_weight,
        oversample_crops=args.oversample_crops.lower() == "true",
        crop_weight=args.crop_weight,
    )
    print(result["message"])


if __name__ == "__main__":
    main()
