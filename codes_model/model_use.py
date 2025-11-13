#!/usr/bin/env python3
"""
infer_and_export.py  (H,W-aware)

Inference script for page-importance classification and export:
- Classifies full-page images as important/unimportant
- Adds _imp_ tag for predicted important pages in a temp folder
- Copies _page_1 + all predicted important pages to final output
- Removes _imp_ and resequences pages per PDF base: _page_1, _page_2, ...

Assumptions:
- Filenames look like "<pdf_base>_page_<N>.<ext>"
- Input folder contains ONLY full pages for inference (no _imp_), possibly nested.
- best.pt was created by train_importance.py (stores arch and size_h/size_w)

Usage:
    python model_use.py --ckpt ./runs/exp_cnn_512x352/best.pt

    !python "/content/drive/MyDrive/codes_model/model_use.py" \
  --ckpt "/content/drive/MyDrive/codes_model/runs/exp_cnn_512x352/best.pt"
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import shutil
from datetime import datetime

# ----------------------------
# Filename parsing helpers
# ----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PAT_PAGE = re.compile(r'(?i)_page_(\d+)')
PAT_IMP  = re.compile(r'(?i)_imp_')  # we will add/remove this around _page_

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def parse_base_and_page(name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Returns (pdf_base, page_num) from "<pdf_base>_page_<N>..."
    """
    m = PAT_PAGE.search(name)
    if not m:
        return None, None
    page = int(m.group(1))
    base = name[:m.start()].rstrip("_ ").strip()
    return base, page

def add_imp_before_page(name: str) -> str:
    """
    Insert '_imp__' immediately before the first '_page_' segment.
    Example: "Bio List_page_2.jpg" -> "Bio List_imp__page_2.jpg"
    """
    m = PAT_PAGE.search(name)
    if not m:
        return name
    return name[:m.start()] + "_imp__" + name[m.start():]

def remove_imp(name: str) -> str:
    """Remove any _imp_ or _imp__ from filename (case-insensitive)."""
    return re.sub(r'(?i)_imp_+', '_', name)

# ----------------------------
# Letterbox (keep aspect, pad to HxW with white)
# ----------------------------

def letterbox(im: Image.Image, size_hw: Tuple[int,int]) -> Image.Image:
    H, W = size_hw
    im = im.convert("RGB")
    im.thumbnail((W, H), Image.BICUBIC)  # fit inside (W,H) preserving aspect
    bg = Image.new("RGB", (W, H), (255, 255, 255))
    x = (W - im.width) // 2
    y = (H - im.height) // 2
    bg.paste(im, (x, y))
    return bg

# ----------------------------
# Models (match training; dynamic ViT/Hybrid)
# ----------------------------

class SimpleCNN(nn.Module):
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
    """Dynamic ViT that builds positional embeddings on the fly."""
    def __init__(self, patch_size=16, dim=256, depth=4, heads=8, mlp_ratio=4,
                 num_classes=2, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(self.patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = None  # (1,1+N,dim), created per input size

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
        assert H % ps == 0 and W % ps == 0, "H and W must be multiples of patch size (letterbox ensures this)."

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
    def __init__(self, num_classes=2, cnn_channels=256, token_dim=256, depth=2, heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, cnn_channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
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
        feat = self.cnn(x)                                 # B,C,H',W'
        B, C, Hp, Wp = feat.shape
        tokens = feat.permute(0,2,3,1).contiguous().view(B, Hp*Wp, C)  # B,N,C
        tokens = self.proj(tokens)                         # B,N,D

        N = tokens.size(1)
        if self.pos_embed is None or self.pos_embed.size(1) != (1 + N):
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + N, tokens.size(2), device=tokens.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, tokens], dim=1)                # B,1+N,D
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)

def build_model(arch: str, num_classes: int = 2):
    if arch == "cnn":
        return SimpleCNN(num_classes)
    elif arch == "vit":
        return ViTClassifier(num_classes=num_classes)  # dynamic HxW
    elif arch == "hybrid":
        return HybridCNNTransformer(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")

# ----------------------------
# Inference
# ----------------------------

DEFAULT_CKPT = Path(__file__).resolve().parent / "runs" / "exp_cnn_512x352" / "best.pt"


def run_inference(
    img_root_dir,
    out_main_dir,
    ckpt_path: Optional[str] = None,
    size_h: Optional[int] = None,
    size_w: Optional[int] = None,
    arch: Optional[str] = None,
    confidence: float = 0.5,
    temp_dirname: str = "_temp_predictions",
) -> Dict:
    """Run inference and export selected images.

    Returns:
        dict with keys:
            success (bool)
            message (str)
            logs (List[str])
            output_dir (str) -- final selected images
            temp_dir (str)   -- temp predictions folder
            summary (Dict[str, int]) -- counts
    """

    logs: List[str] = []

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        logs.append(formatted)
        print(formatted)

    img_root_dir = Path(img_root_dir).expanduser().resolve()
    out_main_dir = Path(out_main_dir).expanduser().resolve()
    if ckpt_path is None:
        ckpt_path = DEFAULT_CKPT
    ckpt_path = Path(ckpt_path).expanduser().resolve()

    if not img_root_dir.exists():
        msg = f"Input folder not found: {img_root_dir}"
        log(f"[ERROR] {msg}")
        return {"success": False, "message": msg, "logs": logs}
    if not ckpt_path.exists():
        msg = f"Checkpoint not found: {ckpt_path}"
        log(f"[ERROR] {msg}")
        return {"success": False, "message": msg, "logs": logs}

    # Prepare output directories
    if out_main_dir.exists():
        shutil.rmtree(out_main_dir)
    out_main_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint (for arch and default sizes)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch = arch or ckpt.get("arch", "cnn")
    size_h = size_h or ckpt.get("size_h", 512)
    size_w = size_w or ckpt.get("size_w", 360)

    if size_h % 16 != 0 or size_w % 16 != 0:
        msg = f"size_h and size_w must be multiples of 16; got {size_h}x{size_w}"
        log(f"[ERROR] {msg}")
        return {"success": False, "message": msg, "logs": logs}

    log(f"Using checkpoint: {ckpt_path}")
    log(f"Model arch={arch}, size={size_h}x{size_w}, confidence={confidence}")

    model = build_model(arch).eval()
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def pil_to_tensor(img: Image.Image) -> torch.Tensor:
        img = letterbox(img, (size_h, size_w))
        t = T.ToTensor()(img)
        t = normalize(t)
        return t

    all_imgs: List[Path] = [p for p in img_root_dir.rglob("*") if p.is_file() and is_image(p)]
    full_pages = [p for p in all_imgs if not PAT_IMP.search(p.name)]
    if not full_pages:
        msg = "No full-page images found (no files without _imp_)."
        log(f"[WARN] {msg}")
        return {"success": False, "message": msg, "logs": logs}

    temp_dir = out_main_dir / temp_dirname
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    preds_per_base: Dict[str, List[Tuple[int, Path, float]]] = {}
    softmax = nn.Softmax(dim=1)

    log(f"Running inference on {len(full_pages)} image(s)...")
    for p in sorted(full_pages):
        base, page = parse_base_and_page(p.name)
        if base is None or page is None:
            continue

        img = Image.open(p).convert("RGB")
        x = pil_to_tensor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = softmax(logits)[:, 1].item()

        out_name = p.name
        if prob >= confidence:
            out_name = add_imp_before_page(out_name)
        shutil.copy2(p, temp_dir / out_name)

        preds_per_base.setdefault(base, []).append((page, p, prob))

    log(f"Temp predictions written to: {temp_dir}")

    selected_count = 0
    total_bases = 0
    log("Exporting selected pages to final output with resequencing...")
    for base, items in preds_per_base.items():
        total_bases += 1
        page2prob = {pg: pr for pg, _, pr in items}
        selected_pages = set()

        if 1 in page2prob:
            selected_pages.add(1)

        for pg, _, pr in items:
            if pr >= confidence:
                selected_pages.add(pg)

        if not selected_pages:
            continue

        sorted_pages = sorted(selected_pages)
        k = 1
        for pg in sorted_pages:
            candidates = (
                list(temp_dir.glob(f"{base}_page_{pg}.*")) +
                list(temp_dir.glob(f"{base}_imp__page_{pg}.*")) +
                list(temp_dir.glob(f"{base}_IMP__page_{pg}.*"))
            )
            if not candidates:
                candidates = list(temp_dir.glob(f"{base}_*page_{pg}.*"))
            if not candidates:
                log(f"[WARN] Could not find temp file for {base} page {pg}")
                continue

            src = candidates[0]
            clean_name = remove_imp(src.name)
            clean_name = re.sub(r'(?i)_page_\\d+', f"_page_{k}", clean_name)
            shutil.copy2(src, out_main_dir / clean_name)
            k += 1
            selected_count += 1

    log(f"Final output written to: {out_main_dir}")
    log("Notes:")
    log(f" - Temp predictions (with _imp_ marks) remain in: {temp_dir}")
    log(" - Final output has _imp_ removed and pages resequenced per document base.")

    return {
        "success": True,
        "message": f"Inference complete. Selected {selected_count} pages across {total_bases} document(s).",
        "logs": logs,
        "output_dir": str(out_main_dir),
        "temp_dir": str(temp_dir),
        "summary": {"selected_pages": selected_count, "documents": total_bases},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None, help="Path to trained checkpoint")
    ap.add_argument("--size_h", type=int, default=None, help="Override height (else use ckpt or default 512)")
    ap.add_argument("--size_w", type=int, default=None, help="Override width  (else use ckpt or default 360)")
    ap.add_argument("--arch", type=str, default=None, help="Override arch (else use ckpt: cnn|vit|hybrid)")
    ap.add_argument("--confidence", type=float, default=0.5, help="Softmax prob threshold for 'important'")
    ap.add_argument("--temp_dirname", type=str, default="_temp_predictions", help="Temp folder inside output")
    ap.add_argument("--input", type=str, default=None, help="Path to input root folder (images)")
    ap.add_argument("--output", type=str, default=None, help="Path to output folder")
    args = ap.parse_args()

    if args.input is None:
        args.input = input("Enter path to input root folder: ").strip()
    if args.output is None:
        args.output = input("Enter path to output main folder: ").strip()

    result = run_inference(
        img_root_dir=args.input,
        out_main_dir=args.output,
        ckpt_path=args.ckpt,
        size_h=args.size_h,
        size_w=args.size_w,
        arch=args.arch,
        confidence=args.confidence,
        temp_dirname=args.temp_dirname,
    )
    print(result["message"])


if __name__ == "__main__":
    main()
