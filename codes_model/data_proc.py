#!/usr/bin/env python3
import os
import re
import shutil
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import csv
from collections import defaultdict, Counter
from datetime import datetime

# ----------------------------
# Configurable constants
# ----------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Group-aware split by PDF base (avoid leakage of crops vs full pages)
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

# If True, any (pdf_base, page) that has at least one cropped "_imp_#"
# will cause the *full page* to be labeled as positive (1).
PROMOTE_PAGES_WITH_CROPS_TO_POSITIVE = True

# If True, when we copy a *promoted full page* into split folders,
# we also rename its filename to include "_imp__" before "_page_".
RENAME_PROMOTED_FULLPAGES_WITH_IMP = True

# ----------------------------
# Helpers
# ----------------------------
def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

# Patterns we support (case-insensitive):
#   "Foo Bar_page_12.jpg"
#   "Foo Bar_page_2_imp.jpg"
#   "Foo Bar_page_2_imp_1.jpg"   <-- your crop naming style
#   "Foo Bar_imp__page_4.jpg"    <-- post-processed style (later stages)
PAT_PAGE = re.compile(r'(?i)_page_(\d+)')
PAT_IMP  = re.compile(r'(?i)_imp_')            # handles "_imp_" and "_imp__"
PAT_CROP = re.compile(r'(?i)_imp_+(\d+)\b')    # e.g., "_imp_1", "_imp__2"

def company_from_root(img_root_dir: Path, src_path: Path) -> str:
    """
    Return the top-level folder name under img_root_dir for this file.
    e.g., dataset_root/CompanyA/input_all/.. -> 'CompanyA'
    """
    rel = src_path.resolve().relative_to(img_root_dir.resolve())
    return rel.parts[0] if len(rel.parts) > 0 else "ROOT"

def parse_filename(name: str) -> Tuple[Optional[str], Optional[int], bool, Optional[str]]:
    """
    Returns (pdf_base, page_num, is_important, crop_id)
    - pdf_base: substring before the first "_page_{num}"
    - page_num: int or None if not found
    - is_important: True if '_imp_' appears anywhere in the name
    - crop_id: digits right after _imp_ if present (e.g., _imp_3) else None
    """
    m_page = PAT_PAGE.search(name)
    if not m_page:
        return None, None, bool(PAT_IMP.search(name)), None

    page_num = int(m_page.group(1))
    base = name[:m_page.start()]
    is_imp = bool(PAT_IMP.search(name))
    m_crop = PAT_CROP.search(name)
    crop_id = m_crop.group(1) if m_crop else None
    base = base.rstrip("_ ").strip()
    return base, page_num, is_imp, crop_id

def add_imp_before_page(name: str) -> str:
    """
    Insert '_imp__' immediately before the first '_page_' segment.
    "Bio List_page_2.jpg" -> "Bio List_imp__page_2.jpg"
    """
    m = PAT_PAGE.search(name)
    if not m:
        return name
    return name[:m.start()] + "_imp__" + name[m.start():]

def remove_imp(name: str) -> str:
    """Remove any '_imp_' or '_imp__' from filename (case-insensitive)."""
    return re.sub(r'(?i)_imp_+', '_', name)

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def write_csv(rows: List[Dict], out_csv: Path):
    safe_mkdir(out_csv.parent)
    fieldnames = ["image_path", "label", "pdf_base", "page_num", "is_crop", "crop_id", "company"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Filter unknown keys so writer never explodes even if rows have extras
            w.writerow({k: r.get(k, "") for k in fieldnames})

def copy_to_split(dst_root: Path, split: str, label: int, src_path: Path, rel_name: str):
    sub = "important" if label == 1 else "unimportant"
    out_dir = dst_root / split / sub
    safe_mkdir(out_dir)
    shutil.copy2(src_path, out_dir / rel_name)

# ----------------------------
# Main
# ----------------------------
def process_dataset(img_root_dir, output_root_dir) -> Dict:
    """Process dataset into train/val/test splits."""

    logs: List[str] = []

    def log(msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        logs.append(formatted)
        print(formatted)

    img_root_dir = Path(img_root_dir).expanduser().resolve()
    output_root_dir = Path(output_root_dir).expanduser().resolve()

    if not img_root_dir.exists():
        msg = f"Input folder not found: {img_root_dir}"
        log(f"[ERROR] {msg}")
        return {"success": False, "message": msg, "logs": logs}
    safe_mkdir(output_root_dir)

    log("Place BOTH folders under one root, e.g.:")
    log("  data_root/input_all/  (full pages, no _imp_)")
    log("  data_root/input_crop/ (cropped positives with _imp_#)")

    # 1) Scan and parse
    all_rows: List[Dict] = []
    files = [p for p in img_root_dir.rglob("*") if p.is_file() and is_image_file(p)]
    if not files:
        log(f"[WARN] No images found in {img_root_dir}")
    files.sort()

    for p in files:
        pdf_base, page_num, is_imp, crop_id = parse_filename(p.name)
        is_crop = 1 if (is_imp and crop_id is not None) else 0
        label = 1 if is_imp else 0

        src_abs = p.resolve()
        comp = company_from_root(img_root_dir, src_abs)

        all_rows.append({
            "image_path": str(p.resolve()),
            "label": label,
            "pdf_base": pdf_base if pdf_base is not None else "",
            "page_num": page_num if page_num is not None else -1,
            "is_crop": is_crop,
            "crop_id": crop_id if crop_id is not None else "",
            "company": comp,
        })

    # 2) Identify pages that have at least one crop-positive
    crop_positive_pages = set()
    for r in all_rows:
        if r["label"] == 1 and r["is_crop"] == 1 and r["pdf_base"] and r["page_num"] >= 0:
            crop_positive_pages.add((r["pdf_base"], r["page_num"]))

    # 3) Optionally PROMOTE matching full pages (is_crop==0) to label=1
    upgraded = 0
    if PROMOTE_PAGES_WITH_CROPS_TO_POSITIVE:
        for r in all_rows:
            if r["is_crop"] == 0 and r["label"] == 0 and r["pdf_base"] and r["page_num"] >= 0:
                key = (r["pdf_base"], r["page_num"])
                if key in crop_positive_pages:
                    r["label"] = 1
                    upgraded += 1
        if upgraded:
            log(f"[INFO] Promoted {upgraded} full-page images to label=1 due to matching crop positives.")

    # 4) Save master CSV
    master_csv = output_root_dir / "labels_master.csv"
    write_csv(all_rows, master_csv)
    log(f"[OK] Wrote master CSV: {master_csv}")

    # 5) Group-aware split by pdf_base
    random.seed(RANDOM_SEED)
    bases = sorted({r["pdf_base"] for r in all_rows if r["pdf_base"]})
    random.shuffle(bases)
    n = len(bases)
    n_train = int(TRAIN_VAL_TEST_SPLIT[0] * n)
    n_val   = int(TRAIN_VAL_TEST_SPLIT[1] * n)
    train_bases = set(bases[:n_train])
    val_bases   = set(bases[n_train:n_train+n_val])
    test_bases  = set(bases[n_train+n_val:])

    split_rows = {"train": [], "val": [], "test": []}
    for r in all_rows:
        b = r["pdf_base"]
        if not b:
            split_rows["train"].append(r)
        elif b in train_bases:
            split_rows["train"].append(r)
        elif b in val_bases:
            split_rows["val"].append(r)
        else:
            split_rows["test"].append(r)

    # 6) Copy files into split folders
    for split in ("train", "val", "test"):
        out_csv = output_root_dir / f"labels_{split}.csv"
        write_csv(split_rows[split], out_csv)

        for r in split_rows[split]:
            src = Path(r["image_path"])
            rel_name = src.name

            if (
                RENAME_PROMOTED_FULLPAGES_WITH_IMP and
                r["is_crop"] == 0 and
                r["label"] == 1 and
                r["pdf_base"] and r["page_num"] >= 0
            ):
                rel_name = add_imp_before_page(rel_name)

            prefixed_name = f"{r['company']}__{rel_name}"
            copy_to_split(output_root_dir, split, r["label"], src, prefixed_name)

        log(f"[OK] Wrote split {split}: {out_csv} and copied files.")

    # 7) Summary
    def summarize(rows, title):
        labels = [r["label"] for r in rows]
        c = Counter(labels)
        total = len(labels)
        pos = c.get(1, 0)
        neg = c.get(0, 0)
        log(f"{title}: total={total} | important(1)={pos} | unimportant(0)={neg}")

    log("=== SUMMARY ===")
    summarize(all_rows, "ALL")
    for split in ("train", "val", "test"):
        summarize(split_rows[split], split.upper())

    log(f"[DONE] Dataset prepared under: {output_root_dir}")
    log("Notes:")
    log(" - Originals are NOT modified; renames happen only in the copied split folders.")
    log(" - Put BOTH 'input_all' and 'input_crop' under the same root; the script walks subfolders recursively.")
    log(" - '_imp_' means important ONLY. Ensure no unimportant file contains '_imp_'.")
    log(" - Grouped split prevents train/val/test leakage between a page and its crops.")

    return {
        "success": True,
        "message": f"Dataset prepared under {output_root_dir}",
        "logs": logs,
        "output_dir": str(output_root_dir),
        "counts": {
            "total_files": len(all_rows),
            "train": len(split_rows["train"]),
            "val": len(split_rows["val"]),
            "test": len(split_rows["test"]),
        },
    }


def main():
    print("Place BOTH folders under one root, e.g.:")
    print("  data_root/input_all/  (full pages, no _imp_)")
    print("  data_root/input_crop/ (cropped positives with _imp_#)")
    img_root_dir = Path(input("Enter path to input root folder: ").strip()).expanduser()
    output_root_dir = Path(input("Enter path to output folder: ").strip()).expanduser()
    result = process_dataset(img_root_dir, output_root_dir)
    print(result["message"])


if __name__ == "__main__":
    main()
