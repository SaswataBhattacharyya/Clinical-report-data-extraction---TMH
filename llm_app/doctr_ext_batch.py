import os
import time
from PIL import Image
from tqdm import tqdm
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
import tempfile
import numpy as np
import regex as re

# --- Backend & device ---
os.environ["USE_TORCH"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load OCR model ---
model = ocr_predictor(det_arch="db_resnet50", reco_arch="sar_resnet31", pretrained=True)  # "sar_resnet31" "crnn_vgg16_bn"
model = model.to(device)

# -------------------------------
# OCR: per-batch -> per-image export (one page each)
# -------------------------------
def run_batch_ocr_export(image_paths, upscale_factor=1):
    """OCR -> return a list of per-image export dicts (one page each)."""
    images = []
    temp_paths = []
    try:
        for img_path in image_paths:
            img = Image.open(img_path)
            if upscale_factor and upscale_factor != 1:
                new_size = (img.width * upscale_factor, img.height * upscale_factor)
                img = img.resize(new_size, Image.LANCZOS)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Use PNG temp (lossless) to preserve edges
            tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(tmp_file.name, format="PNG", optimize=True)
            images.append(tmp_file.name)
            temp_paths.append(tmp_file.name)

        doc = DocumentFile.from_images(images)
        results = model(doc)              # Document
        full_export = results.export()    # dict with {"pages": [ ... ]}
        pages = full_export.get("pages", [])

        # One input image corresponds to one page in the same order.
        return [{"pages": [p]} for p in pages]
    finally:
        for temp in temp_paths:
            try:
                os.remove(temp)
            except Exception:
                pass

# -------------------------------
# Geometry helpers
# -------------------------------
def _collect_words_from_export(page_dict):
    """
    Flatten Doctr export page into a list of words with normalized coords.
    Returns: list of dicts with keys: text, xc, yc, x0, x1, y0, y1
    """
    words = []
    for block in page_dict.get("blocks", []):
        for line in block.get("lines", []):
            for w in line.get("words", []):
                txt = (w.get("value") or "").strip()
                if not txt:
                    continue
                # geometry: [[x0, y0], [x1, y1]] normalized [0,1]
                (x0, y0), (x1, y1) = w.get("geometry", [[0, 0], [0, 0]])
                xc = (x0 + x1) / 2.0
                yc = (y0 + y1) / 2.0
                words.append({
                    "text": txt, "x0": x0, "y0": y0, "x1": x1, "y1": y1, "xc": xc, "yc": yc
                })
    return words

def _cluster_rows(words, y_tol=0.012):
    """
    Group words into rows using y (normalized). Sort words left->right inside each row.
    y_tol controls how close in y two words must be to be considered the same row.
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: w["yc"])
    rows = []
    current = []
    last_y = None
    for w in words:
        if last_y is None or abs(w["yc"] - last_y) <= y_tol:
            current.append(w)
            last_y = w["yc"] if last_y is None else (0.7 * last_y + 0.3 * w["yc"])  # smooth moving avg
        else:
            rows.append(sorted(current, key=lambda x: x["x0"]))
            current = [w]
            last_y = w["yc"]
    if current:
        rows.append(sorted(current, key=lambda x: x["x0"]))
    return rows

# -------------------------------
# Two alignment styles (TEXT only)
# -------------------------------
def make_aligned_text(rows, cols_per_line=180, factor=1):
    """
    Grid-aligned: map x0 -> fixed character grid (cols_per_line * factor).
    """
    lines = []
    col_scale = max(10, int(cols_per_line * factor))

    for row in rows:
        if not row:
            lines.append("")
            continue

        # (optional) per-row char width estimate (kept for future tuning)
        char_ws = []
        for w in row:
            wlen = max(1, len(w["text"]))
            char_ws.append((w["x1"] - w["x0"]) / wlen)
        _med_cw = np.median(char_ws) if char_ws else 0.01

        buf = [" "] * (col_scale + 8)
        for w in row:
            col = int(round(w["x0"] * col_scale))
            col = max(0, min(col, col_scale))
            word = w["text"]
            for i, ch in enumerate(word):
                idx = col + i
                if idx >= len(buf):
                    break
                if buf[idx] == " ":
                    buf[idx] = ch
                else:
                    if idx + 1 < len(buf) and buf[idx + 1] == " ":
                        buf[idx + 1] = ch
        lines.append("".join(buf).rstrip())

    return "\n".join(lines)

def make_spaced_text(rows, char_scale=140, min_spaces=1):
    """
    Gap-based: insert spaces proportional to actual horizontal gap.
      spaces = max(min_spaces, round((x0_next - x1_prev) * char_scale))
    """
    lines = []
    for row in rows:
        if not row:
            lines.append("")
            continue
        row = sorted(row, key=lambda w: w["x0"])
        pieces = [row[0]["text"]]
        last_right = row[0]["x1"]
        for w in row[1:]:
            gap = max(0.0, w["x0"] - last_right)   # normalized 0..1
            spaces = max(min_spaces, int(round(gap * char_scale)))
            pieces.append(" " * spaces + w["text"])
            last_right = w["x1"]
        lines.append("".join(pieces))
    return "\n".join(lines)

# -------------------------------
# Batch driver (writes TWO text outputs per image)
# -------------------------------
def extract_images_to_text(
    img_root_dir,
    output_root_dir="doctr_text",
    batch_size=2,
    y_tol=0.012,
    upscale_factor=1,
):
    """
    For each image, write exactly two text files:
      - {base}.aligned.grid.cols180.txt
      - {base}.aligned.spaced.cs140.txt
    """
    os.makedirs(output_root_dir, exist_ok=True)

    IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    image_file_info = []
    for root, _, files in os.walk(img_root_dir):
        for file in files:
            if file.lower().endswith(IMG_EXTS):
                img_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, img_root_dir)
                output_subdir = os.path.join(output_root_dir, rel_path)
                os.makedirs(output_subdir, exist_ok=True)

                base = os.path.splitext(file)[0]
                m = re.match(r"^(.*)_page_([0-9]+)$", base, flags=re.IGNORECASE)
                if m:
                    report_base, page_no = m.group(1), int(m.group(2))
                else:
                    report_base, page_no = base, 1  # fallback if no _page_N in name
                image_file_info.append((img_path, output_subdir, base))

    print(f"Found {len(image_file_info)} images to process under {img_root_dir}")

    for i in tqdm(range(0, len(image_file_info), batch_size), desc="Batch OCR & Reflow"):
        batch_info = image_file_info[i:i + batch_size]
        image_paths = [info[0] for info in batch_info]

        per_image_exports = run_batch_ocr_export(image_paths, upscale_factor=upscale_factor)

        for export, (_, out_dir, base) in zip(per_image_exports, batch_info):
            pages = export.get("pages", [])
            if not pages:
                # write an empty marker so you know it ran
                open(os.path.join(out_dir, f"{base}.EMPTY.txt"), "w", encoding="utf-8").close()
                print(f"⚠️ No page content; wrote empty outputs for {base}")
                continue

            page = pages[0]
            words = _collect_words_from_export(page)
            rows = _cluster_rows(words, y_tol=y_tol)

            # --- GRID @ cols_per_line = 180 ---
            # derive report_base + page_no from `base`
            m = re.match(r"^(.*)_page_([0-9]+)$", base, flags=re.IGNORECASE)
            if m:
                report_base, page_no = m.group(1), int(m.group(2))
            else:
                report_base, page_no = base, 1

            # filenames you want
            grid_name   = f"{report_base}_aligned_grid_page_{page_no}.txt"
            spaced_name = f"{report_base}_aligned_spaced_page_{page_no}.txt"

            # --- GRID @ cols_per_line = 180 ---
            grid_txt = make_aligned_text(rows, cols_per_line=180, factor=1)
            with open(os.path.join(out_dir, grid_name), "w", encoding="utf-8") as f:
                f.write(grid_txt)

            # --- GAP-SPACED @ char_scale = 140 ---
            spaced_txt = make_spaced_text(rows, char_scale=140, min_spaces=1)
            with open(os.path.join(out_dir, spaced_name), "w", encoding="utf-8") as f:
                f.write(spaced_txt)

            print(f"✅ Saved: {grid_name}  &  {spaced_name}  in {out_dir}")


        time.sleep(0.2)

# --- Usage ---
def main(img_root_dir=None, output_root_dir=None):
    """Main function for command-line usage"""
    if img_root_dir is None:
        img_root_dir = input("Enter path to input root folder: ").strip()
    if output_root_dir is None:
        output_root_dir = input("Enter path to output folder: ").strip()
    
    try:
        extract_images_to_text(
            img_root_dir,
            output_root_dir=output_root_dir,
            batch_size=2,
            y_tol=0.012,     # adjust if rows merge/split
            upscale_factor=1 # keep 1 for your 300-dpi PNG/JPG inputs
        )
        return {"success": True, "message": f"Successfully processed OCR from {img_root_dir}"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

if __name__ == "__main__":
    result = main()
    print(result["message"])
