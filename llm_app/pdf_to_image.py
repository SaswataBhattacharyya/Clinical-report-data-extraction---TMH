#!/usr/bin/env python3
import os
import re
import shutil
import subprocess
import logging
import tempfile
from tqdm import tqdm
from pdf2image import convert_from_path

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- HELPERS ---
def safe_filename(filename: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", filename)

def run(cmd, **kwargs):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)

# ---- IMAGE CONVERSION ----
def convert_pdf_to_images(pdf_path, base_name, relative_dir, image_output_dir):
    """Convert first 4 pages of a PDF to JPEGs under image_output_dir/relative_dir."""
    try:
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=4)
        output_subdir = os.path.join(image_output_dir, relative_dir)
        os.makedirs(output_subdir, exist_ok=True)

        for i, image in enumerate(images, start=1):
            image_name = f"{safe_filename(base_name)}_page_{i}.jpg"
            image_path = os.path.join(output_subdir, image_name)
            image.save(image_path, "JPEG", quality=85, optimize=True)

        logging.info(f"Converted {pdf_path} to {len(images)} image(s).")
    except Exception as e:
        logging.error(f"PDF conversion failed for {pdf_path}: {e}")

def office_to_pdf(off_path, out_dir):
    """
    Convert .doc/.docx to PDF using LibreOffice headless.
    Returns path to the generated PDF, or None on failure.
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        cmd = ['soffice', '--headless', '--convert-to', 'pdf', '--outdir', out_dir, off_path]
        r = run(cmd)
        if r.returncode != 0:
            cmd[0] = 'libreoffice'
            r = run(cmd)
        if r.returncode != 0:
            logging.error(f"LibreOffice conversion failed for {off_path}: {r.stderr.strip()}")
            return None

        base = os.path.splitext(os.path.basename(off_path))[0]
        pdf_guess = os.path.join(out_dir, base + '.pdf')
        if not os.path.exists(pdf_guess):
            candidates = [p for p in os.listdir(out_dir) if p.lower().endswith('.pdf') and p.startswith(base)]
            if not candidates:
                logging.error(f"No PDF found after conversion for {off_path}")
                return None
            pdf_guess = os.path.join(out_dir, candidates[0])
        return pdf_guess
    except FileNotFoundError:
        logging.error("LibreOffice/soffice not found on PATH. Install it to enable DOC/DOCX→PDF.")
        return None
    except Exception as e:
        logging.error(f"Office→PDF conversion error for {off_path}: {e}")
        return None

def copy_image_if_supported(src_path, relative_dir, image_output_dir):
    """Copy .jpg/.jpeg/.png files into image_output_dir/relative_dir preserving name."""
    try:
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in ('.jpg', '.jpeg', '.png'):
            return False
        out_dir = os.path.join(image_output_dir, relative_dir)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        logging.info(f"Copied image: {src_path} -> {dst}")
        return True
    except Exception as e:
        logging.error(f"Failed to copy image {src_path}: {e}")
        return False

def copy_excel_if_supported(src_path, relative_dir):
    """Copy Excel/CSV files into EXCEL_OUTPUT_DIR/relative_dir preserving name."""
    try:
        ext = os.path.splitext(src_path)[1].lower()
        if ext not in ('.xlsx', '.xls', '.xlsm', '.csv'):
            return False
        out_dir = os.path.join(EXCEL_OUTPUT_DIR, relative_dir)
        os.makedirs(out_dir, exist_ok=True)
        dst = os.path.join(out_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dst)
        logging.info(f"Copied excel: {src_path} -> {dst}")
        return True
    except Exception as e:
        logging.error(f"Failed to copy excel {src_path}: {e}")
        return False

def main(input_root_dir=None, image_output_dir=None):
    if input_root_dir is None:
        input_root_dir = input("Enter the root folder path: ").strip()
    if image_output_dir is None:
        image_output_dir = os.path.join(os.getcwd(), 'pdf_image')
    
    return convert_pdfs_to_images(input_root_dir, image_output_dir)

def convert_pdfs_to_images(input_root_dir, image_output_dir):
    """Main function to convert PDFs/DOCs to images"""
    os.makedirs(image_output_dir, exist_ok=True)
    
    try:
        for root, _, files in os.walk(input_root_dir):
            rel_dir = os.path.relpath(root, input_root_dir)

            for filename in tqdm(files, desc=f"Processing {root}"):
                ext = os.path.splitext(filename)[1].lower()
                file_path = os.path.join(root, filename)
                base_name = os.path.splitext(os.path.basename(filename))[0]

                # 1) Direct copy: images
                if copy_image_if_supported(file_path, rel_dir, image_output_dir):
                    continue

                # 2) Direct copy: excel/csv (skip for now as not needed in pipeline)
                # if copy_excel_if_supported(file_path, rel_dir):
                #     continue

                # 3) Convert DOC/DOCX → PDF → images
                if ext in ('.docx', '.doc'):
                    with tempfile.TemporaryDirectory() as tmpd:
                        pdf_path = office_to_pdf(file_path, tmpd)
                        if pdf_path:
                            convert_pdf_to_images(pdf_path, base_name, rel_dir, image_output_dir)
                    continue

                # 4) Convert existing PDF → images
                if ext == '.pdf':
                    convert_pdf_to_images(file_path, base_name, rel_dir, image_output_dir)
                    continue

                logging.info(f"Skipped unsupported file: {file_path}")
        
        return {"success": True, "message": f"Successfully processed files from {input_root_dir}"}
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}"}

if __name__ == "__main__":
    result = main()
    print(result["message"])
