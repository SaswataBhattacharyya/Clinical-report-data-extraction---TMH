#!/usr/bin/env python3
"""
Streamlit App for LLM-based Document Processing Pipeline
Orchestrates: PDF→Image, OCR, LLM Parsing, Excel Conversion
"""

import io
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from copy import deepcopy
from itertools import product
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pyreadstat
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
CODES_MODEL_DIR = ROOT_DIR / "codes_model"
if str(CODES_MODEL_DIR) not in sys.path:
    sys.path.append(str(CODES_MODEL_DIR))

# Import refactored modules
from check import check_gpu
from pdf_to_image import convert_pdfs_to_images
from doctr_ext_batch import extract_images_to_text
from textfile_collate import main as collate_main
from gpt_parser_new import main as gpt_main
from llama_deepseek import main as llama_main
from parse_to_excel import main as parse_excel_main
from model_use import run_inference
from data_proc import process_dataset
from train import train_model


def ensure_dataset_structure(dataset_root: Path) -> bool:
    """Check that dataset_root contains the required subfolders."""
    return (
        (dataset_root / "input_all").exists()
        and (dataset_root / "input_crop").exists()
    )


def merge_dataset_zip(upload_file, dataset_root: Path):
    """Merge uploaded dataset ZIP into the dataset_root."""
    logs = []
    if upload_file is None:
        return True, "No additional dataset uploaded.", logs

    try:
        with tempfile.TemporaryDirectory(prefix="dataset_upload_") as tmpdir:
            tmpdir_path = Path(tmpdir)
            zip_path = tmpdir_path / upload_file.name
            zip_path.write_bytes(upload_file.getvalue())

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir_path)

            candidate_root = None
            for path in tmpdir_path.rglob("input_all"):
                root = path.parent
                if (root / "input_crop").exists():
                    candidate_root = root
                    break

            if candidate_root is None:
                msg = "Uploaded archive must contain 'input_all' and 'input_crop' folders."
                logs.append(f"[ERROR] {msg}")
                return False, msg, logs

            for sub in ["input_all", "input_crop"]:
                src_dir = candidate_root / sub
                if not src_dir.exists():
                    msg = f"Uploaded archive missing folder: {sub}"
                    logs.append(f"[ERROR] {msg}")
                    return False, msg, logs

                dst_dir = dataset_root / sub
                dst_dir.mkdir(parents=True, exist_ok=True)

                for item in src_dir.iterdir():
                    target = dst_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)

            msg = "Dataset merged successfully."
            logs.append(f"[INFO] {msg}")
            return True, msg, logs
    except Exception as e:
        msg = f"Failed to merge dataset: {e}"
        logs.append(f"[ERROR] {msg}")
        return False, msg, logs

# Page config
st.set_page_config(
    page_title="LLM Document Processing Pipeline",
    page_icon="📄",
    layout="wide"
)

st.title("📄 LLM Document Processing Pipeline")
st.markdown("Process PDFs through OCR and LLM to extract structured data")

# Initialize session state
if "processing" not in st.session_state:
    st.session_state.processing = False
if "temp_folders" not in st.session_state:
    st.session_state.temp_folders = []
if "current_step" not in st.session_state:
    st.session_state.current_step = -1

# =========================
# STEPPER COMPONENT
# =========================
def render_process_stepper(steps, current_step=-1):
    """Render a visual step-by-step progress stepper using Streamlit components"""
    num_steps = len(steps)
    
    # Inject CSS for styling
    st.markdown("""
    <style>
    .stepper-wrapper {
        position: relative;
        padding: 40px 0 60px 0;
        margin: 20px 0;
    }
    .stepper-line {
        position: absolute;
        top: 210px;
        left: 130px;
        right: 130px;
        height: 3px;
        background-color: #000;
        z-index: 0;
    }
    .stepper-step-container {
        position: relative;
        z-index: 2;
    }
    .stepper-circle-div {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 32px;
        color: white;
        margin: 0 auto 0 auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        position: relative;
        z-index: 2;
    }
    .stepper-circle-gray {
        background-color: #D9D9D9;
    }
    .stepper-circle-green {
        background-color: #47DD10;
    }
    .stepper-label-div {
        text-align: center;
        margin-top: 10px;
        font-size: 14px;
        font-weight: 500;
        color: #333;
        word-wrap: break-word;
        position: relative;
        z-index: 2;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create wrapper with line
    st.markdown('<div class="stepper-wrapper"><div class="stepper-line"></div>', unsafe_allow_html=True)
    
    # Create columns for each step
    cols = st.columns(num_steps)
    
    for i, step in enumerate(steps):
        with cols[i]:
            is_completed = i <= current_step
            circle_color = "stepper-circle-green" if is_completed else "stepper-circle-gray"
            circle_content = "✓" if is_completed else ""
            
            # Render step container with circle
            st.markdown(
                f'<div class="stepper-step-container"><div class="stepper-circle-div {circle_color}">{circle_content}</div>',
                unsafe_allow_html=True
            )
            
            # Render label
            st.markdown(
                f'<div class="stepper-label-div">{step}</div></div>',
                unsafe_allow_html=True
            )
    
    # Close wrapper
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# CONFIGURATION SECTION
# =========================
st.header("⚙️ Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📁 Input/Output Paths")
    
    input_path = st.text_input(
        "INPUT Folder (PDFs/DOCs)",
        placeholder="/path/to/pdfs",
        help="Folder containing PDF or DOC files to process"
    )
    
    output_path = st.text_input(
        "OUTPUT Folder (Final Excel)",
        placeholder="/path/to/output",
        help="Folder where final Excel files will be saved"
    )
    
    excel2_path = st.text_input(
        "EXCEL2 Path (MRN Mapping)",
        placeholder="/path/to/mrn_mapping.xlsx",
        help="Excel file with 'filename' and 'MRN' columns for mapping"
    )

with col2:
    st.subheader("🔧 Processing Options")
    
    use_dl = st.toggle(
        "Enable DL Model",
        value=False,
        help="Use deep learning model for image sorting (requires GPU)"
    )
    
    llm_type = st.selectbox(
        "LLM Type",
        ["OpenAI", "Deepseek", "Llama"],
        help="Select the LLM provider for text extraction"
    )
    
    # Show API key input only for OpenAI
    api_key = None
    if llm_type == "OpenAI":
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Your OpenAI API key"
        )
    elif llm_type in ["Deepseek", "Llama"]:
        st.info("💡 Using local Ollama. Ensure Ollama is running and models are installed.")
    
    delete_temp = st.checkbox(
        "Delete Intermediate Folders",
        value=True,
        help="Clean up temporary folders after processing"
    )

# =========================
# PROCESSING SECTION
# =========================
st.header("🚀 Processing Pipeline")

# Show preview stepper - always visible
st.markdown("**📋 Processing Steps Preview:**")
if use_dl:
    preview_steps = [
        "pdf to image",
        "DL sorting",
        "image to text",
        "text collation",
        f"{llm_type} extraction",
        "parse to excel",
        "All done"
    ]
else:
    preview_steps = [
        "pdf to image",
        "image to text",
        "text collation",
        f"{llm_type} extraction",
        "parse to excel",
        "All done"
    ]

render_process_stepper(preview_steps, -1)  # Show all gray (not started)
st.markdown("---")

# Validate inputs
can_start = bool(input_path and output_path and excel2_path)
if llm_type == "OpenAI" and not api_key:
    can_start = False
    st.warning("⚠️ Please provide OpenAI API key")

if st.button("▶️ Start Processing", type="primary", disabled=not can_start or st.session_state.processing):
    st.session_state.processing = True
    st.session_state.current_step = -1
    
    # Define steps based on DL toggle
    if use_dl:
        steps = [
            "pdf to image",
            "DL sorting",
            "image to text",
            "text collation",
            f"{llm_type} extraction",
            "parse to excel",
            "All done"
        ]
    else:
        steps = [
            "pdf to image",
            "image to text",
            "text collation",
            f"{llm_type} extraction",
            "parse to excel",
            "All done"
        ]
    
    # Create an empty placeholder for the processing stepper (will update in place)
    stepper_placeholder = st.empty()
    
    # Function to update stepper in the placeholder
    def update_stepper():
        with stepper_placeholder.container():
            st.markdown("---")
            render_process_stepper(steps, st.session_state.current_step)
            st.markdown("---")
    
    # Display initial stepper
    update_stepper()
    
    # Create temp folders
    base_dir = os.path.dirname(os.path.abspath(input_path)) if input_path else os.getcwd()
    out_image_temp = os.path.join(base_dir, "out_image_temp")
    images_selected = os.path.join(base_dir, "images_selected")
    filecollate_temp = os.path.join(base_dir, "filecollate_temp")
    gpt_input_temp = os.path.join(base_dir, "gpt_input_temp")
    gpt_out = os.path.join(base_dir, "gpt_out")
    
    st.session_state.temp_folders = [
        out_image_temp,
        images_selected,
        filecollate_temp,
        gpt_input_temp,
        gpt_out,
    ]
    
    progress_bar = st.progress(0)
    status_container = st.container()
    
    try:
        # =========================
        # STEP 1: GPU Check
        # =========================
        with status_container:
            with st.status("🔍 Step 1: Checking GPU Availability...", expanded=True) as status:
                gpu_result = check_gpu()
                if gpu_result["available"]:
                    st.success(f"✅ GPU Available: {gpu_result['device_name']}")
                    if not use_dl:
                        st.info("💡 DL model is disabled, GPU won't be used")
                else:
                    st.warning("⚠️ No GPU detected - DL not recommended!")
                    if use_dl:
                        st.warning("⚠️ Continuing with DL anyway, but it may be slow")
                status.update(label="✅ Step 1 Complete: GPU Check", state="complete")
        
        progress_bar.progress(10)
        
        # =========================
        # STEP 2: PDF to Images
        # =========================
        with status_container:
            with st.status("🖼️ Step 2: Converting PDFs to Images...", expanded=True) as status:
                result = convert_pdfs_to_images(input_path, out_image_temp)
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
                    raise Exception(result["message"])
                status.update(label="✅ Step 2 Complete: PDF to Images", state="complete")
        
        st.session_state.current_step = 0  # PDF to image completed
        update_stepper()
        progress_bar.progress(30)
        
        # =========================
        # STEP 3A: DL Model (if enabled)
        # =========================
        if use_dl:
            with status_container:
                with st.status("🤖 Step 3A: Running DL Model...", expanded=True) as status:
                    ckpt_path = CODES_MODEL_DIR / "runs" / "exp_cnn_512x352" / "best.pt"
                    if not ckpt_path.exists():
                        msg = f"Checkpoint not found at {ckpt_path}"
                        status.write(f"❌ {msg}")
                        st.error(msg)
                        raise FileNotFoundError(msg)
                    inference_result = run_inference(
                        img_root_dir=out_image_temp,
                        out_main_dir=images_selected,
                        ckpt_path=ckpt_path,
                        confidence=0.5,
                        temp_dirname="_temp_predictions",
                    )
                    for log_entry in inference_result.get("logs", []):
                        status.write(log_entry)
                    if not inference_result.get("success", False):
                        st.error(inference_result.get("message", "DL model inference failed"))
                        raise Exception(inference_result.get("message", "DL model inference failed"))
                    # Cleanup temp predictions to avoid leaking downstream
                    temp_dir_path = inference_result.get("temp_dir")
                    if temp_dir_path and os.path.exists(temp_dir_path):
                        try:
                            shutil.rmtree(temp_dir_path)
                        except Exception as cleanup_err:
                            status.write(f"⚠️ Could not delete temp folder {temp_dir_path}: {cleanup_err}")
                    st.success(inference_result.get("message", "DL inference complete"))
                    status.update(label="✅ Step 3A Complete: DL Model", state="complete")
            
            st.session_state.current_step = 1  # DL sorting completed
            update_stepper()
            progress_bar.progress(40)
            ocr_input = images_selected
        else:
            ocr_input = out_image_temp
            progress_bar.progress(40)
        
        # =========================
        # STEP 3B: OCR (Doctr)
        # =========================
        with status_container:
            with st.status("👁️ Step 3B: Running OCR (Doctr)...", expanded=True) as status:
                try:
                    extract_images_to_text(
                        ocr_input,
                        output_root_dir=filecollate_temp,
                        batch_size=2,
                        y_tol=0.012,
                        upscale_factor=1
                    )
                    st.success(f"✅ OCR completed, text files saved to {filecollate_temp}")
                except Exception as e:
                    st.error(f"❌ OCR failed: {str(e)}")
                    raise
                status.update(label="✅ Step 3B Complete: OCR", state="complete")
        
        # Update stepper: image to text completed (step 1 if no DL, step 2 if DL)
        st.session_state.current_step = 1 if not use_dl else 2
        update_stepper()
        progress_bar.progress(50)
        
        # =========================
        # STEP 4: Text File Collation
        # =========================
        with status_container:
            with st.status("📝 Step 4: Collating Text Files...", expanded=True) as status:
                try:
                    result = collate_main(
                        input_root=filecollate_temp,
                        output_root=gpt_input_temp
                    )
                    if result["success"]:
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
                        raise Exception(result["message"])
                except Exception as e:
                    st.error(f"❌ Text collation failed: {str(e)}")
                    raise
                status.update(label="✅ Step 4 Complete: Text Collation", state="complete")
        
        # Update stepper: text collation completed (step 2 if no DL, step 3 if DL)
        st.session_state.current_step = 2 if not use_dl else 3
        update_stepper()
        progress_bar.progress(65)
        
        # =========================
        # STEP 5: LLM Parsing
        # =========================
        with status_container:
            llm_status_label = f"🧠 Step 5: Running LLM Parsing ({llm_type})..."
            with st.status(llm_status_label, expanded=True) as status:
                if llm_type == "OpenAI":
                    result = gpt_main(
                        input_dir=gpt_input_temp,
                        output_dir=gpt_out,
                        api_key=api_key
                    )
                elif llm_type == "Deepseek":
                    result = llama_main(
                        input_dir=gpt_input_temp,
                        output_dir=gpt_out,
                        model="deepseek-r1:latest"
                    )
                elif llm_type == "Llama":
                    result = llama_main(
                        input_dir=gpt_input_temp,
                        output_dir=gpt_out,
                        model="llama3:latest"
                    )
                else:
                    st.error(f"❌ Unknown LLM type: {llm_type}")
                    raise Exception(f"Unknown LLM type: {llm_type}")
                
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
                    raise Exception(result["message"])
                status.update(label=f"✅ Step 5 Complete: LLM Parsing ({llm_type})", state="complete")
        
        # Update stepper: LLM extraction completed (step 3 if no DL, step 4 if DL)
        st.session_state.current_step = 3 if not use_dl else 4
        update_stepper()
        progress_bar.progress(80)
        
        # =========================
        # STEP 6: Parse to Excel
        # =========================
        with status_container:
            with st.status("📊 Step 6: Converting to Excel...", expanded=True) as status:
                result = parse_excel_main(
                    input_root=gpt_out,
                    output_root=output_path,
                    excel2_path=excel2_path
                )
                if result["success"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
                    raise Exception(result["message"])
                status.update(label="✅ Step 6 Complete: Excel Conversion", state="complete")
        
        # Update stepper: parse to excel completed (step 4 if no DL, step 5 if DL)
        st.session_state.current_step = 4 if not use_dl else 5
        update_stepper()
        progress_bar.progress(100)
        
        # =========================
        # CLEANUP
        # =========================
        if delete_temp:
            with status_container:
                with st.status("🧹 Cleaning up temporary folders...", expanded=False) as status:
                    for temp_folder in st.session_state.temp_folders:
                        if os.path.exists(temp_folder):
                            try:
                                shutil.rmtree(temp_folder)
                                st.info(f"🗑️ Deleted: {temp_folder}")
                            except Exception as e:
                                st.warning(f"⚠️ Could not delete {temp_folder}: {e}")
                    status.update(label="✅ Cleanup Complete", state="complete")
        
        # Mark all steps as complete
        st.session_state.current_step = len(steps) - 1
        update_stepper()
        
        # Show final completion message
        st.markdown("""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 15px; margin: 20px 0;">
            <h1 style="color: white; font-size: 48px; margin: 0;">All Done Bro 👍</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.success("🎉 Processing Complete! Check your output folder.")
        st.session_state.processing = False
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        st.session_state.processing = False
        if delete_temp:
            st.info("💡 Intermediate folders preserved due to error")

st.divider()
st.header("🛠️ DL Model Training")
st.markdown(
    """
    Upload additional data (optional) and retrain the CNN importance classifier.
    The dataset should contain two folders:
    - `input_all`: all uncropped pages
    - `input_crop`: cropped/annotated important regions
    """
)

train_toggle = st.checkbox("Enable DL Model Training", value=False)

dataset_root = CODES_MODEL_DIR / "dataset"
processed_data_root = CODES_MODEL_DIR / "processed_data"
default_runs_dir = CODES_MODEL_DIR / "runs" / "exp_cnn_512x360"

if train_toggle:
    dataset_upload = st.file_uploader(
        "Upload additional dataset (.zip)", type=["zip"], accept_multiple_files=False
    )
    st.caption("If no file is uploaded, the existing dataset will be used.")

    start_training = st.button("🚀 Start Training", type="primary")

    if start_training:
        dataset_root.mkdir(parents=True, exist_ok=True)
        merge_success, merge_message, merge_logs = merge_dataset_zip(dataset_upload, dataset_root)
        if merge_message:
            st.info(merge_message)
        for entry in merge_logs:
            st.write(entry)
        if not merge_success:
            st.error(merge_message)

        structure_ok = ensure_dataset_structure(dataset_root)
        if not structure_ok:
            st.error(
                "The dataset must contain 'input_all' and 'input_crop' folders. "
                f"Current dataset root: {dataset_root}"
            )

        gpu_info = check_gpu()
        if gpu_info["available"]:
            st.success(f"GPU Available: {gpu_info['device_name']}")
        else:
            st.error("❌ No GPU found! Model training requires a GPU.")

        if merge_success and structure_ok and gpu_info["available"]:
            processed_data_root.mkdir(parents=True, exist_ok=True)
            training_failed = False

            with st.status("📦 Preparing dataset...", expanded=True) as status:
                prep_result = process_dataset(dataset_root, processed_data_root)
                for log_entry in prep_result.get("logs", []):
                    status.write(log_entry)
                if prep_result.get("success", False):
                    status.update(label="✅ Dataset prepared", state="complete")
                else:
                    status.update(label="❌ Dataset preparation failed", state="error")
                    st.error(prep_result.get("message", "Dataset preparation failed"))
                    training_failed = True

            if not training_failed:
                with st.status("🧠 Training CNN model...", expanded=True) as status:
                    train_result = train_model(
                        data_root=processed_data_root,
                        save_dir=default_runs_dir,
                        arch="cnn",
                        size_h=512,
                        size_w=360,
                        epochs=12,
                        batch_size=16,
                        lr=1e-3,
                        seed=42,
                        pos_weight=1.0,
                        oversample_crops=False,
                        crop_weight=1.0,
                    )
                    for log_entry in train_result.get("logs", []):
                        status.write(log_entry)
                    if train_result.get("success", False):
                        status.update(label="✅ Training complete", state="complete")
                    else:
                        status.update(label="❌ Training failed", state="error")
                        st.error(train_result.get("message", "Training failed"))
                        training_failed = True

            if not training_failed:
                history = train_result.get("history", [])
                if history:
                    hist_df = pd.DataFrame(history).set_index("epoch")
                    st.subheader("Training Accuracy")
                    st.line_chart(hist_df[["train_acc", "val_acc"]], use_container_width=True)
                    st.subheader("Training Loss")
                    st.line_chart(hist_df[["train_loss", "val_loss"]], use_container_width=True)

                results = train_result.get("results", {})
                if results:
                    st.subheader("Validation Metrics")
                    st.json(results.get("val_metrics", {}))
                    st.subheader("Test Metrics")
                    st.json(results.get("test_metrics", {}))

                st.success("Trained Bro 👍")
                st.write(f"Best checkpoint saved at `{train_result.get('best_checkpoint')}`")
        else:
            st.warning("Resolve the issues above to start training.")

# =========================
# INFO SECTION
# =========================
with st.expander("ℹ️ Pipeline Information"):
    st.markdown("""
    ### Processing Steps:
    1. **GPU Check**: Verifies CUDA/GPU availability
    2. **PDF to Images**: Converts PDFs/DOCs to images
    3. **DL Model** (optional): Sorts/processes images using deep learning
    4. **OCR**: Extracts text from images using Doctr
    5. **LLM Parsing**: Extracts structured data using selected LLM
    6. **Excel Conversion**: Converts LLM outputs to Excel with MRN mapping
    
    ### LLM Options:
    - **OpenAI**: Uses GPT-4o via API (requires API key)
    - **Deepseek**: Uses deepseek-r1:latest via Ollama (local)
    - **Llama**: Uses llama3:latest via Ollama (local)
    
    ### Requirements:
    - For OpenAI: API key required
    - For Deepseek/Llama: Ollama must be installed and running
    - For DL Model: GPU recommended
    """)

