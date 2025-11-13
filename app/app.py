# app.py
# Streamlit Excel -> SPSS (.sav) converter with filters, wide flattening, encoding, and schema
# ------------------------------------------------------------
# Usage: streamlit run app.py
# Requires: pip install streamlit pandas pyreadstat openpyxl

import io
import math
import os
import re
import shutil
import tempfile
from copy import deepcopy

import pandas as pd
import streamlit as st
import pyreadstat
from excel_spss_new_format_copy import merge_sb_format
from excel_spss_format_dr_minit import convert_long_to_dr_minit_format, get_target_sheets_from_specimen_vaf, process_case_group

# =========================
# ======= CONFIG ==========
# =========================
# Unique patient identifier (grouping key)

def _infer_spss_format(col: str) -> str:
    c = col.lower()

    # Dates
    if "date" in c:
        return "DATE10"          # SPSS DATE dd-mmm-yyyy (pyreadstat supports DATE10)

    # Explicit integers
    if c in {"age", "serial"}:
        return "F8.0"

    # Integer-ish buckets
    if any(k in c for k in ["count", "serial", "exon_no", "year_of_ngs", "age_recode"]):
        return "F8.0"

    # Continuous numeric (floats)
    if any(k in c for k in [
        "vaf", "score", "fraction", "read_count", "tumor_fraction", "depth"]):
        return "F10.2"

    # Copy numbers explicitly tend to be fractional
    if "copy" in c:
        return "F10.2"

    # Tiers as numbers unless a "name" field
    if ("tier" in c) and ("name" not in c):
        return "F8.0"

    # Everything else: generous string length
    return "A1000"

def build_schema_from_columns(columns) -> dict:
    """
    Build SCHEMA dict (col -> SPSS format) from a list of column names
    using the pattern rules above.
    """
    return {col: _infer_spss_format(str(col)) for col in columns}

GENE_COLUMNS = [
    "Gene_name", "Mutation(nucleotide)", "Mutation(protein)",
    "VAF (tissue)", "VAF (liq)", "VAF (pf)", "VAF (csf)", "VAF (others)",
    "Exon_no", "NM ID", "Fusion",
    "Fusion_Read_Count (tissue)", "Fusion_Read_Count (liq)", "Fusion_Read_Count (pf)",
    "Fusion_Read_Count (csf)", "Fusion_Read_Count (others)",
    "Amplification", "Amp_read/copy number (tissue)", "Amp_read/copy number (liq)",
    "Amp_read/copy number (pf)", "Amp_read/copy number (csf)", "Amp_read/copy number (others)",
    "Variant Type", "Tier"
]

MRN_COL = "MRN"

# Columns to show as filters (user will select values from unique list)
# Example: ["Gene", "Mutation", "Specimen", "Variant Type", ...]
FILTER_COLS = [
    "Gene_name","Specimen", "Variant Type", "Tier", "NGS_Tech_used", "Reporting_company", "Gender", "Age"
]

# Dependent filter map: independent -> [dependents]
# A dependent filter is disabled until its independent has at least one selection.
# Example: {"Gene": ["Mutation"]}
DEP_FILTER_COLS = {
    "Gene_name": ["Mutation(nucleotide)", "Mutation(protein)", "Exon_no"]
}

# Wide flattening (single columns)
# Example: ["Gene", "Mutation", "VAF (liq)"]
WIDE_COLS = [
    # "Gene", "Mutation"
]

# Wide flattening (paired sets) — list of lists
# Example: [
#   ["Gene", "Mutation_nucleotide", "Mutation_protein", "VAF (liq)", "VAF (solid)", "VAF (pf)"],
#   ["Review_previous", "Review_post"]
# ]
WIDE_PAIR_COLS = [
    # ["Gene", "Mutation_nucleotide", "Mutation_protein", "VAF (liq)", "VAF (solid)", "VAF (pf)"]
]

# Columns to encode to numeric (value labels created)
# NOTE: Any derived columns like Gene_1..Gene_n will also be encoded if base in ENCODE_COLUMNS
# All columns will be encoded EXCEPT those listed in DO_NOT_ENCODE_COLUMNS

# Columns that should NOT be encoded (keep as text/numeric)
DO_NOT_ENCODE_COLUMNS = [
    # Administrative columns
    "Serial", "Data_Entry_Done_By", "MRN", "Date_Discussed_MTB", 
    "Addictions_Others", "Date_Diagnosis", "ROS_H_Score", "PDL1_Absolute_Value",
    "Source_File", "Age", "Name",
    
    # Treatment dates and regimens
    "Previous_Treatment_Line_1_Start_Date", "Previous_Treatment_Line_1_Stop_Date", 
    "Previous_Treatment_Line_1_Regimen", "Previous_Treatment_Line_2_Start_Date", 
    "Previous_Treatment_Line_2_Stop_Date", "Previous_Treatment_Line_2_Regimen", 
    "Previous_Treatment_Line_3_Start_Date", "Previous_Treatment_Line_3_Stop_Date", 
    "Previous_Treatment_Line_3_Regimen", "Previous_Treatment_Line_4_Start_Date", 
    "Previous_Treatment_Line_4_Stop_Date", "Previous_Treatment_Line_4_Regimen", 
    "MTB_Treatment_Plan", "Alternative_Treatment_Plan_Regimen", 
    "Post_MTB_Treatment_Line_1_Start_Date", "Post_MTB_Treatment_Line_1_Stop_Date", 
    "Post_MTB_Treatment_Line_1_Regimen", "Post_MTB_Treatment_Line_2_Start_Date", 
    "Post_MTB_Treatment_Line_2_Stop_Date", "Post_MTB_Treatment_Line_2_Regimen", 
    "Post_MTB_Treatment_Line_3_Start_Date", "Post_MTB_Treatment_Line_3_Stop_Date", 
    "Post_MTB_Treatment_Line_3_Regimen", "Post_MTB_Treatment_Line_4_Start_Date", 
    "Post_MTB_Treatment_Line_4_Stop_Date", "Post_MTB_Treatment_Line_4_Regimen", 
    "Date_OS_Final", "VAR00013", "VAR00001"
]

def get_columns_to_encode(df_columns):
    """
    Determine which columns should be encoded based on the DO_NOT_ENCODE_COLUMNS list.
    Excludes columns with 'read', 'copy', 'count', 'tier', 'vaf', 'depth', 'date', 'rna', 'patient' in their names (case insensitive).
    """
    columns_to_encode = []
    
    for col in df_columns:
        col_str = str(col).lower()
        
        # Skip if in DO_NOT_ENCODE_COLUMNS
        if col in DO_NOT_ENCODE_COLUMNS:
            continue
            
        # Skip if contains keywords that should not be encoded (case insensitive)
        exclude_keywords = ['read', 'copy', 'count', 'tier', 'vaf', 'depth', 'date', 'rna', 'patient']
        if any(keyword in col_str for keyword in exclude_keywords):
            continue
            
        # Skip if it's already numeric (detected by schema)
        # This will be handled by the schema detection
        
        columns_to_encode.append(col)
    
    return columns_to_encode

# Legacy ENCODE_COLUMNS list - will be dynamically generated
ENCODE_COLUMNS = []

ENCODE_COMPOSITES = [
    ("Gene_name", "Mutation(nucleotide)"),
    ("Gene_name", "Mutation(protein)")
]


# Keep original text before encoding as <col>_text
KEEP_ORIGINAL_TEXT_FOR_ENCODED = True

# SPSS schema: column -> SPSS format
# Strings use 'A{width}' (e.g., 'A100'), numeric use 'F8.0' (int-like) or 'F10.2' (float)
# Build SCHEMA dynamically from the output columns the user is exporting

# =========================
# ====== UTILITIES ========
# =========================
def sort_unique_values(values):
    return sorted(values, key=lambda v: ("" if pd.isna(v) else str(v)).lower())

def build_numeric_codes(series: pd.Series):
    vals = series.dropna().unique().tolist()
    vals = sort_unique_values(vals)
    code_map, labels = {}, {}
    code = 1
    for v in vals:
        s = "" if pd.isna(v) else str(v).strip()
        if s == "": continue
        code_map[v] = code
        labels[code] = s
        code += 1
    return code_map, labels

def apply_schema_types(df, schema):
    out = df.copy()
    for col, fmt in schema.items():
        if col not in out.columns or not fmt:
            continue
        fmtU = fmt.upper()

        # ── v NEW: leave DATE* columns as-is (keep strings); pyreadstat will format using variable_format
        if fmtU.startswith("DATE"):
            continue
        # ── ^ NEW

        if fmtU.startswith("A"):
            out[col] = out[col].astype(str).where(~out[col].isna(), "")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if ".0" in fmtU:
                out[col] = out[col].apply(
                    lambda x: int(x) if pd.notna(x) and float(x).is_integer() else (int(x) if pd.notna(x) else math.nan)
                )
    return out

def inherit_schema_for_derived_columns(df, schema):
    variable_format = {}
    for col in df.columns:
        if col in schema and schema[col]:
            variable_format[col] = schema[col]
            continue
        if "_" in col:
            base = col.split("_")[0]
            if base in schema and schema[base]:
                variable_format[col] = schema[base]
    return variable_format

def encode_columns_with_labels(df, encode_cols, keep_original=True):
    df = df.copy(); variable_value_labels = {}
    targets = set()
    for base in encode_cols:
        for c in df.columns:
            if c == base or c.startswith(f"{base}_"):
                targets.add(c)
    for col in sorted(targets):
        if col not in df.columns: continue
        if keep_original:
            df[f"{col}_text"] = df[col].astype(str)
        code_map, labels = build_numeric_codes(df[col])
        if code_map:
            df[col] = df[col].map(code_map).astype("float")
            variable_value_labels[col] = labels
    return df, variable_value_labels

def filter_dataframe(df, selections, mode):
    active = {c: v for c, v in selections.items() if v}
    if not active: return df
    
    def age_in_bin(age, bin_str):
        """Check if age falls within the specified bin"""
        try:
            age_val = float(age)
            if pd.isna(age_val):
                return False
            
            if bin_str == "<10":
                return age_val < 10
            elif bin_str == "90+":
                return age_val >= 90
            else:
                # Parse bins like "10-20", "20-30", etc.
                parts = bin_str.split("-")
                if len(parts) == 2:
                    lower = float(parts[0])
                    upper = float(parts[1])
                    return lower <= age_val < upper
            return False
        except (ValueError, TypeError):
            return False
    
    # Handle multi-gene filtering and Age filtering
    gene_selections = {}
    age_selections = []
    other_selections = {}
    
    for col, vals in active.items():
        if col == "Age":
            # Age bins to filter
            age_selections = vals
        elif "_" in col and any(gene in col for gene in df["Gene_name"].dropna().unique()):
            # This is a gene-specific filter (e.g., "Mutation(nucleotide)_EGFR")
            gene_selections[col] = vals
        else:
            # This is a regular filter
            other_selections[col] = vals
    
    if mode == "Row-wise":
        mask = pd.Series([True]*len(df), index=df.index)
        
        # Apply Age filter
        if age_selections and "Age" in df.columns:
            age_mask = pd.Series([False]*len(df), index=df.index)
            for bin_str in age_selections:
                age_mask |= df["Age"].apply(lambda x: age_in_bin(x, bin_str))
            mask &= age_mask
        
        # Apply regular filters
        for col, vals in other_selections.items():
            mask &= df[col].isin(vals)
        
        # Apply gene-specific filters
        for col, vals in gene_selections.items():
            if "_" in col:
                # Extract gene and mutation type from column name
                parts = col.split("_")
                if len(parts) >= 2:
                    gene = "_".join(parts[1:])  # Handle genes with underscores
                    mutation_type = parts[0]
                    
                    # Filter rows where Gene_name matches AND mutation type matches
                    gene_mask = df["Gene_name"] == gene
                    mutation_mask = df[mutation_type].isin(vals)
                    mask &= (gene_mask & mutation_mask)
        
        return df[mask]
    
    else:  # Patient-wise filtering
        tmp = df.copy()
        mask = pd.Series([True]*len(tmp), index=tmp.index)
        
        # Apply Age filter
        if age_selections and "Age" in tmp.columns:
            age_mask = pd.Series([False]*len(tmp), index=tmp.index)
            for bin_str in age_selections:
                age_mask |= tmp["Age"].apply(lambda x: age_in_bin(x, bin_str))
            mask &= age_mask
        
        # Apply regular filters
        for col, vals in other_selections.items():
            mask &= tmp[col].isin(vals)
        
        # Apply gene-specific filters
        for col, vals in gene_selections.items():
            if "_" in col:
                parts = col.split("_")
                if len(parts) >= 2:
                    gene = "_".join(parts[1:])
                    mutation_type = parts[0]
                    
                    gene_mask = tmp["Gene_name"] == gene
                    mutation_mask = tmp[mutation_type].isin(vals)
                    mask &= (gene_mask & mutation_mask)
        
        matching = set(tmp.loc[mask, MRN_COL].dropna().unique().tolist())
        return df[df[MRN_COL].isin(matching)]

def _enumerate_series_values(series):
    return [x for x in series.tolist() if pd.notna(x) and str(x).strip() != ""]

def flatten_wide_single(df, group_key, cols):
    frames, flat_dicts, index_values = [], [], []
    grouped = df.groupby(group_key, dropna=False, sort=False)
    for mrn, g in grouped:
        d = {}
        for col in cols:
            vals = _enumerate_series_values(g[col] if col in g.columns else pd.Series([],dtype=object))
            for i,v in enumerate(vals,start=1):
                d[f"{col}_{i}"] = v
        index_values.append(mrn); flat_dicts.append(d)
    out = pd.DataFrame(flat_dicts, index=index_values).reset_index()
    out.rename(columns={"index": group_key}, inplace=True); return out

def flatten_wide_pairs(df, group_key, pair_sets):
    grouped = df.groupby(group_key, dropna=False, sort=False)
    index_values, flat_rows = [], []
    for mrn, g in grouped:
        row = {}
        for colset in pair_sets:
            lists = [ _enumerate_series_values(g[c] if c in g.columns else pd.Series([],dtype=object)) for c in colset ]
            K = max((len(x) for x in lists), default=0)
            for i in range(1,K+1):
                for col, vals in zip(colset, lists):
                    row[f"{col}_{i}"] = vals[i-1] if i-1 < len(vals) else ""
        index_values.append(mrn); flat_rows.append(row)
    out = pd.DataFrame(flat_rows, index=index_values).reset_index()
    out.rename(columns={"index": group_key}, inplace=True); return out

def to_gene_wide_format(df, mrn_col, gene_col, gene_columns):
    wide_rows, long_rows = [], []
    for mrn, g in df.groupby(mrn_col, dropna=False, sort=False):
        row = {mrn_col: mrn}
        seen_genes = set()
        for _, subrow in g.iterrows():
            gene = subrow.get(gene_col)
            if pd.isna(gene) or str(gene).strip() == "":
                continue
            if gene in seen_genes:
                # fallback: keep this row in long format
                long_rows.append(subrow.to_dict())
                continue
            seen_genes.add(gene)
            for c in gene_columns:
                if c in subrow:
                    row[f"{gene}_{c}"] = subrow[c]
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows).fillna("")
    long_df = pd.DataFrame(long_rows).fillna("") if long_rows else pd.DataFrame()
    return wide_df, long_df

def encode_with_singles_and_composites(
    df: pd.DataFrame,
    singles: list[str],
    composites: list[tuple[str, str]],
    keep_original: bool = True,
    joiner: str = " • "
):
    """
    Encodes:
      - single categorical columns in `singles`
      - and composite pairs in `composites` (adds new encoded columns "A+B")
    Returns: (encoded_df, variable_value_labels)
    """
    df = df.copy()
    variable_value_labels = {}

    def _encode_one(col: str):
        nonlocal df, variable_value_labels
        if col not in df.columns:
            return
        if keep_original:
            df[f"{col}_text"] = df[col].astype(str)
        code_map, labels = build_numeric_codes(df[col])
        if code_map:
            df[col] = df[col].map(code_map).astype("float")
            variable_value_labels[col] = labels

    # Singles
    for col in singles:
        _encode_one(col)

    # Composites -> add a new column "A+B" with joined text, then encode it
    for a, b in composites:
        if a in df.columns and b in df.columns:
            new_col = f"{a}_{b}"
            df[new_col] = (
                df[[a, b]]
                .astype(str)
                .apply(lambda s: joiner.join([x.strip() for x in s if str(x).strip() != ""]), axis=1)
            )
            _encode_one(new_col)

    return df, variable_value_labels

def clean_column_names_for_sav(df):
    """
    Clean column names for SPSS SAV file compatibility while preserving column order.
    Rules:
    - Remove parentheses and join with underscore
    - Replace '/', '-', or spaces with underscore
    - Ensure names start with letter
    - Truncate to 64 characters
    """
    cleaned_df = df.copy()
    new_columns = {}
    
    # Process columns in their original order
    for col in df.columns:
        # Start with original column name
        clean_name = str(col)
        
        # Remove parentheses and join with underscore
        clean_name = clean_name.replace('(', '_').replace(')', '_')
        
        # Replace '/', '-', or spaces with underscore
        clean_name = clean_name.replace('/', '_').replace('-', '_').replace(' ', '_')
        
        # Remove multiple consecutive underscores
        import re
        clean_name = re.sub(r'_+', '_', clean_name)
        
        # Remove leading/trailing underscores
        clean_name = clean_name.strip('_')
        
        # Ensure name starts with letter (SPSS requirement)
        if clean_name and not clean_name[0].isalpha():
            clean_name = 'VAR_' + clean_name
        
        # Truncate to 64 characters (SPSS limit)
        if len(clean_name) > 64:
            clean_name = clean_name[:64]
        
        # Handle empty names
        if not clean_name:
            clean_name = 'VAR_UNNAMED'
        
        # Handle duplicates by adding numbers
        original_clean_name = clean_name
        counter = 1
        while clean_name in new_columns.values():
            clean_name = f"{original_clean_name}_{counter}"
            counter += 1
        
        new_columns[col] = clean_name
    
    # Rename columns while preserving order
    cleaned_df.columns = [new_columns[col] for col in df.columns]
    
    return cleaned_df, new_columns

def build_variable_map(df, variable_value_labels, schema_formats):
    rows = []
    for col in df.columns:
        fmt = schema_formats.get(col, "")
        if col in variable_value_labels:
            labels = variable_value_labels[col]
            for code,label in labels.items():
                rows.append({"Variable":col,"SPSS_Format":fmt,"Encoded":"Yes","Code":code,"Label":label})
        else:
            rows.append({"Variable":col,"SPSS_Format":fmt,"Encoded":"No","Code":"","Label":""})
    return pd.DataFrame(rows)

# =========================
# ===== STREAMLIT UI ======
# =========================
st.set_page_config(page_title="Excel → Multi-format Converter", layout="wide")
st.title("Excel → SPSS / Excel / CSV / TSV Converter with Filters and Wide/Long Options")

# Input section with simple browse functionality
st.subheader("📁 Input Configuration")

# Initialize session state for uploaded files
if "uploaded_molbio_files" not in st.session_state:
    st.session_state.uploaded_molbio_files = []
if "uploaded_clinician_files" not in st.session_state:
    st.session_state.uploaded_clinician_files = []
if "uploaded_order_file" not in st.session_state:
    st.session_state.uploaded_order_file = None
if "uploaded_dr_minit_order_file" not in st.session_state:
    st.session_state.uploaded_dr_minit_order_file = None

col1, col2 = st.columns([3, 1])

with col1:
    # Mol-bio files input
    molbio_path = st.text_input("Folder path for mol-bio Excel files", 
                               placeholder="e.g., /path/to/molbio/folder",
                               help="Enter the full path to folder containing mol-bio Excel files")
    
    # Clinician files input
    clinician_path = st.text_input("Folder path for clinician Excel files",
                                  placeholder="e.g., /path/to/clinician/folder", 
                                  help="Enter the full path to folder containing clinician Excel files")
    
    # Column order file input (optional)
    order_excel = st.text_input("Optional Excel path for column order (leave blank if none)",
                               placeholder="e.g., /path/to/order.xlsx",
                               help="Excel file with desired column order in first column")
    
    # Output path
    output_path = st.text_input("Output folder path for merged Excel",
                               placeholder="e.g., /path/to/output/folder",
                               help="Folder where the merged Excel file will be saved")

with col2:
    st.markdown("**Or browse files:**")
    
    # Simple browse links
    if st.button("📁 Browse mol-bio files", help="Click to upload mol-bio Excel files"):
        st.session_state.show_molbio_uploader = True
    
    if st.button("📁 Browse clinician files", help="Click to upload clinician Excel files"):
        st.session_state.show_clinician_uploader = True
        
    if st.button("📄 Browse order file", help="Click to upload column order Excel file", key="browse_order_file"):
        st.session_state.show_order_uploader = True
        
    if st.button("📂 Browse output folder", help="Click to select output folder"):
        st.session_state.show_output_selector = True

# Show file uploaders when browse buttons are clicked
if st.session_state.get("show_molbio_uploader", False):
    molbio_files = st.file_uploader("Upload mol-bio Excel files", 
                                   type=['xlsx', 'xls', 'xlsm'], 
                                   accept_multiple_files=True,
                                   key="molbio_upload")
    if molbio_files:
        st.session_state.uploaded_molbio_files = molbio_files
        st.session_state.show_molbio_uploader = False
        st.success(f"✅ {len(molbio_files)} mol-bio files selected")
        st.rerun()

if st.session_state.get("show_clinician_uploader", False):
    clinician_files = st.file_uploader("Upload clinician Excel files",
                                      type=['xlsx', 'xls', 'xlsm'],
                                      accept_multiple_files=True,
                                      key="clinician_upload")
    if clinician_files:
        st.session_state.uploaded_clinician_files = clinician_files
        st.session_state.show_clinician_uploader = False
        st.success(f"✅ {len(clinician_files)} clinician files selected")
        st.rerun()

if st.session_state.get("show_order_uploader", False):
    order_file = st.file_uploader("Upload column order Excel file",
                                 type=['xlsx', 'xls', 'xlsm'],
                                 key="order_upload")
    if order_file:
        st.session_state.uploaded_order_file = order_file
        st.session_state.show_order_uploader = False
        st.success(f"✅ Column order file selected: {order_file.name}")
        st.rerun()

if st.session_state.get("show_output_selector", False):
    output_folder_input = st.text_input(
        "Enter output folder path:",
        placeholder="/path/to/your/output/folder",
        key="output_folder_input"
    )
    if st.button("✅ Use This Path"):
        st.session_state.show_output_selector = False
        st.success("✅ You can now use the path above")
        st.rerun()

# Show selected files status
if st.session_state.uploaded_molbio_files:
    st.info(f"📁 {len(st.session_state.uploaded_molbio_files)} mol-bio files ready")
if st.session_state.uploaded_clinician_files:
    st.info(f"📁 {len(st.session_state.uploaded_clinician_files)} clinician files ready")
if st.session_state.uploaded_order_file:
    st.info(f"📄 Column order file ready: {st.session_state.uploaded_order_file.name}")
# Reset button
if st.button("🔄 Reset/Clear All", help="Clear all data and file selections"):
    for key in ["df_raw", "uploaded_molbio_files", "uploaded_clinician_files", "uploaded_order_file", 
                "show_molbio_uploader", "show_clinician_uploader", "show_order_uploader", "show_output_selector"]:
        if key in st.session_state:
            del st.session_state[key]
    st.success("✅ All data cleared!")
    st.rerun()

# ── v NEW: initialize session storage once
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "merge_in_progress" not in st.session_state:
    st.session_state.merge_in_progress = False
# ─────────────────────────────────────────

# Helper function to save uploaded files temporarily
def save_uploaded_files(uploaded_files, prefix="temp"):
    """Save uploaded files to temporary directory and return the directory path"""
    if not uploaded_files:
        return None
    
    temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    return temp_dir

# Check if paths are provided (Stage 1 validation)
def check_paths_provided():
    use_molbio_upload = st.session_state.uploaded_molbio_files and len(st.session_state.uploaded_molbio_files) > 0
    use_clinician_upload = st.session_state.uploaded_clinician_files and len(st.session_state.uploaded_clinician_files) > 0
    
    molbio_ready = use_molbio_upload or molbio_path.strip()
    clinician_ready = use_clinician_upload or clinician_path.strip()
    output_ready = output_path.strip()
    
    return molbio_ready and clinician_ready and output_ready

paths_provided = check_paths_provided()
data_merged = st.session_state.df_raw is not None

st.divider()

# Merge button with conditional enabling
merge_button_disabled = not paths_provided or st.session_state.merge_in_progress

if st.button("🚀 Run Merge", type="primary", disabled=merge_button_disabled, 
             help="Provide all required paths first" if not paths_provided else "Click to merge data"):
    
    st.session_state.merge_in_progress = True
    
    # Determine input sources
    use_molbio_upload = st.session_state.uploaded_molbio_files and len(st.session_state.uploaded_molbio_files) > 0
    use_clinician_upload = st.session_state.uploaded_clinician_files and len(st.session_state.uploaded_clinician_files) > 0
    use_order_upload = st.session_state.uploaded_order_file is not None
    
    temp_dirs_to_cleanup = []
    try:
        with st.spinner("🔄 Merging mol-bio and clinician Excel files..."):
            # Prepare paths - use uploads if available, otherwise use text inputs
            final_molbio_path = molbio_path.strip()
            final_clinician_path = clinician_path.strip()
            final_order_path = order_excel.strip() if order_excel.strip() else None
            final_output_path = output_path.strip()
            
            # Handle uploaded files
            if use_molbio_upload:
                temp_molbio_dir = save_uploaded_files(st.session_state.uploaded_molbio_files, "molbio")
                final_molbio_path = temp_molbio_dir
                temp_dirs_to_cleanup.append(temp_molbio_dir)
                st.info(f"Using {len(st.session_state.uploaded_molbio_files)} uploaded mol-bio files")
            
            if use_clinician_upload:
                temp_clinician_dir = save_uploaded_files(st.session_state.uploaded_clinician_files, "clinician")
                final_clinician_path = temp_clinician_dir
                temp_dirs_to_cleanup.append(temp_clinician_dir)
                st.info(f"Using {len(st.session_state.uploaded_clinician_files)} uploaded clinician files")
            
            if use_order_upload:
                temp_order_dir = save_uploaded_files([st.session_state.uploaded_order_file], "order")
                final_order_path = os.path.join(temp_order_dir, st.session_state.uploaded_order_file.name)
                temp_dirs_to_cleanup.append(temp_order_dir)
                st.info(f"Using uploaded column order file: {st.session_state.uploaded_order_file.name}")
            
            # Create output directory if it doesn't exist
            os.makedirs(output_path, exist_ok=True)
            
            # Call the merge function
            df_raw = merge_sb_format(final_molbio_path, final_clinician_path, final_order_path, final_output_path)
            
            # Store in session
            st.session_state.df_raw = df_raw
            st.session_state.merge_in_progress = False
            
        st.success(f"✅ Merge complete! Processed {len(df_raw):,} rows with {df_raw.shape[1]} columns.")
        st.rerun()  # Refresh to activate the filtering options
        
    except Exception as e:
        st.session_state.merge_in_progress = False
        st.error(f"❌ Error during merge: {str(e)}")
        st.error("Please check your input files and try again.")
        # Optionally show the full traceback for debugging
        with st.expander("Show detailed error"):
            st.exception(e)
    
    finally:
        # Cleanup temporary directories
        for temp_dir in temp_dirs_to_cleanup:
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                st.warning(f"Could not cleanup temporary directory {temp_dir}: {e}")

# Show loading state
if st.session_state.merge_in_progress:
    st.info("🔄 Processing your data... Please wait.")
    st.stop()

# =========================
# ALL OPTIONS SHOWN UPFRONT (with conditional enabling)
# =========================

st.divider()

# Determine if data is available for processing
data_available = st.session_state.df_raw is not None

# Prepare data if available
if data_available:
    df_raw = st.session_state.df_raw.copy()
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    
    if MRN_COL not in df_raw.columns:
        st.error(f"Required unique patient column '{MRN_COL}' not found.")
        st.stop()
    
    filter_cols_present = [c for c in FILTER_COLS if c in df_raw.columns]
    dep_map = {k:[d for d in v if d in df_raw.columns] for k,v in DEP_FILTER_COLS.items() if k in df_raw.columns}
else:
    df_raw = None
    filter_cols_present = FILTER_COLS  # Show placeholder names
    dep_map = DEP_FILTER_COLS

# --- Step 1: Filters ---
st.subheader("🔍 1) Data Filters")

if not data_available:
    st.info("💡 Complete the merge above to activate filtering options")

filter_mode = st.radio("Filter mode", ["Row-wise","Patient-wise"], horizontal=True, 
                      disabled=not data_available,
                      help="Row-wise: filter individual rows | Patient-wise: include all rows for matching patients")

filter_selections = {}
cols_left, cols_right = st.columns(2)

with cols_left:
    st.markdown("**Independent Filters:**")
    for col in filter_cols_present:
        if any(col in deps for deps in dep_map.values()): 
            continue
        
        if data_available and col in df_raw.columns:
            # Special handling for Age - show bins instead of individual values
            if col == "Age":
                # Define age bin options in order
                options = ["<10", "10-20", "20-30", "30-40", "40-50", "50-60", 
                          "60-70", "70-80", "80-90", "90+"]
            else:
                options = sorted(df_raw[col].dropna().astype(str).unique().tolist())
        else:
            options = []
            
        sel = st.multiselect(
            f"{col}", 
            options, 
            default=[], 
            key=f"ind_{col}",
            disabled=not data_available,
            help=f"Filter by {col} values" if data_available else "Complete merge to activate"
        )
        filter_selections[col] = sel
        
        # Show gene selection status
        if col == "Gene_name" and data_available and sel:
            st.info(f"🎯 {len(sel)} gene(s) selected: {', '.join(sel)}")

with cols_right:
    st.markdown("**Dependent Filters:**")
    
    # Handle gene-dependent filters dynamically
    if data_available and "Gene_name" in df_raw.columns:
        selected_genes = filter_selections.get("Gene_name", [])
        
        if len(selected_genes) == 0:
            # No genes selected - show disabled dropdowns
            for dep in ["Mutation(nucleotide)", "Mutation(protein)", "Exon_no"]:
                if dep in df_raw.columns:
                    st.multiselect(
                        f"{dep} (depends on Gene_name)", 
                        [], 
                        default=[], 
                        key=f"dep_{dep}",
                        disabled=True,
                        help="Select genes first to activate"
                    )
                    filter_selections[dep] = []
        
        elif len(selected_genes) == 1:
            # Single gene selected - show filtered options
            gene = selected_genes[0]
            gene_data = df_raw[df_raw["Gene_name"] == gene]
            
            for dep in ["Mutation(nucleotide)", "Mutation(protein)", "Exon_no"]:
                if dep in df_raw.columns:
                    # Filter options to only show values for the selected gene
                    options = sorted(gene_data[dep].dropna().astype(str).unique().tolist())
                    sel = st.multiselect(
                        f"{dep} ({gene})", 
                        options, 
                        default=[], 
                        key=f"dep_{dep}",
                        disabled=False,
                        help=f"Filter by {dep} values for {gene}"
                    )
                    filter_selections[dep] = sel
        
        else:
            # Multiple genes selected - create dynamic columns
            st.markdown(f"**Multi-gene filters ({len(selected_genes)} genes selected):**")
            
            # Create columns dynamically based on number of selected genes
            gene_cols = st.columns(len(selected_genes))
            
            for i, gene in enumerate(selected_genes):
                with gene_cols[i]:
                    st.markdown(f"**{gene}**")
                    gene_data = df_raw[df_raw["Gene_name"] == gene]
                    
                    for dep in ["Mutation(nucleotide)", "Mutation(protein)", "Exon_no"]:
                        if dep in df_raw.columns:
                            # Filter options to only show values for this specific gene
                            options = sorted(gene_data[dep].dropna().astype(str).unique().tolist())
                            sel = st.multiselect(
                                f"{dep}", 
                                options, 
                                default=[], 
                                key=f"dep_{dep}_{gene}",
                                disabled=False,
                                help=f"Filter by {dep} values for {gene}"
                            )
                            # Store selections with gene prefix
                            filter_selections[f"{dep}_{gene}"] = sel
    else:
        # Fallback for when data is not available
        for indep, deps in dep_map.items():
            active = filter_selections.get(indep, []) if data_available else []
            for dep in deps:
                if data_available and dep in df_raw.columns:
                    options = sorted(df_raw[dep].dropna().astype(str).unique().tolist())
                else:
                    options = []
                    
                is_enabled = data_available and len(active) > 0
                sel = st.multiselect(
                    f"{dep} (depends on {indep})", 
                    options, 
                    default=[], 
                    key=f"dep_{dep}",
                    disabled=not is_enabled,
                    help=f"Select {indep} first to activate" if not is_enabled else f"Filter by {dep} values"
                )
                filter_selections[dep] = sel if is_enabled else []

# Apply filters if data is available
if data_available:
    df_filtered = filter_dataframe(df_raw, filter_selections, filter_mode)
else:
    df_filtered = None

# --- Step 2: Output Shape ---
st.subheader("📊 2) Output Format")

if not data_available:
    st.info("💡 Complete the merge above to activate format options")

shape_mode = st.radio(
    "Choose output format", 
    ["Long format","Dr. Minit's format","Gene-wide"], 
    horizontal=True,
    disabled=not data_available,
    help="Long: keep all rows | Dr. Minit's format: wide format for Dr. Minit | Gene-wide: genes as columns"
)

# Process data shape if available
if data_available and df_filtered is not None:
    if shape_mode=="Long format":
        final_df = df_filtered.copy()
    elif shape_mode=="Dr. Minit's format":
        # Use the Dr. Minit format conversion
        final_df = convert_long_to_dr_minit_format(df_filtered, None, None)
        # Ensure no duplicate columns
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    elif shape_mode=="Gene-wide":
        wide_df, long_df = to_gene_wide_format(df_filtered, MRN_COL, "Gene_name", GENE_COLUMNS)
        if not long_df.empty:
            st.warning("⚠️ Some genes had multiple rows → kept in long format.")
            final_df = pd.concat([wide_df, long_df], ignore_index=True, sort=False)
        else:
            final_df = wide_df
else:
    final_df = None

# --- Step 3: Column Removal (only for Dr. Minit's format) ---
if shape_mode == "Dr. Minit's format":
    st.subheader("❌ 3) Remove Columns")
    
    if not data_available:
        st.info("💡 Complete the merge above to activate column removal")
    
    # Order file input for Dr. Minit's format
    col1, col2 = st.columns([3, 1])
    
    with col1:
        dr_minit_order_path = st.text_input("Optional Excel path for column order (leave blank if none)",
                                           placeholder="e.g., /path/to/order.xlsx",
                                           help="Excel file with desired column order in first column",
                                           key="dr_minit_order_path")
    
    with col2:
        st.markdown("**Or browse file:**")
        if st.button("📄 Browse order file", help="Click to upload column order Excel file", key="browse_dr_minit_order_file"):
            st.session_state.show_dr_minit_order_uploader = True
    
    # Show file uploader when browse button is clicked
    if st.session_state.get("show_dr_minit_order_uploader", False):
        dr_minit_order_file = st.file_uploader("Upload column order Excel file for Dr. Minit's format",
                                             type=['xlsx', 'xls', 'xlsm'],
                                             key="dr_minit_order_upload")
        if dr_minit_order_file:
            st.session_state.uploaded_dr_minit_order_file = dr_minit_order_file
            st.session_state.show_dr_minit_order_uploader = False
            st.success(f"✅ Column order file selected: {dr_minit_order_file.name}")
            st.rerun()
    
    # Show selected file status
    if st.session_state.uploaded_dr_minit_order_file:
        st.info(f"📄 Column order file ready: {st.session_state.uploaded_dr_minit_order_file.name}")
    
    if data_available and final_df is not None:
        available_columns = list(final_df.columns)
    else:
        available_columns = []
    
    remove_cols = st.multiselect(
        "Select columns to remove", 
        options=available_columns, 
        default=[],
        disabled=not data_available,
        help="Choose columns to exclude from final output"
    )
    
    if data_available and final_df is not None:
        final_df = final_df.drop(columns=remove_cols, errors="ignore")
        
        # Re-process with order file if provided
        use_dr_minit_order_upload = st.session_state.uploaded_dr_minit_order_file is not None
        
        if st.session_state.get("dr_minit_order_path", "").strip() or use_dr_minit_order_upload:
            # Save uploaded file temporarily if needed
            temp_order_path = None
            if use_dr_minit_order_upload:
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="dr_minit_order_")
                temp_order_path = os.path.join(temp_dir, st.session_state.uploaded_dr_minit_order_file.name)
                with open(temp_order_path, "wb") as f:
                    f.write(st.session_state.uploaded_dr_minit_order_file.getbuffer())
            
            final_order_path = st.session_state.get("dr_minit_order_path", "").strip() if st.session_state.get("dr_minit_order_path", "").strip() else temp_order_path
            
            # Re-convert with order file
            final_df = convert_long_to_dr_minit_format(df_filtered, final_order_path, None)
            
            # Cleanup temp file
            if temp_order_path and os.path.exists(temp_order_path):
                try:
                    os.remove(temp_order_path)
                    os.rmdir(os.path.dirname(temp_order_path))
                except:
                    pass

# --- Preview ---
st.subheader("👁️ Preview")

if data_available and final_df is not None:
    st.dataframe(final_df.head(50), use_container_width=True)
    st.caption(f"Rows: {len(final_df):,} | Cols: {final_df.shape[1]}")
else:
    st.info("💡 Complete the merge above to see data preview")
    # Show placeholder
    placeholder_data = {"Column 1": ["Sample data will appear here..."], 
                       "Column 2": ["...after merge is complete"]}
    st.dataframe(pd.DataFrame(placeholder_data), use_container_width=True)

# --- Step 4: Export Options ---
st.subheader("💾 4) Export Formats")

if not data_available:
    st.info("💡 Complete the merge above to activate export options")

# File naming section
col1, col2 = st.columns([2, 1])
with col1:
    output_filename = st.text_input("Output file name (without extension)", 
                                   value="output", 
                                   disabled=not data_available,
                                   help="Enter the base name for output files (extensions will be added automatically)")
with col2:
    st.markdown("**Example:**")
    st.caption("`output` → output.xlsx, output.sav, output.csv")

export_sav = st.checkbox("SPSS (.sav)", disabled=not data_available, help="Statistical analysis format")
export_xlsx = st.checkbox("Excel (.xlsx)", value=True, disabled=not data_available, help="Spreadsheet format")
export_csv = st.checkbox("CSV (.csv)", disabled=not data_available, help="Comma-separated values")
export_tsv = st.checkbox("TSV (.tsv)", disabled=not data_available, help="Tab-separated values")

export_button_disabled = not data_available or not any([export_sav, export_xlsx, export_csv, export_tsv])

if st.button("📥 Export Selected Formats", 
             disabled=export_button_disabled,
             help="Select at least one format and complete merge first" if export_button_disabled else "Save your processed data"):
    
    # Ensure output path exists
    final_output_path = output_path.strip() if output_path.strip() else "."
    os.makedirs(final_output_path, exist_ok=True)
    
    # Get base filename
    base_filename = output_filename.strip() if output_filename.strip() else "output"
    
    # Get dynamic columns to encode
    dynamic_encode_columns = get_columns_to_encode(final_df.columns)
    
    # Handle Dr. Minit's format differently
    if shape_mode == "Dr. Minit's format":
        # For Dr. Minit's format, only export the wide format (not long format)
        if export_xlsx:
            # Save Dr. Minit format to output path
            # Determine order file path
            use_dr_minit_order_upload = st.session_state.uploaded_dr_minit_order_file is not None
            temp_order_path = None
            
            if use_dr_minit_order_upload:
                import tempfile
                temp_dir = tempfile.mkdtemp(prefix="dr_minit_order_")
                temp_order_path = os.path.join(temp_dir, st.session_state.uploaded_dr_minit_order_file.name)
                with open(temp_order_path, "wb") as f:
                    f.write(st.session_state.uploaded_dr_minit_order_file.getbuffer())
            
            # Get order path from UI input
            final_order_path = st.session_state.get("dr_minit_order_path", "").strip() if st.session_state.get("dr_minit_order_path", "").strip() else temp_order_path
            
            # Convert and save Dr. Minit format
            dr_minit_df = convert_long_to_dr_minit_format(df_filtered, final_order_path, None)
            
            # Save with user-specified filename
            xlsx_path = os.path.join(final_output_path, f"{base_filename}.xlsx")
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                # Write to separate sheets (Tissue and Liquid)
                sheets = {"Tissue": [], "Liquid": []}
                
                # Group by MRN and Date-Case_discussed to recreate the sheet structure
                grouped = df_filtered.groupby(["MRN", "Date-Case_discussed"], sort=False, dropna=False)
                for _, group in grouped:
                    group = group.copy()
                    target_sheets = set()
                    for _, row in group.iterrows():
                        target_sheets.update(get_target_sheets_from_specimen_vaf(row))
                    
                    for specimen_block in target_sheets:
                        # Process each group for each sheet
                        processed = process_case_group(group, dr_minit_df.columns.tolist(), specimen_block)
                        if any(row for row in processed if any(val not in ["", "NA", "nan"] for val in row.values())):
                            sheets[specimen_block].extend(processed)
                
                # Write each sheet
                for sheet_name, rows in sheets.items():
                    if rows:
                        df_sheet = pd.DataFrame(rows)
                        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        # Create empty sheet
                        empty_df = pd.DataFrame(columns=dr_minit_df.columns)
                        empty_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            st.success(f"✅ Dr. Minit's format Excel saved: {xlsx_path}")
            
            # Cleanup temp file
            if temp_order_path and os.path.exists(temp_order_path):
                try:
                    os.remove(temp_order_path)
                    os.rmdir(os.path.dirname(temp_order_path))
                except:
                    pass
        
        # For other formats, use the processed final_df
        if export_sav or export_csv or export_tsv:
            # (A) Build SCHEMA from the actual columns you're exporting
            SCHEMA = build_schema_from_columns(final_df.columns)
            # (B) Encode singles + composites with dynamic columns
            encoded_df, value_labels = encode_with_singles_and_composites(
                apply_schema_types(final_df, SCHEMA),
                singles=dynamic_encode_columns,
                composites=ENCODE_COMPOSITES,
                keep_original=False  # Don't create _text columns
            )

            # variable formats, inherited for derived numbered columns (if any)
            var_format = inherit_schema_for_derived_columns(encoded_df, SCHEMA)
            
            if export_sav:
                # Clean column names for SAV compatibility
                cleaned_df, column_mapping = clean_column_names_for_sav(encoded_df)
                
                # Update value labels and variable format with cleaned column names
                cleaned_value_labels = {}
                cleaned_var_format = {}
                
                for old_col, new_col in column_mapping.items():
                    if old_col in value_labels:
                        cleaned_value_labels[new_col] = value_labels[old_col]
                    if old_col in var_format:
                        cleaned_var_format[new_col] = var_format[old_col]
                
                # Save SAV file
                sav_path = os.path.join(final_output_path, f"{base_filename}.sav")
                pyreadstat.write_sav(cleaned_df, sav_path, variable_value_labels=cleaned_value_labels, variable_format=cleaned_var_format)
                st.success(f"✅ SAV file saved: {sav_path}")
                
            if export_csv:
                csv_path = os.path.join(final_output_path, f"{base_filename}.csv")
                encoded_df.to_csv(csv_path, index=False)
                st.success(f"✅ CSV file saved: {csv_path}")
                
            if export_tsv:
                tsv_path = os.path.join(final_output_path, f"{base_filename}.tsv")
                encoded_df.to_csv(tsv_path, sep="\t", index=False)
                st.success(f"✅ TSV file saved: {tsv_path}")
    
    else:
        # Original logic for Long format and Gene-wide
        # (A) Build SCHEMA from the actual columns you're exporting
        SCHEMA = build_schema_from_columns(final_df.columns)
        # (B) Encode singles + composites with dynamic columns
        encoded_df, value_labels = encode_with_singles_and_composites(
            apply_schema_types(final_df, SCHEMA),
            singles=dynamic_encode_columns,
            composites=ENCODE_COMPOSITES,
            keep_original=False  # Don't create _text columns
        )

        # variable formats, inherited for derived numbered columns (if any)
        var_format = inherit_schema_for_derived_columns(encoded_df, SCHEMA)
        
        if export_sav:
            # Clean column names for SAV compatibility
            cleaned_df, column_mapping = clean_column_names_for_sav(encoded_df)
            
            # Update value labels and variable format with cleaned column names
            cleaned_value_labels = {}
            cleaned_var_format = {}
            
            for old_col, new_col in column_mapping.items():
                if old_col in value_labels:
                    cleaned_value_labels[new_col] = value_labels[old_col]
                if old_col in var_format:
                    cleaned_var_format[new_col] = var_format[old_col]
            
            # Save SAV file
            sav_path = os.path.join(final_output_path, f"{base_filename}.sav")
            pyreadstat.write_sav(cleaned_df, sav_path, variable_value_labels=cleaned_value_labels, variable_format=cleaned_var_format)
            st.success(f"✅ SAV file saved: {sav_path}")
            
        if export_xlsx:
            xlsx_path = os.path.join(final_output_path, f"{base_filename}.xlsx")
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                encoded_df.to_excel(writer, sheet_name="Data", index=False)
                build_variable_map(encoded_df, value_labels, var_format).to_excel(writer, sheet_name="Variable_Map", index=False)
            st.success(f"✅ Excel file saved: {xlsx_path}")
            
        if export_csv:
            csv_path = os.path.join(final_output_path, f"{base_filename}.csv")
            encoded_df.to_csv(csv_path, index=False)
            st.success(f"✅ CSV file saved: {csv_path}")
            
        if export_tsv:
            tsv_path = os.path.join(final_output_path, f"{base_filename}.tsv")
            encoded_df.to_csv(tsv_path, sep="\t", index=False)
            st.success(f"✅ TSV file saved: {tsv_path}")
    
    st.success("🎉 All selected formats exported successfully!")
# =========================
# ======= NOTES ===========
# =========================
st.markdown(
    """
**Notes / How it works**
- **Dependent filters**: any filter in `DEP_FILTER_COLS[indep]` is disabled until the *independent* has a selection.
- **Patient-wise filtering**: keeps *all rows* for MRNs where at least one row matches all active filters.
- **Flattened (wide) output**:
  - `WIDE_COLS`: expands each listed column independently to `<col>_1..K` per MRN.
  - `WIDE_PAIR_COLS`: expands each group together with a shared width `K` (max over the group for that MRN), emitting `<col>_1..K` for each col in the group.
- **Encoding**:
  - Every column in `ENCODE_COLUMNS` gets numeric codes + value labels.
  - Any derived numbered columns (e.g., `Gene_1`) also get encoded automatically.
  - If `KEEP_ORIGINAL_TEXT_FOR_ENCODED=True`, the original values are kept in `<col>_text`.
- **Schema**:
  - SPSS formats are inherited for derived columns (e.g., `Gene` → `Gene_1`..`Gene_k`).
  - Use `'A{width}'` for strings, `'F8.0'` for integers, `'F10.2'` for floats, etc.
"""
)
