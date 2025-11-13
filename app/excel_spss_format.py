import os
import pandas as pd
import re
from difflib import get_close_matches

# Load additional MRN-specific metadata
# Load and deduplicate MRN metadata

matched_genes_v = ['erbb2', 'braf', 'kras', 'met', 'egfr', 'ros1', 'alk', 'ret', 'msi', 'ntrk']

variant_types_allowed = [
    "Indel", "SNV", "Duplication", "Amplification", "Missense", "Frameshift", "Insertion", "Truncation", "Deletion", "LOH",
    "delins", "Exon skipping", "Splice donor SNV", "Splice site deletion", "InFrame insertion", "Nonframeshift block substitution",
    "InFrame deletion", "Nonframeshift deletion", "Copy number loss", "Nonframeshift Mutation", "Nonframeshift Insertion",
    "Splice Variant", "Stop gain", "Nonsense", "InFrame Indel", "Splice acceptor SNV", "Frameshift deletion", "Inframe delins", "MNV"
]

# ====== Stable output schema (used when no SPSS template is present) ======
MATCHED_GENES = ['EGFR','KRAS','BRAF','ERBB2','MET','ALK','ROS1','RET','MSI','NTRK']  # order matters

def _gene_mut_fields(g):
    b = g.upper()
    return [f"{b}_Mutation", f"{b}_Mutation_AA_Change", f"{b}_Mutation_Exon",
            f"{b}_Mutation_Type", f"{b}_Mutation_Tier", f"{b}_Mutation_VAF",
            f"{b}_Mutation_Yes_No"]

def _gene_fusion_fields(g):
    b = g.upper()
    return [f"{b}_Fusion_Yes_No", f"{b}_Fusion_Partner", f"{b}_Fusion_NM_ID", f"{b}_Fusion_Read_Count"]

def _gene_amp_fields(g):
    b = g.upper()
    return [f"{b}_Amplification_Yes_No", f"{b}_Amplification_Copy_Number"]

EGFR_SPECIAL_FIELDS = [
    "EGFR_Mutation_1","EGFR_AA_Change_1","EGFR_Mutation_Type_1","EGFR_VAF_1","EGFR_Tier_1",
    "EGFR_Mutation_2","EGFR_AA_Change_2","EGFR_Mutation_Type_2","EGFR_VAF_2","EGFR_Tier_2",
    "EGFR_Mutation_3","EGFR_AA_Change_3","EGFR_Mutation_Type_3","EGFR_VAF_3","EGFR_Tier_3",
    "EGFR_Exon19Del", "EGFR_Exon21_L858R"
]

OTHER_MUT_ALL_FIELDS = []
for i in range(1,5):
    OTHER_MUT_ALL_FIELDS += [
        f"Other_Mutation_{i}", f"Other_Mutation_{i}_AA_Change", f"Other_Mutation_{i}_Exon",
        f"Other_Mutation_{i}_Type", f"Other_Mutation_{i}_Tier", f"Other_Mutation_{i}_VAF",
        f"Other_Mutation_{i}_Yes_No"
    ]

BIOMARKER_FIELDS = [
    "TMB_High_Low","MSI_High_Statble_Low","LOH_Score","HRD_Score",
    "Telomeric_Allelic_Imbalance_Yes_No","Large_Scale_State_Transitions",
    "Genes_Tier4_Name","Genes_Tier4_Alterations_Yes_No"
]

MOLBIO_COLUMNS = (
    EGFR_SPECIAL_FIELDS
    + sum([_gene_mut_fields(g) for g in MATCHED_GENES if g != "EGFR"], [])
    + sum([_gene_fusion_fields(g) for g in MATCHED_GENES], [])
    + sum([_gene_amp_fields(g) for g in MATCHED_GENES], [])
    + OTHER_MUT_ALL_FIELDS
    + BIOMARKER_FIELDS
)
# ========================================================================

def first_val(df_group, col, default="NA"):
    """Return the first non-empty (post-ffill) value for the column within the group."""
    if col not in df_group.columns:
        return default
    # df_group is already ffilled; .iloc[0] is fine, but keep a safe check:
    v = str(df_group[col].iloc[0]) if len(df_group) else ""
    v = v.strip()
    return default if v.lower() in ("", "na", "nan") else v

def fuzzy_match_variant(vtype):
    matches = get_close_matches(str(vtype), variant_types_allowed, n=1, cutoff=0.6)
    return matches[0] if matches else vtype

def load_header_order(path: str) -> list[str]:
    """
    Load a header order from a CSV or Excel file (first sheet).
    Returns a list of column names in desired order.
    """
    low = path.lower()
    if low.endswith(".csv"):
        df0 = pd.read_csv(path, nrows=0)
    elif low.endswith((".xlsx", ".xls")):
        df0 = pd.read_excel(path, nrows=0)
    else:
        raise ValueError(f"Unsupported column-order file type: {path}")
    return df0.columns.tolist()

def filter_block_for_specimen_type(block, specimen_type):
    def is_valid(v):
        try:
            s = str(v).strip().lower()
            return s not in ["", "na", "nan"]
        except:
            return False

    raw_specimen = str(block.iloc[0].get("Specimen", "")).strip().lower()

    # Decide which site's VAF validates presence on this page
    def vaf_col_for(block_specimen, page):
        if page == "Tissue":
            if block_specimen in ["pf", "pleural", "pleural fluid"]:
                return "VAF (pf)"
            if block_specimen == "csf":
                return "VAF (csf)"
            if block_specimen in ["plasma", "ctdna", "liquid", "blood"]:
                return "VAF (liq)"
            return "VAF (tissue)"  # default
        if page == "Liquid":
            if "tissue and plasma" in block_specimen:
                return "VAF (liq)"
            if "pf and tissue" in block_specimen:
                return "VAF (pf)"
            if "csf and tissue" in block_specimen:
                return "VAF (csf)"
            if "plasma and pleural" in block_specimen:
                return "VAF (liq)"
        return None

    col = vaf_col_for(raw_specimen, specimen_type)
    if not col:
        return block.iloc[0:0]  # empty

    return block[block[col].apply(is_valid)].copy()

def is_specimen_allowed_for_page(specimen, page):
    specimen = str(specimen).strip().lower()
    if page == "Tissue":
        return specimen in [
            "tissue", "ffpe", "paraffin",
            "tissue and plasma",
            "pf", "pleural", "pleural fluid",
            "csf",
            "pf and tissue",
            "csf and tissue",
            "plasma and pleural",
            "plasma", "ctdna", "liquid", "blood"
        ]
    elif page == "Liquid":
        return specimen in [
            "tissue and plasma",
            "pf and tissue",
            "csf and tissue",
            "plasma and pleural"
        ]
    return False

def clean_mutation(m):
    if pd.isna(m): return "NA"
    m = str(m).strip()
    m = re.sub(r'^[cp]\.?', '', m)
    m = re.sub(r'[(){}\[\]]', '', m)
    return m

def normalize_g3_value(val):
    val = str(val).strip().lower()
    if val in ["", "na", "nan"]:
        return ""
    val = val.replace(",", "").replace(".0", "")
    return val

def mutation_block_from_group(df_group, spss_row, specimen_type):
    other_mut_counter = 1
    other_amp_counter = 1
    egfr_counter = 1

    for _, row in df_group.iterrows():
        gene = str(row.get("Gene_name", "")).strip()
        gene_lower = gene.lower()
        mut_type = str(row.get("Variant Type", "")).lower()
        block_specimen = str(df_group.iloc[0].get("Specimen", "")).strip().lower()

        def site_for_page(block_specimen, page):
            if page == "Tissue":
                if block_specimen in ["pf", "pleural", "pleural fluid"]:
                    return "pf"
                if block_specimen == "csf":
                    return "csf"
                if block_specimen in ["plasma", "ctdna", "liquid", "blood"]:
                    return "liq"
                return "tissue"
            if page == "Liquid":
                if "tissue and plasma" in block_specimen:
                    return "liq"
                if "pf and tissue" in block_specimen:
                    return "pf"
                if "csf and tissue" in block_specimen:
                    return "csf"
                if "plasma and pleural" in block_specimen:
                    return "liq"
            return None

        site = site_for_page(block_specimen, specimen_type)
        if site is None:
            # default to tissue to avoid crashes, but row will likely be filtered earlier
            site = "tissue"

        vaf_col = f"VAF ({site})"
        amp_col = f"Amp_read/copy number ({ 'tissue' if site=='tissue' else site })"  # matches your headers


        vaf_val = str(row.get(vaf_col, "NA"))
        norm_nuc = clean_mutation(row.get("Mutation(nucleotide)", ""))
        norm_aa = clean_mutation(row.get("Mutation(protein)", ""))

        fusion_present = str(row.get("Fusion", "")).strip().lower() not in ["", "na", "nan"]
        amp_present = str(row.get("Amplification", "")).strip().lower() == "yes"

        if gene_lower in matched_genes_v:
            spss_row[f"{gene.upper()}_Mutation_Yes_No"] = "Yes"

            if not fusion_present and not amp_present and gene_lower != "egfr":
                base = gene.upper()
                spss_row[f"{base}_Mutation"] = norm_nuc
                spss_row[f"{base}_Mutation_AA_Change"] = norm_aa
                spss_row[f"{base}_Mutation_Exon"] = row.get("Exon_no", "NA")
                spss_row[f"{base}_Mutation_Type"] = fuzzy_match_variant(mut_type)
                spss_row[f"{base}_Mutation_Tier"] = row.get("Tier", "NA")
                spss_row[f"{base}_Mutation_VAF"] = vaf_val


        # ---- FUSION detection & partner selection ----
        fusion_raw = str(row.get("Fusion", "")).strip()
        fusion_nm  = str(row.get("NM ID", "")).strip()

        gene_clean = str(gene).strip()
        gene_valid = gene_clean.lower() not in ["", "na", "nan"]

        # Try to parse "GENEA - GENEB" first; else treat 'Fusion' as just the partner
        fusion_a, fusion_b = None, None
        if " - " in fusion_raw:
            try:
                a, b = [p.strip() for p in fusion_raw.split("-", 1)]
                fusion_a, fusion_b = a, b
            except Exception:
                pass

        # Decide the two genes to consider
        if fusion_a and fusion_b:
            cand_genes = [gene_clean.lower(), fusion_a.lower(), fusion_b.lower()]
        else:
            # legacy rows where Fusion holds only the partner
            partner_only = fusion_raw
            cand_genes = [gene_clean.lower(), partner_only.lower()]

        # Which of these cand genes is a priority gene?
        hit = None
        for cg in cand_genes:
            if cg in matched_genes_v:
                hit = cg
                break

        if hit and gene_valid:
            g = hit.upper()

            # Choose partner: the "other side" of the fusion w.r.t. the current row's gene
            if fusion_a and fusion_b:
                if gene_clean.lower() == fusion_a.lower():
                    partner = fusion_b
                elif gene_clean.lower() == fusion_b.lower():
                    partner = fusion_a
                else:
                    # current gene isn't either side (rare) — keep the raw string
                    partner = fusion_raw
            else:
                partner = fusion_raw if partner_only == "" else partner_only

            # per-site Fusion Read Count (preferred) with legacy fallback
            frc_col = f"Fusion_Read_Count ({site})"
            fusion_read_site = row.get(frc_col, None)
            if fusion_read_site is None or str(fusion_read_site).strip().lower() in ["", "na", "nan"]:
                fusion_read_site = row.get("Other fusion_Read Count", "NA")

            spss_row[f"{g}_Fusion_Yes_No"]     = "Yes"
            spss_row[f"{g}_Fusion_Partner"]    = partner
            spss_row[f"{g}_Fusion_NM_ID"]      = fusion_nm
            spss_row[f"{g}_Fusion_Read_Count"] = "NA" if str(fusion_read_site).strip() == "" else str(fusion_read_site).strip()


        elif str(row.get("Amplification", "")).strip().lower() == "yes":
            #amp_col = "Amp_read/copy number (TISSUE)" if specimen_type == "Tissue" else "Amp_read/copy number (liquid)"
            amp_val = row.get(amp_col, "NA")
            if gene_lower in matched_genes_v:
                spss_row[f"{gene.upper()}_Amplification_Yes_No"] = "Yes"
                spss_row[f"{gene.upper()}_Amplification_Copy_Number"] = amp_val
            elif other_amp_counter <= 4:
                spss_row[f"Other_Amplification_{other_amp_counter}_Yes_No"] = "Yes"
                spss_row[f"Other_Amplification_{other_amp_counter}_Name"] = gene
                spss_row[f"Other_Amplification_{other_amp_counter}_Copy_Number"] = amp_val
                other_amp_counter += 1

        elif gene_lower == "egfr":
            if egfr_counter <= 3:
                spss_row[f"EGFR_Mutation_{egfr_counter}"] = norm_nuc
                spss_row[f"EGFR_Mutation_Type_{egfr_counter}"] = fuzzy_match_variant(mut_type)
                spss_row[f"EGFR_AA_Change_{egfr_counter}"] = norm_aa
                spss_row[f"EGFR_VAF_{egfr_counter}"] = vaf_val
                spss_row[f"EGFR_Tier_{egfr_counter}"] = row.get("Tier", "NA")
                egfr_counter += 1
            else:
                spss_row["Other_Rare_EGFR_Mutations"] = norm_nuc

        elif gene_lower not in matched_genes_v and other_mut_counter <= 4:
            mut_label = f"{gene} - {norm_nuc}"
            aa_label = f"{gene} - {norm_aa}"
            exon_label = f"{gene} - {row.get('Exon_no', 'NA')}"
            spss_row[f"Other_Mutation_{other_mut_counter}"] = mut_label
            spss_row[f"Other_Mutation_{other_mut_counter}_AA_Change"] = aa_label
            spss_row[f"Other_Mutation_{other_mut_counter}_Exon"] = exon_label
            spss_row[f"Other_Mutation_{other_mut_counter}_Tier"] = row.get("Tier", "NA")
            spss_row[f"Other_Mutation_{other_mut_counter}_Type"] = fuzzy_match_variant(mut_type)
            spss_row[f"Other_Mutation_{other_mut_counter}_VAF"] = vaf_val
            spss_row[f"Other_Mutation_{other_mut_counter}_Yes_No"] = "Yes"
            other_mut_counter += 1

        # EGFR special mutation logic
    egfr_rows = df_group[df_group["Gene_name"].astype(str).str.upper() == "EGFR"]

    spss_row["EGFR_Exon19Del"] = "No"
    spss_row["EGFR_Exon21_L858R"] = "No"

    for _, row in egfr_rows.iterrows():
        exon = str(row.get("Exon_no", "")).strip()
        aa = str(row.get("Mutation(protein)", "")).strip().lower()
        vtype = str(row.get("Variant Type", "")).strip().lower()

        if exon == "19" and "deletion" in vtype:
            spss_row["EGFR_Exon19Del"] = "Yes"
        if exon == "21" and aa in ["l858r", "leu858arg"]:
            spss_row["EGFR_Exon21_L858R"] = "Yes"    

    return spss_row

def get_target_sheets_from_specimen_vaf(row):
    specimen = str(row.get("Specimen", "")).strip().lower()
    sheets = set()

    def valid(v):
        s = str(v).strip().lower()
        return s not in ["", "na", "nan"]

    v_t = row.get("VAF (tissue)")
    v_l = row.get("VAF (liq)")
    v_p = row.get("VAF (pf)")
    v_c = row.get("VAF (csf)")

    # Tissue page destinations
    if specimen in ["tissue", "ffpe", "paraffin"] and valid(v_t):
        sheets.add("Tissue")
    if specimen == "csf" and valid(v_c):
        sheets.add("Tissue")
    if specimen in ["pf", "pleural", "pleural fluid"] and valid(v_p):
        sheets.add("Tissue")
    if "tissue and plasma" in specimen:
        if valid(v_t): sheets.add("Tissue")
        if valid(v_l): sheets.add("Liquid")
    if "pf and tissue" in specimen:
        if valid(v_t): sheets.add("Tissue")
        if valid(v_p): sheets.add("Liquid")
    if "csf and tissue" in specimen:
        if valid(v_t): sheets.add("Tissue")
        if valid(v_c): sheets.add("Liquid")
    if "plasma and pleural" in specimen:
        if valid(v_p): sheets.add("Tissue")
        if valid(v_l): sheets.add("Liquid")

    # Single-source liquid
    if specimen in ["plasma", "ctdna", "liquid", "blood"] and valid(v_l):
        sheets.add("Tissue")

    return list(sheets)

def extract_common_g3_values(df_group, g3_columns):
    values = {}
    for col in g3_columns:
        if col not in df_group.columns:
            continue
        clean_vals = [
            normalize_g3_value(v) for v in df_group[col]
            if normalize_g3_value(v) != ""
        ]

        unique_vals = list(set(clean_vals))
        if len(unique_vals) == 1:
            values[col] = unique_vals[0]
    return values

def process_case_group_block(df_group, all_columns, specimen_type, g3_values, anchor_row):
    spss_row = {}

    # Split the known schema into clinician vs molbio from all_columns
    molbio_set = set(MOLBIO_COLUMNS)
    for colname in all_columns:
        if colname in molbio_set:
            # Default NA for mol-bio fields (gene & biomarkers)
            if re.match(r".*_(Mutation|Amplification|Fusion|Other_Mutation)(_\d+)?_Yes_No$", colname):
                spss_row[colname] = "No"
            elif colname in ["TMB_High_Low","MSI_High_Statble_Low","LOH_Score","HRD_Score",
                            "Telomeric_Allelic_Imbalance_Yes_No","Large_Scale_State_Transitions",
                            "Genes_Tier4_Name","Genes_Tier4_Alterations_Yes_No"]:
                spss_row[colname] = "NA"
            else:
                spss_row[colname] = "NA"
        else:
            # Clinician/admin → default empty
            spss_row[colname] = ""

    for col in ["EGFR_UncommonEGFR", "EGFR_UncommonExon18", "EGFR_UncommonExon19", "EGFR_UncommonExon20", "EGFR_UncommonExon21"]:
        spss_row[col] = "No"

    spss_row["MRN"] = first_val(df_group, "MRN")
    spss_row["Name"] = first_val(df_group, "Patient_name")

    date_reporting = first_val(df_group, "Date-reporting_date")
    spss_row["Date_NGS_Perfomed"] = date_reporting
    spss_row["NGS_Year"] = date_reporting[:4] if len(date_reporting) >= 4 and date_reporting[:4].isdigit() else "NA"

    spss_row["Date_Discussed_MTB"] = first_val(df_group, "Date-Case_discussed")
    spss_row["NGS_Done_Yes_No"] = "Yes"

    spss_row["Source_File"] = first_val(df_group, "Source_File", "NA")

    lab = first_val(df_group, "Reporting_company", "")
    lab_l = lab.lower()
    if lab_l in ("", "na", "nan"):
        spss_row["TMH_Outside_Lab"] = "NA"
    elif "tmh" in lab_l:
        spss_row["TMH_Outside_Lab"] = "TMH, Mumbai"
    else:
        spss_row["TMH_Outside_Lab"] = f"Outside, {lab}"

    molbio_set = set(MOLBIO_COLUMNS)
    for c in all_columns:
        if c in molbio_set:
            continue  # skip mol-bio schema; those are handled separately
        if c in df_group.columns:
            v = str(df_group[c].iloc[0]).strip()
            # Only overwrite empty defaults; do not clobber admin we already set
            if spss_row.get(c, "") in ("", "NA") and v.lower() not in ("", "na", "nan"):
                spss_row[c] = v

    specimen_vals = df_group["Specimen"].dropna().unique()
    raw_specimen = str(specimen_vals[0]).strip().lower() if len(specimen_vals) else ""

    if raw_specimen == "tissue and plasma":
        spss_row["Specimen"] = "Paired Tissue (FFPE)" if specimen_type == "Tissue" else "Paired Blood"
    elif raw_specimen in ["tissue", "ffpe", "paraffin"]:
        spss_row["Specimen"] = "Tissue (FFPE)"
    elif raw_specimen in ["plasma", "ctdna", "liquid", "blood"]:
        spss_row["Specimen"] = "Blood"
    elif raw_specimen in ["pf", "pleural", "pleural fluid"]:
        # pf-only goes to Tissue page per new rule
        spss_row["Specimen"] = "Others (Pleural)" if specimen_type == "Tissue" else "NA"
    elif raw_specimen == "csf":
        # csf-only goes to Tissue page per new rule
        spss_row["Specimen"] = "Others (CSF)" if specimen_type == "Tissue" else "NA"
    elif "pf and tissue" in raw_specimen:
        spss_row["Specimen"] = "Tissue (FFPE)" if specimen_type == "Tissue" else "Others (Pleural)"
    elif "csf and tissue" in raw_specimen:
        spss_row["Specimen"] = "Tissue (FFPE)" if specimen_type == "Tissue" else "Others (CSF)"
    elif "plasma and pleural" in raw_specimen:
        spss_row["Specimen"] = "Others (Pleural)" if specimen_type == "Tissue" else "Blood"

    # G3 Fields
    if "NGS_Tech_used" in g3_values:
        spss_row["NGS_Panel_Technique"] = g3_values["NGS_Tech_used"]
    if "Depth_of_Coverage" in g3_values:
        spss_row["Depth_Of_Coverage"] = g3_values["Depth_of_Coverage"]
    if "RNA_Passing_Filter_Reads" in g3_values:
        spss_row["RNA_Passing_Filter_Reads"] = g3_values["RNA_Passing_Filter_Reads"]
    if "Tumor_Fraction" in g3_values:
        spss_row["Tumor_Fraction"] = g3_values["Tumor_Fraction"]

    # Decide which site to read for this page based on the block's Specimen
    raw_specimen = first_val(df_group, "Specimen", "").lower()

    def pick_site_for_page(raw_specimen, page):
        # Tissue page rules (pf-only & csf-only go to Tissue)
        if page == "Tissue":
            if raw_specimen in ["pf", "pleural", "pleural fluid"]:
                return "pf"
            if raw_specimen == "csf":
                return "csf"
            if raw_specimen in ["plasma", "ctdna", "liquid", "blood"]:
                return "liq"
            return "tissue"  # default for tissue or combos including tissue

        # Liquid page rules (non-tissue component of combos, or plasma-alone)
        if page == "Liquid":
            if "tissue and plasma" in raw_specimen:
                return "liq"
            if "pf and tissue" in raw_specimen:
                return "pf"
            if "csf and tissue" in raw_specimen:
                return "csf"
            if "plasma and pleural" in raw_specimen:
                return "liq"
        return None

    site = pick_site_for_page(raw_specimen, specimen_type)

    def get_first_non_na(col):
        if col not in df_group.columns:
            return "NA"
        vals = [str(v).strip() for v in df_group[col].tolist()
                if str(v).strip().lower() not in ["", "na", "nan"]]
        return vals[0] if vals else "NA"

    if site:
        spss_row["TMB_High_Low"] = get_first_non_na(f"TMB ({site})")
        spss_row["MSI_High_Statble_Low"] = get_first_non_na(f"MSI ({site})")
        spss_row["LOH_Score"] = get_first_non_na(f"LOH_Score ({site})")
        spss_row["HRD_Score"] = get_first_non_na(f"HRD_Score ({site})")
        spss_row["Telomeric_Allelic_Imbalance_Yes_No"] = get_first_non_na(f"Telomeric_Allelic_Imbalance ({site})")
        spss_row["Large_Scale_State_Transitions"] = get_first_non_na(f"Large_Scale_State_Transition ({site})")
        spss_row["Genes_Tier4_Name"] = get_first_non_na("Gene_Tier_4_Name")
        tier4_val = str(spss_row["Genes_Tier4_Name"]).strip().lower()
        spss_row["Genes_Tier4_Alterations_Yes_No"] = "Yes" if tier4_val and tier4_val not in ["na", "no", ""] else "No"

    # 🧬 Call your mutation/amplification/fusion code here
    mutation_block_from_group(df_group, spss_row, specimen_type)

    return spss_row


def get_g3_signature(row, g3_fields):
    return tuple([normalize_g3_value(row.get(col, '')) for col in g3_fields])


def process_case_group(df_group, all_columns, specimen_type):
    df_group = df_group.copy()

    output_rows = []

    # Group rows by consecutive Specimen changes
    specimen_blocks = []
    current_block = []
    last_specimen = None

    for _, row in df_group.iterrows():
        current_specimen = str(row["Specimen"]).strip().lower()
        if current_specimen != last_specimen and current_block:
            specimen_blocks.append(pd.DataFrame(current_block))
            current_block = []
        current_block.append(row)
        last_specimen = current_specimen
    if current_block:
        specimen_blocks.append(pd.DataFrame(current_block))

    anchor_row = df_group.iloc[0]  # first row of entire group

    # Now process each block independently
    for block in specimen_blocks:
        raw_specimen = str(block.iloc[0].get("Specimen", "")).strip().lower()
        if not is_specimen_allowed_for_page(raw_specimen, specimen_type):
            continue
        filtered_block = filter_block_for_specimen_type(block, specimen_type)
        if filtered_block.empty:
            continue

        g3_fields = [
            "NGS_Tech_used",
            "Depth_of_Coverage",
            "RNA_Passing_Filter_Reads",
            "Tumor_Fraction",
            "Date-reporting_date"
        ]
        g3_values = extract_common_g3_values(filtered_block, g3_fields)

        # Updated: G2 logic from entire block

        row_out = process_case_group_block(filtered_block, all_columns, specimen_type, g3_values, anchor_row)
        output_rows.append(row_out)

    return output_rows

def build_workbook(input_xlsx_path, order_path, output_xlsx_path):
    df = pd.read_excel(input_xlsx_path, sheet_name='All')

    # Desired column order from the provided file (Excel/CSV)
    desired_order = load_header_order(order_path)

    grouped = df.groupby(["MRN", "Date-Case_discussed"], sort=False, dropna=False)
    sheets = {"Tissue": [], "Liquid": []}

    for _, group in grouped:
        group = group.copy()
        target_sheets = set()
        for _, row in group.iterrows():
            target_sheets.update(get_target_sheets_from_specimen_vaf(row))

        for specimen_block in target_sheets:
            # Pass desired_order into processing so row init has a consistent schema
            processed = process_case_group(group, desired_order, specimen_block)
            if any(row for row in processed if any(val not in ["", "NA", "nan"] for val in row.values())):
                sheets[specimen_block].extend(processed)

    with pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
        for sheet, rows in sheets.items():
            df_sheet = pd.DataFrame(rows)

            # (Optional) run your renames BEFORE diff so names match the order file
            if "ERBB2_Mutation_AA_Change" in df_sheet.columns:
                df_sheet.rename(columns={"ERBB2_Mutation_AA_Change": "ERBB2_AA_Change"}, inplace=True)

            # --- diff: what's missing or extra vs desired order?
            present = list(df_sheet.columns)
            desired_set = set(desired_order)
            present_set = set(present)

            missing = [c for c in desired_order if c not in present_set]
            extra = [c for c in present if c not in desired_set]

            if missing:
                print(f"[{sheet}] Missing columns (expected but not produced): {missing}")
            if extra:
                print(f"[{sheet}] Extra columns (produced but not in order file): {extra}")

            # --- reorder to desired order (then optionally keep extras at end)
            extras_in_order = [c for c in present if c not in desired_set]
            final_order = desired_order + extras_in_order
            df_sheet = df_sheet.reindex(columns=final_order, fill_value="")

            # If you truly want to drop a column such as Source_File, do it AFTER the diff:
            # if "Source_File" in df_sheet.columns:
            #     df_sheet.drop(columns=["Source_File"], inplace=True)

            df_sheet.to_excel(writer, sheet_name=sheet, index=False)

def main():
    # 1) Input Excel
    input_xlsx_path = input("Enter INPUT Excel path: ").strip()
    if not os.path.isfile(input_xlsx_path):
        raise SystemExit(f"❌ File not found: {input_xlsx_path}")
    
    # 3) Optional column-order file (Excel/CSV). Leave blank to skip.
    order_path = input("Enter OPTIONAL column-order file (Excel/CSV) path (or leave blank): ").strip()
    if order_path and not os.path.isfile(order_path):
        raise SystemExit(f"❌ Column-order file not found: {order_path}")

    # 2) Output folder
    output_folder = input("Enter OUTPUT folder path: ").strip()
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Build default output file path
    base = os.path.splitext(os.path.basename(input_xlsx_path))[0]
    output_xlsx_path = os.path.join(output_folder, f"{base}_formatted.xlsx")

    # Optional SPSS template (CSV) beside this script
    # Use order file only if the user provided one
    if order_path:
        # Normal path: use provided order (CSV/Excel)
        build_workbook(input_xlsx_path, order_path, output_xlsx_path)
    else:
        # --- Fallback path: NO order file ---
        df = pd.read_excel(input_xlsx_path, sheet_name='All')

        # 1) Detect clinician columns (from "Serial" to the end)
        cols_lower = [str(c).strip().lower() for c in df.columns]
        if "serial" in cols_lower:
            serial_idx = cols_lower.index("serial")
            CLINICIAN_COLUMNS = list(df.columns[serial_idx:])
        else:
            CLINICIAN_COLUMNS = list(df.columns)

        # 2) Build final schema (clinician + molbio)
        REQUIRED_ADMIN = [
            "MRN", "Patient_name", "Date-Case_discussed", "Date-reporting_date",
            "Reporting_company", "Specimen", "Source_File"
        ]
        seen = set()
        FINAL_COLUMNS = []
        for c in REQUIRED_ADMIN + CLINICIAN_COLUMNS:
            if c not in seen:
                FINAL_COLUMNS.append(c); seen.add(c)
        FINAL_COLUMNS += MOLBIO_COLUMNS

        grouped = df.groupby(["MRN", "Date-Case_discussed"], sort=False, dropna=False)

        # 3) Produce rows with known schema passed into processing
        sheets = {"Tissue": [], "Liquid": []}
        for _, group in grouped:
            group = group.copy()
            target_sheets = set()
            for _, row in group.iterrows():
                target_sheets.update(get_target_sheets_from_specimen_vaf(row))
            for specimen_block in target_sheets:
                processed = process_case_group(group, FINAL_COLUMNS, specimen_block)
                if any(row for row in processed if any(val not in ["", "NA", "nan"] for val in row.values())):
                    sheets[specimen_block].extend(processed)

        # 4) Write (no reordering file provided, so we just write FINAL_COLUMNS)
        with pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
            for sheet, rows in sheets.items():
                df_sheet = pd.DataFrame(rows, columns=FINAL_COLUMNS)

                # (Optional) do your renames/drops AFTER any diff checks (none in fallback)
                if "ERBB2_Mutation_AA_Change" in df_sheet.columns:
                    df_sheet.rename(columns={"ERBB2_Mutation_AA_Change": "ERBB2_AA_Change"}, inplace=True)
                # KEEP "Source_File" unless you really want to drop it; dropping hides it from any later diff
                # if "Source_File" in df_sheet.columns:
                #     df_sheet.drop(columns=["Source_File"], inplace=True)

                df_sheet.to_excel(writer, sheet_name=sheet, index=False)


    print(f"✅ Saved: {output_xlsx_path}")

if __name__ == "__main__":
    main()


# build_workbook("MTB_Details_2025.xlsx", "spss_empty_template.csv", "final_output.xlsx")