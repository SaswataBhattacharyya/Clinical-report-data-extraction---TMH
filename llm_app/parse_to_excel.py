import os
import re
import json
import shutil
import pandas as pd
from tqdm import tqdm
import difflib

# ---------- FIELDS (updated schema) ----------
PATIENT_FIELDS = [
    "Patient_name", "Case_ID", "Cases_Discussed_Not_discussed",
    "Specimen", "Date-reporting_date",
    "Year_of_NGS", "NGS_Tech_used",
    "TMB (tissue)", "TMB (liq)", "TMB (pf)", "TMB (csf)", "TMB (others)",
    "MSI (tissue)", "MSI (liq)", "MSI (pf)", "MSI (csf)", "MSI (others)",
    "Telomeric_Allelic_Imbalance (tissue)", "Telomeric_Allelic_Imbalance (liq)",
    "Telomeric_Allelic_Imbalance (pf)", "Telomeric_Allelic_Imbalance (csf)",
    "Telomeric_Allelic_Imbalance (others)",
    "Large_Scale_State_Transition (tissue)", "Large_Scale_State_Transition (liq)",
    "Large_Scale_State_Transition (pf)", "Large_Scale_State_Transition (csf)",
    "Large_Scale_State_Transition (others)",
    "LOH_Score (tissue)", "LOH_Score (liq)", "LOH_Score (pf)",
    "LOH_Score (csf)", "LOH_Score (others)",
    "HRD_Score (tissue)", "HRD_Score (liq)", "HRD_Score (pf)",
    "HRD_Score (csf)", "HRD_Score (others)",
    "Tumor_Fraction", "Gene_Tier_4_Name", "Depth_of_Coverage", "RNA_Passing_Filter_Reads",
]

GENE_FIELDS = [
    "Gene_name",
    "Mutation(nucleotide)", "Mutation(protein)",
    "VAF (tissue)", "VAF (liq)", "VAF (pf)", "VAF (csf)", "VAF (others)",
    "Exon_no", "NM ID",
    "Fusion",
    "Fusion_Read_Count (tissue)", "Fusion_Read_Count (liq)", "Fusion_Read_Count (pf)",
    "Fusion_Read_Count (csf)", "Fusion_Read_Count (others)",
    "Amplification",
    "Amp_read/copy number (tissue)", "Amp_read/copy number (liq)",
    "Amp_read/copy number (pf)", "Amp_read/copy number (csf)", "Amp_read/copy number (others)",
    "Variant Type", "Tier",
]

FINAL_FIELDS = PATIENT_FIELDS + GENE_FIELDS + [
    "Reporting_company", "Date-Case_discussed", "Source_File"
]

# ---------- TEXT PARSER (kept, but supports expanded fields) ----------
# We split the whole content into two parts:
#  - header/common fields (patient-level)
#  - one or more gene blocks beginning with "Gene_name:"
GENE_BLOCK_SPLIT = re.compile(r"\n(?=Gene_name\s*:)", flags=re.IGNORECASE)

# --- HELPER: normalize filenames for robust matching ---

DEBUG = False  # toggle verbose diagnostics

def _dprint(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)

def _sample(seq, n=10):
    seq = list(seq)
    return seq[:n] if len(seq) > n else seq
def _norm_name(x):
    import os
    return os.path.basename(str(x)).strip().lower()

def _stem_key(x: str) -> str:
    """
    Make a case-insensitive 'stem' for matching:
      - excel1: remove trailing '_gpt.txt' / '_gpt.json'
      - excel2: remove trailing '.pdf'
    Do NOT strip generic extensions like '.25' (part of dates).
    """
    import os, re
    b = os.path.basename(str(x)).strip()

    # Remove trailing GPT markers + txt/json (excel1 side pattern)
    b = re.sub(r'(?i)(?:_gpt)?\.(?:txt|json)$', '', b)

    # Remove only a true .pdf (excel2 side pattern)
    b = re.sub(r'(?i)\.pdf$', '', b)

    # If you also want to handle Excel/CSV inputs from excel2, add them here:
    # b = re.sub(r'(?i)\.(?:xlsx|xls|csv)$', '', b)

    return b.strip().lower()



def parse_gpt_text(content: str, filename: str):
    content = content.strip()
    if not content:
        row = {f: "NA" for f in FINAL_FIELDS}
        row["Source_File"] = filename
        return [row]

    # Split blocks: first chunk = common fields; subsequent chunks = gene-level
    blocks = GENE_BLOCK_SPLIT.split(content)
    common_chunk = blocks[0] if blocks else ""
    gene_chunks = blocks[1:] if len(blocks) > 1 else []

    # Extract common (patient-level) fields from the common chunk
    common_data = {}
    for line in common_chunk.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key_l = key.strip().lower()
        for pf in PATIENT_FIELDS + ["Reporting_company", "Date-Case_discussed"]:
            if key_l == pf.lower():
                common_data[pf] = val.strip()
                break

    # If no gene chunks, still emit a single row with NAs for gene fields
    if not gene_chunks:
        row = {f: "NA" for f in FINAL_FIELDS}
        # fill patient fields found
        for k, v in common_data.items():
            row[k] = v
        row["Source_File"] = filename
        return [row]

    rows = []
    for gchunk in gene_chunks:
        row = {f: "NA" for f in FINAL_FIELDS}
        # put patient/common values first
        for k, v in common_data.items():
            row[k] = v
        # parse gene-level lines
        for line in gchunk.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key_l = key.strip().lower()
            val = val.strip()
            for gf in GENE_FIELDS:
                if key_l == gf.lower():
                    row[gf] = val
                    break
        row["Source_File"] = filename
        rows.append(row)

    return rows


# ---------- JSON PARSER ----------
def parse_gpt_json_file(json_path: str, filename_for_source: str):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to read JSON {json_path}: {e}")
        row = {f: "NA" for f in FINAL_FIELDS}
        row["Source_File"] = filename_for_source
        return [row]

    if not isinstance(data, dict) or "records" not in data or not isinstance(data["records"], list):
        print(f"⚠️ Malformed JSON in {json_path}: missing top-level 'records' list")
        row = {f: "NA" for f in FINAL_FIELDS}
        row["Source_File"] = filename_for_source
        return [row]

    rows = []
    for rec in data["records"]:
        row = {f: "NA" for f in FINAL_FIELDS}
        if isinstance(rec, dict):
            for key in FINAL_FIELDS:
                if key in rec and isinstance(rec[key], str):
                    row[key] = rec[key].strip()
                elif key in rec and rec[key] is not None:
                    row[key] = str(rec[key])
        row["Source_File"] = filename_for_source
        rows.append(row)

    if not rows:
        row = {f: "NA" for f in FINAL_FIELDS}
        row["Source_File"] = filename_for_source
        rows = [row]

    return rows


# ---------- MAIN ----------
def main(input_root=None, output_root=None, excel2_path=None):
    if input_root is None:
        INPUT_ROOT = input("Enter INPUT root (folder containing *_gpt.json / *_gpt.txt): ").strip()
    else:
        INPUT_ROOT = input_root

    if output_root is None:
        OUTPUT_ROOT = input("Enter OUTPUT root (where compiled Excel + copied CSVs go): ").strip()
    else:
        OUTPUT_ROOT = output_root

    # --- Load excel2 (REQUIRES: filename, MRN, Patient_name, Cases_Discussed_Not_discussed) ---
    if excel2_path is None:
        EXCEL2_PATH = input("Enter INPUT excel2 path: ").strip()
    else:
        EXCEL2_PATH = excel2_path

    # dicts keyed by suffix/stem
    mrn_map, pname_map, caseflag_map = {}, {}, {}

    try:
        if EXCEL2_PATH:
            try:
                df_map = pd.read_excel(EXCEL2_PATH, dtype={"MRN": str})
                # Normalize headers (strip spaces)
                df_map = df_map.rename(columns={c: c.strip() for c in df_map.columns})

                required = {"filename", "MRN", "Patient_name", "Cases_Discussed_Not_discussed"}
                if not required.issubset(set(df_map.columns)):
                    missing = required.difference(set(df_map.columns))
                    raise ValueError(f"excel2 missing required columns: {sorted(missing)}")

                # Use only the required columns
                df_map = df_map[["filename", "MRN", "Patient_name", "Cases_Discussed_Not_discussed"]].copy()
                # Build stem key (strip .pdf OR _gpt.txt/_gpt.json then one ext)
                df_map["__key"] = df_map["filename"].map(_stem_key)

                mrn_map      = dict(zip(df_map["__key"], df_map["MRN"].astype(str).str.strip()))
                pname_map    = dict(zip(df_map["__key"], df_map["Patient_name"].astype(str).str.strip()))
                caseflag_map = dict(zip(df_map["__key"], df_map["Cases_Discussed_Not_discussed"].astype(str).str.strip()))

                if DEBUG:
                    # Basic stats
                    _dprint(f"excel2 rows: {len(df_map)}")
                    _dprint("excel2 keys (sample):", _sample(df_map["__key"].unique(), 12))

                    # Check for duplicate keys in excel2 (these can cause silent overwrites)
                    dup_keys = df_map["__key"][df_map["__key"].duplicated(keep=False)]
                    if len(dup_keys):
                        _dprint(f"WARNING: {dup_keys.nunique()} duplicate key(s) in excel2; showing a few:")
                        _dprint(_sample(dup_keys.unique(), 10))

                    # Show a few filename → key transforms to confirm _stem_key()
                    preview = df_map[["filename", "__key"]].head(12).to_dict("records")
                    for row in preview:
                        _dprint(f"excel2 filename → key: {row['filename']} → {row['__key']}")


                print(f"Loaded excel2: {len(df_map)} rows "
                      f"(MRN:{len(mrn_map)}, Patient_name:{len(pname_map)}, Cases_Discussed:{len(caseflag_map)})")

            except Exception as e:
                print(f"⚠️ Could not load excel2 from {EXCEL2_PATH}: {e}")
                mrn_map, pname_map, caseflag_map = {}, {}, {}
        else:
            print("⚠️ No excel2 path provided; mapping will be skipped.")

        if not os.path.isdir(INPUT_ROOT):
            return {"success": False, "message": f"INPUT root does not exist: {INPUT_ROOT}"}
        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        for root, _, files in os.walk(INPUT_ROOT):
            rel_path = os.path.relpath(root, INPUT_ROOT)
            out_dir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(out_dir, exist_ok=True)

            if DEBUG:
                _dprint(f"[walk] {root} → {len(files)} files")
                _dprint("Files (sample):", _sample(sorted(files), 12))

            all_rows = []

            # First, copy any CSVs present
            for file in files:
                if file.lower().endswith(".csv"):
                    src = os.path.join(root, file)
                    dst = os.path.join(out_dir, file)
                    shutil.copy2(src, dst)

            # Parse JSON first (preferred), then TXT (legacy)
            for file in tqdm(files, desc=f"Parsing in {rel_path}"):
                in_path = os.path.join(root, file)
                low = file.lower()

                if low.endswith("_gpt.json"):
                    rows = parse_gpt_json_file(in_path, filename_for_source=file)
                    all_rows.extend(rows)

                elif low.endswith("_gpt.txt"):
                    with open(in_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    rows = parse_gpt_text(content, filename=file)
                    all_rows.extend(rows)

            # Write compiled Excel for this folder, if any rows
            if all_rows:
                folder_name = os.path.basename(root.rstrip(os.sep)) or "compiled"
                xlsx_path = os.path.join(out_dir, f"{folder_name}_compiled.xlsx")
                df = pd.DataFrame(all_rows, columns=FINAL_FIELDS)

                # --- Apply mappings by stem/suffix (excel1.Source_File vs excel2.filename) ---
                try:
                    # 1) Ensure Case ID / helper cols exist
                    case_id_col = "Case-ID" if "Case-ID" in df.columns else ("Case_ID" if "Case_ID" in df.columns else None)
                    if case_id_col is None:
                        case_id_col = "Case_ID"
                        df[case_id_col] = "NA"

                    if "Patient_name" not in df.columns:
                        df["Patient_name"] = "NA"
                    if "Cases_Discussed_Not_discussed" not in df.columns:
                        df["Cases_Discussed_Not_discussed"] = "NA"

                    # 2) Build keys on excel1 side
                    df["__key"] = df["Source_File"].map(_stem_key)

                    if DEBUG:
                        _dprint("excel1 keys (sample):", _sample(df["__key"].dropna().unique(), 12))
                        # Show a few Source_File → key transforms
                        for sf, key in df[["Source_File","__key"]].head(12).values:
                            _dprint(f"excel1 source → key: {sf} → {key}")

                    # 3) Hits check per-map
                    keys_mrn   = set(mrn_map.keys())
                    keys_pname = set(pname_map.keys())
                    keys_flag  = set(caseflag_map.keys())
                    keys_all   = keys_mrn | keys_pname | keys_flag

                    df["__hit_mrn"]   = df["__key"].isin(keys_mrn)
                    df["__hit_pname"] = df["__key"].isin(keys_pname)
                    df["__hit_flag"]  = df["__key"].isin(keys_flag)
                    df["__hit_any"]   = df["__hit_mrn"] | df["__hit_pname"] | df["__hit_flag"]

                    if DEBUG:
                        _dprint(f"excel1 rows total: {len(df)}")
                        _dprint(f"hits: MRN={int(df['__hit_mrn'].sum())}, "
                                f"PName={int(df['__hit_pname'].sum())}, "
                                f"Flag={int(df['__hit_flag'].sum())}, "
                                f"Any={int(df['__hit_any'].sum())}")

                        # Show a few non-matching rows with close-match suggestions
                        nomatch = df.loc[~df["__hit_any"], ["Source_File","__key"]].head(20).copy()
                        if len(nomatch):
                            _dprint(f"First {len(nomatch)} non-matching rows:")
                            for _, r in nomatch.iterrows():
                                key = r["__key"]
                                sugg = difflib.get_close_matches(key, keys_all, n=2, cutoff=0.6)
                                _dprint(f"  NO MATCH: {r['Source_File']} | key={key} | suggestions={sugg}")

                    # 4) Apply mappings (only where there are hits)
                    hits = df["__hit_any"]

                    if hits.any():
                        k = df.loc[hits, "__key"]

                        # Save pre-values (for delta diagnostics)
                        if DEBUG:
                            before_case = df.loc[hits, case_id_col].copy()
                            before_name = df.loc[hits, "Patient_name"].copy()
                            before_flag = df.loc[hits, "Cases_Discussed_Not_discussed"].copy()

                        if mrn_map:
                            df.loc[hits, case_id_col] = k.map(mrn_map).fillna(df.loc[hits, case_id_col])

                        if pname_map:
                            df.loc[hits, "Patient_name"] = k.map(pname_map).fillna(df.loc[hits, "Patient_name"])

                        if caseflag_map:
                            df.loc[hits, "Cases_Discussed_Not_discussed"] = (
                                k.map(caseflag_map).fillna(df.loc[hits, "Cases_Discussed_Not_discussed"])
                            )

                        if DEBUG:
                            # Count how many actually changed
                            changed_case = (df.loc[hits, case_id_col] != before_case).sum()
                            changed_name = (df.loc[hits, "Patient_name"] != before_name).sum()
                            changed_flag = (df.loc[hits, "Cases_Discussed_Not_discussed"] != before_flag).sum()
                            _dprint(f"updated values: CaseID={int(changed_case)}, "
                                    f"Patient_name={int(changed_name)}, Cases_Discussed={int(changed_flag)}")
                    else:
                        print("→ No updates from excel2 (no stem/suffix matches).")

                    # 5) Reorder flag column next to Case ID
                    cols = list(df.columns)
                    if "Cases_Discussed_Not_discussed" in cols:
                        cols.remove("Cases_Discussed_Not_discussed")
                        insert_at = cols.index(case_id_col) + 1 if case_id_col in cols else 0
                        cols.insert(insert_at, "Cases_Discussed_Not_discussed")
                        df = df[cols]

                    # 6) Optional: write a per-folder debug CSV
                    if DEBUG:
                        debug_out = os.path.join(out_dir, f"{folder_name}__mapping_debug.csv")
                        debug_cols = ["Source_File","__key","Patient_name",case_id_col,"Cases_Discussed_Not_discussed",
                                    "__hit_mrn","__hit_pname","__hit_flag","__hit_any"]
                        df[debug_cols].to_csv(debug_out, index=False)
                        _dprint(f"mapping debug → {debug_out}")

                    # Cleanup helper cols
                    df.drop(columns=["__key","__hit_mrn","__hit_pname","__hit_flag","__hit_any"], errors="ignore", inplace=True)

                    print(f"→ Updated from excel2 for {int(hits.sum())} rows (stem/suffix match).")
                except Exception as e:
                    print(f"⚠️ Mapping step failed for {xlsx_path}: {e}")


                # Save after mapping
                df.to_excel(xlsx_path, index=False)
                print(f"✅ Saved: {xlsx_path}  ({len(all_rows)} rows)")

        return {"success": True, "message": f"Successfully parsed to Excel from {INPUT_ROOT}"}
    except Exception as e:
        return {"success": False, "message": f"Error during parse to Excel: {str(e)}"}


if __name__ == "__main__":
    result = main()
    if result:
        print(result["message"])

