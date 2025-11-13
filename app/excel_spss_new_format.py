#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import re
import pandas as pd
from typing import List, Optional, Tuple

# -----------------------------
# Helpers to read many Excels
# -----------------------------
EXCEL_EXTS = {".xlsx", ".xls", ".xlsm"}

def list_excel_files(folder: str) -> List[str]:
    paths = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in EXCEL_EXTS:
            paths.append(p)
    paths.sort()
    return paths

def read_excels_by_date(folder: str) -> dict:
    files = list_excel_files(folder)
    if len(files) > 1:
        # Case A: multiple files
        out = {}
        for f in files:
            date_key = os.path.splitext(os.path.basename(f))[0]
            df = pd.read_excel(f, sheet_name=0, dtype=object)
            out[date_key] = df
        return out
    elif len(files) == 1:
        # Case B: single file, multiple sheets
        f = files[0]
        xls = pd.ExcelFile(f)
        out = {}
        for sheet in xls.sheet_names:
            date_key = sheet.strip()
            df = xls.parse(sheet_name=sheet, dtype=object)
            out[date_key] = df
        return out
    return {}

# -------------------------------------------
# Group-key building with hierarchical fallback
# -------------------------------------------
def build_groupkey_inplace(df: pd.DataFrame,
                           mrn_col: str = "MRN",
                           date_discussed: str = "Date-Case_discussed") -> pd.DataFrame:
    """
    Build a composite key in df["_groupkey"]:
    - Always start with MRN
    - If Date-Case_discussed exists, append it (so rows with same MRN but different discussed date split)
    - Else just MRN
    """
    if mrn_col not in df.columns:
        df[mrn_col] = ""

    keys = []
    for _, row in df.iterrows():
        mrn = str(row.get(mrn_col, "")).strip()
        dc = str(row.get(date_discussed, "")).strip()

        if dc and dc.lower() not in {"nan", "na"}:
            keys.append(mrn + "|" + dc)
        else:
            keys.append(mrn)

    df["_groupkey"] = keys
    return df

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

# -------------------------------------------
# Merge logic (long format, clinician to first row)
# -------------------------------------------
def pick_best_row(df_group: pd.DataFrame) -> pd.Series:
    """
    Choose a single clinician row out of a group.
    Strategy: the row with the most non-null/non-empty values (excluding __source_file__ and keys).
    If tie, take the first one by original order.
    """
    if len(df_group) == 1:
        return df_group.iloc[0]

    def score_row(s: pd.Series) -> int:
        cnt = 0
        for c, v in s.items():
            if c.startswith("__"):
                continue
            sv = str(v).strip()
            if sv and sv.lower() not in {"nan", "na"}:
                cnt += 1
        return cnt

    scores = df_group.apply(score_row, axis=1)
    idx = scores.idxmax()
    return df_group.loc[idx]

BLANK_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "n.a.", "."}

def is_blankish_series(s: pd.Series) -> pd.Series:
    return s.isna() | s.astype(str).str.strip().str.lower().isin(BLANK_TOKENS)

def first_non_blank(s: pd.Series):
    m = ~is_blankish_series(s)
    if m.any():
        return s[m].iloc[0]
    return pd.NA

def normalize_dc_to_ddmmyy(series: pd.Series) -> pd.Series:
    """
    Canonicalise to 'DD.MM.YY'.
    Accepts 'DD.MM.YY', 'DD.MM.YYYY', separators '.', '/', '-', and Excel serials.
    Leaves truly unparsable as ''.
    """
    raw = series.copy()

    # unify separators to '.'
    s = raw.astype(str).str.strip().str.replace(r"[/-]", ".", regex=True)

    # "DD.MM.YYYY" -> "DD.MM.YY"
    s = s.str.replace(
        r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$",
        lambda m: f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)[-2:]}",
        regex=True,
    )
    # ensure leading zeros for day/month in "DD.MM.YY"
    s = s.str.replace(
        r"^(\d{1,2})\.(\d{1,2})\.(\d{2})$",
        lambda m: f"{int(m.group(1)):02d}.{int(m.group(2)):02d}.{m.group(3)}",
        regex=True,
    )

    # strict parse now that we forced "DD.MM.YY"
    dt = pd.to_datetime(s, format="%d.%m.%y", errors="coerce")

    # Excel numeric fallback
    mask = dt.isna()
    if mask.any():
        num = pd.to_numeric(raw, errors="coerce")
        dt.loc[mask & num.notna()] = pd.to_datetime(
            num[mask & num.notna()], errors="coerce", unit="D", origin="1899-12-30"
        )

    out = dt.dt.strftime("%d.%m.%y")
    return out.fillna("").replace("NaT", "")

def merge_long_format(
    molbio: pd.DataFrame,
    clinician: pd.DataFrame,
    verbose: bool = True,
    apply_blanking: bool = False,    # blanks duplicates within groups
    apply_filldown: bool = False,   # NEW: forward-fills within groups (opposite of blanking)
    sample: int = 10
) -> pd.DataFrame:
    """
    Verbose, merge-first implementation with MRN primary + Date fallback.
    - No blanking unless apply_blanking=True.
    - Prints diagnostics to help pinpoint key mismatches.
    """

    mrn_col = "MRN"
    date_col = "Date-Case_discussed"

    # ---- validations ----
    if mrn_col not in molbio.columns or mrn_col not in clinician.columns:
        raise ValueError("Both molbio and clinician must contain 'MRN'.")

    # ---- copies & column hygiene ----
    m = molbio.copy()
    c = clinician.copy()
    m.columns = [str(x).strip() for x in m.columns]
    c.columns = [str(x).strip() for x in c.columns]

    # Normalize MRN
    def clean_mrn(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip()
        s = s.str.replace(r"\s+", " ", regex=True)
        s = s.str.replace(r"\.0$", "", regex=True)
        return s

    m[mrn_col] = clean_mrn(m[mrn_col])
    c[mrn_col] = clean_mrn(c[mrn_col])

    # Normalize Date-Case_discussed ONLY if present; else set ""
    if date_col in m.columns:
        m[date_col] = normalize_dc_to_ddmmyy(m[date_col])
    else:
        m[date_col] = ""

    if date_col in c.columns:
        c[date_col] = normalize_dc_to_ddmmyy(c[date_col])
    else:
        c[date_col] = ""

    # ---- helpers for diagnostics ----
    BLANK_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "n.a.", "."}
    def is_blankish_series(s: pd.Series) -> pd.Series:
        return s.isna() | s.astype(str).str.strip().str.lower().isin(BLANK_TOKENS)

    def first_non_blank(s: pd.Series):
        mask = ~is_blankish_series(s)
        if mask.any():
            return s[mask].iloc[0]
        return pd.NA

    # ---- MRN/date cardinalities ----
    m_mrns = set(m[mrn_col].astype(str))
    c_mrns = set(c[mrn_col].astype(str))
    common_mrns = m_mrns & c_mrns

    # distinct non-blank dates per MRN
    def nunique_nonblank_date(df):
        s = df[date_col]
        nb = ~is_blankish_series(s)
        return df.loc[nb].groupby(mrn_col)[date_col].nunique()

    m_nd = nunique_nonblank_date(m)
    c_nd = nunique_nonblank_date(c)
    mrns_m_multidate = set(m_nd[m_nd > 1].index)
    mrns_c_multidate = set(c_nd[c_nd > 1].index)
    mrns_multidate = mrns_m_multidate | mrns_c_multidate

    # ---- clinician collapsing (prevents many-to-many on MRN-only) ----
    clin_cols = [col for col in c.columns if col not in {mrn_col, date_col, "_groupkey", "__source_file__"}]
    if len(clin_cols) == 0:
        c1_collapsed = c[[mrn_col]].drop_duplicates().copy()
        c2_collapsed = c[[mrn_col, date_col]].drop_duplicates().copy()
    else:
        c1_collapsed = (
            c.groupby(mrn_col, dropna=False)[clin_cols]
             .agg(first_non_blank)
             .reset_index()
        )
        c2_collapsed = (
            c.groupby([mrn_col, date_col], dropna=False)[clin_cols]
             .agg(first_non_blank)
             .reset_index()
        )

    # ---- split mol-bio by MRN kind (single-date vs multi-date) ----
    m2 = m[m[mrn_col].isin(mrns_multidate)].copy()   # join on (MRN, Date)
    m1 = m[~m[mrn_col].isin(mrns_multidate)].copy()  # join on MRN

    # ---- do the joins ----
    merged_parts = []
    merged_m1 = pd.DataFrame()
    merged_m2 = pd.DataFrame()

    if not m1.empty:
        merged_m1 = pd.merge(
            m1, c1_collapsed, on=mrn_col, how="left", suffixes=("", "_clin"), copy=False
        )
        merged_parts.append(merged_m1)

    if not m2.empty:
        merged_m2 = pd.merge(
            m2, c2_collapsed, on=[mrn_col, date_col], how="left", suffixes=("", "_clin"), copy=False
        )
        merged_parts.append(merged_m2)

    merged = pd.concat(merged_parts, ignore_index=True, sort=False) if merged_parts else m.copy()

    # =========================
    # VERBOSE DIAGNOSTICS
    # =========================
    if verbose:
        print("—"*72)
        print("[merge] molbio rows:", len(m), " | clinician rows:", len(c))
        print("[merge] distinct MRNs  molbio:", len(m_mrns), "  clinician:", len(c_mrns),
              "  common:", len(common_mrns))
        print("[merge] MRNs that require MRN+Date join (multi-date on either side):", len(mrns_multidate))
        if mrns_multidate:
            print("       e.g.:", list(sorted(mrns_multidate))[:sample])

        # for multi-date MRNs, show date sets on each side
        shown = 0
        for mrn in mrns_multidate:
            if shown >= sample: break
            md = set(m.loc[m[mrn_col]==mrn, date_col].astype(str).tolist())
            cd = set(c.loc[c[mrn_col]==mrn, date_col].astype(str).tolist())
            # strip blanks for clarity
            md_clean = sorted([x for x in md if x.strip() != ""])
            cd_clean = sorted([x for x in cd if x.strip() != ""])
            if md_clean != cd_clean:
                print(f" [warn] MRN {mrn}: molbio dates {md_clean} vs clinician {cd_clean}")
                if not md_clean:
                    print("        -> molbio has blank dates for this MRN; MRN+Date join will miss.")
                if not cd_clean:
                    print("        -> clinician has blank dates for this MRN; MRN+Date join will miss.")
                shown += 1

        # match coverage
        if clin_cols:
            row_has_clin_m1 = merged_m1[clin_cols].notna().any(axis=1) if not merged_m1.empty else pd.Series([], dtype=bool)
            row_has_clin_m2 = merged_m2[clin_cols].notna().any(axis=1) if not merged_m2.empty else pd.Series([], dtype=bool)
            print("[merge] m1 (MRN-only) rows:", len(merged_m1),
                  " | with any clinician data:", int(row_has_clin_m1.sum()))
            print("[merge] m2 (MRN+Date) rows:", len(merged_m2),
                  " | with any clinician data:", int(row_has_clin_m2.sum()))

            # examples of MRNs with no match
            def examples_no_match(df):
                if df.empty: return []
                mask = ~df[clin_cols].notna().any(axis=1)
                ex = df.loc[mask, mrn_col].astype(str).head(sample).tolist()
                return ex

            ex1 = examples_no_match(merged_m1)
            ex2 = examples_no_match(merged_m2)
            if ex1:
                print(f"[merge] examples (MRN-only) with no clinician attach: {ex1}")
            if ex2:
                print(f"[merge] examples (MRN+Date) with no clinician attach: {ex2}")
        else:
            print("[merge] No clinician payload columns found to attach (only MRN/Date present).")

        print("—"*72)

    # =========================
    # OPTIONAL BLANKING (ON if requested)
    # =========================
    if apply_blanking and apply_filldown:
        raise ValueError("Choose either apply_blanking OR apply_filldown, not both.")

    if not apply_blanking and not apply_filldown:
        return merged

    # ---- duplicate blanking (after merge) ----
    # Protect: gene columns + clinician-only payload.
    # DO NOT protect group_cols -> MRN/Date may be blanked.
    def blank_duplicates_within(df: pd.DataFrame, group_cols: list) -> tuple[pd.DataFrame, int]:
        """Within each group, blank later duplicates in candidate columns.
        Keeps the first non-blank occurrence of each distinct value.
        Returns (deflated_df, num_cells_blankened)."""
        if df.empty:
            return df, 0
        out = df.copy()

        skip_cols = set(GENE_COLUMNS) | set(clin_cols) | {"_groupkey", "__source_file__"}
        cand_cols = [col for col in out.columns if col not in skip_cols]

        total_blankened = 0
        for _, g in out.groupby(group_cols, sort=False, dropna=False):
            idx = g.index
            for col in cand_cols:
                s = out.loc[idx, col]
                # non-blank mask
                nb = ~is_blankish_series(s)
                if not nb.any():
                    continue
                # normalise to find duplicates case-insensitively
                norm = s.astype(str).str.strip().str.lower()
                dup_mask = norm.duplicated(keep="first") & nb  # keep first, blank later dupes
                if dup_mask.any():
                    out.loc[idx[dup_mask], col] = pd.NA
                    total_blankened += int(dup_mask.sum())
        return out, total_blankened
    
    def fill_down_within(df: pd.DataFrame, group_cols: list, exclude_cols: list = None) -> pd.DataFrame:
        """
        Forward-fill within each group, without spilling across groups.
        Excludes any columns in exclude_cols (e.g., gene-dependent columns).
        """
        if df.empty:
            return df
        out = df.copy()
        if exclude_cols is None:
            exclude_cols = []
        # choose only columns that exist and are not excluded
        cols_to_fill = [c for c in out.columns if c not in exclude_cols]

        for _, g in out.groupby(group_cols, sort=False, dropna=False):
            idx = g.index
            # Only ffill the chosen subset (prevents touching gene block)
            filled = out.loc[idx, cols_to_fill].ffill()
            # Avoid FutureWarning: assign back, then (optionally) coerce types
            out.loc[idx, cols_to_fill] = filled.infer_objects(copy=False)
            # Optional: quiet the future downcasting change (keeps current behavior)
            # out.loc[idx, cols_to_fill] = filled.infer_objects(copy=False)

        return out
    
    # Apply per your grouping rule:
    #   - MRNs that are single-date: group on [MRN]
    #   - MRNs that are multi-date: group on [MRN, Date]
    if apply_blanking:
        # (your existing two-path blanking by groups)
        part_a, a_blanked = blank_duplicates_within(merged[~merged[mrn_col].isin(mrns_multidate)], [mrn_col])
        part_b, b_blanked = blank_duplicates_within(merged[merged[mrn_col].isin(mrns_multidate)], [mrn_col, date_col])
        merged = pd.concat([part_a, part_b], ignore_index=True, sort=False)

    if apply_filldown:
        part_a = fill_down_within(
            merged[~merged[mrn_col].isin(mrns_multidate)],
            [mrn_col],
            exclude_cols=GENE_COLUMNS
        )
        part_b = fill_down_within(
            merged[merged[mrn_col].isin(mrns_multidate)],
            [mrn_col, date_col],
            exclude_cols=GENE_COLUMNS
        )
        merged = pd.concat([part_a, part_b], ignore_index=True, sort=False)

    if verbose:
        if apply_blanking:
            print(f"[blank] total cells blanked: {a_blanked + b_blanked}  "
                f"(MRN-only: {a_blanked}, MRN+Date: {b_blanked})")
        elif apply_filldown:
            print("[filldown] group-scoped forward-fill completed (gene columns excluded).")


    return merged

# -------------------------------------------
# Column order handling
# -------------------------------------------
def read_column_order(order_path: str) -> List[str]:
    """
    Accepts an Excel (any first sheet) containing a single column with desired column names,
    or a sheet with headers where the first column lists the desired names.
    """
    xls = pd.ExcelFile(order_path)
    sheet = xls.sheet_names[0]
    df = xls.parse(sheet_name=sheet, header=None, dtype=str)
    # Flatten non-empty strings from first column
    order = [str(v).strip() for v in df.iloc[:, 0].tolist() if isinstance(v, str) and str(v).strip() != ""]
    return order

def apply_column_order(df: pd.DataFrame, order: Optional[List[str]]) -> pd.DataFrame:
    if not order:
        return df
    order_set = set(order)
    # keep listed order first, then append any columns not in the order list
    tail = [c for c in df.columns if c not in order_set]
    return df[[c for c in order if c in df.columns] + tail]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge mol-bio & clinician Excels (per date) with optional duplicate blanking or group-scoped filldown."
    )
    parser.add_argument("--molbio-dir", required=True, help="Folder with mol-bio Excel(s) (one or many; file names or sheet names act as dates).")
    parser.add_argument("--clinician-dir", required=True, help="Folder with clinician Excel(s).")
    parser.add_argument("--order", default="", help="Excel path whose first column gives desired output column order (optional).")
    parser.add_argument("--out-dir", required=True, help="Folder to write the merged Excel.")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print merge diagnostics (default: on).")

    # Mutually exclusive: either blanking or filldown or none
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply-blanking", action="store_true", help="Blank repeated values within group (opposite of filldown).")
    group.add_argument("--apply-filldown", action="store_true", help="Forward-fill within group (inverse of blanking).")

    return parser.parse_args()

# -------------------------------------------
# Main
# -------------------------------------------
def main():
    # --- Interactive inputs ---
    args = parse_args()
    molbio_dir   = args.molbio_dir
    clinician_dir= args.clinician_dir
    order_path   = args.order
    out_dir      = args.out_dir

    out_path = os.path.join(out_dir, "spss_formatted_excel.xlsx")

    # --- Load Excel files/sheets by date ---
    molbio_dict = read_excels_by_date(molbio_dir)
    clin_dict   = read_excels_by_date(clinician_dir)

    # --- Pair only common dates ---
    def safe_date_key(x):
        # Keys are like '20.04.24' (DD.MM.YY) coming from file/sheet names.
        dt = pd.to_datetime(x, format="%d.%m.%y", errors="coerce")
        return x if pd.isna(dt) else dt

    common_dates = sorted(set(molbio_dict.keys()) & set(clin_dict.keys()),
                        key=safe_date_key)

    merged_frames = []
    for d in common_dates:
        mdf = molbio_dict[d]
        cdf = clin_dict[d]

        # --- Column hygiene per pair ---
        if "MRN" not in mdf.columns and "Case_ID" in mdf.columns:
            mdf = mdf.rename(columns={"Case_ID": "MRN"})
        if "MRN" not in mdf.columns:
            raise ValueError(f"Mol-bio data for date {d} must contain 'MRN' (or 'Case_ID').")

        if "MRN" not in cdf.columns and "Case_ID" in cdf.columns:
            cdf = cdf.rename(columns={"Case_ID": "MRN"})
        if cdf.empty:
            print(f"⚠️  Clinician data for date {d} is empty; skipping this pair.")
            continue

        if "Remarks" in mdf.columns:
            mask = ~mdf["Remarks"].astype(str).str.contains("no ngs report", case=False, na=False)
            mdf = mdf[mask].copy()

        if "MRN" in mdf.columns:
            mdf["MRN"] = mdf["MRN"].ffill()

        # --- Merge this pair ---
        merged_pair = merge_long_format(
            mdf, cdf,
            verbose=args.verbose,
            apply_blanking=bool(args.apply_blanking),
            apply_filldown=bool(args.apply_filldown)
        )
        merged_pair["__merge_date__"] = d  # keep track of which date this came from
        merged_frames.append(merged_pair)

    # --- Combine all pairs ---
    if merged_frames:
        final = pd.concat(merged_frames, ignore_index=True, sort=False)
        # Sort by merge date if available
        if "__merge_date__" in final.columns:
            final = final.sort_values("__merge_date__").reset_index(drop=True)
            final.drop(columns=["__merge_date__"], inplace=True)
    else:
        final = pd.DataFrame()

    # --- Column order ---
    if order_path and os.path.isfile(order_path):
        try:
            desired = read_column_order(order_path)
        except Exception as e:
            print(f"⚠️  Could not read order file: {e}. Using default order.")
            desired = None
        final = apply_column_order(final, desired)
    else:
        # Default: put keys first if present, then the rest
        keys_first = ["MRN", "Date-Case_discussed"]
        lead = [k for k in keys_first if k in final.columns]
        rest = [c for c in final.columns if c not in lead]
        final = final[lead + rest]

    # --- Write output ---
    os.makedirs(out_dir or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        final.to_excel(writer, sheet_name="All", index=False)

    print(f"✅ Wrote: {out_path}  rows={len(final):,}  cols={final.shape[1]}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")