import os
import re  # <<< changed
import csv
import pandas as pd  # <<< changed
from tqdm import tqdm
from openai import OpenAI

# ---------- Config ----------
# INPUT/OUTPUT now taken from user at runtime (not hardcoded)  # <<< changed
# (Removed hardcoded OPENAI_API_KEY; rely on env var only)      # <<< changed

# Initialize client - will be set by main function
client = None

def init_client(api_key=None):
    """Initialize OpenAI client with API key"""
    global client
    if api_key:
        client = OpenAI(api_key=api_key)
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            raise ValueError("OpenAI API key not provided")
    return client

# ---------- Prompts ----------
LIST_EXTRACTION_PROMPT = """You are processing a patient list document.

Your ONLY job is to extract **patient name** and **case ID** in order of appearance.

Start numbering from 1. Use the format:

1 John Doe ABC123
2 Jane Smith XYZ456
...

If name or ID is missing, write [UNCLEAR].

Return only the list — no extra explanation or formatting.
"""

# build_structured_prompt now accepts BOTH texts (grid + spaced) and includes the expanded schema  # <<< changed
def build_structured_prompt(grid_text, spaced_text, discussion_date):  # <<< changed
    return f"""You are a medical report parser.

You must extract information **per genetic perturbation**, listing all shared patient-level details for each mutation separately.

For Specimen parameter-
Specimen: <choose one of [Plasma, PF & PF cell pellet / Plasma and Pleural / Plasma / Tissue and Plasma / Tissue / Pleural / Tissue and Pleural / NA]>

- If ctDNA, Liquid, blood or plasma is mentioned → use "Plasma".
- If paraffin block, FFPE, or Tissue is mentioned → use "Tissue".
- If Pleural fluid (PF) is mentioned → use "Pleural".
- If both tissue and pleural are reported → use "Tissue and Pleural".
- If both tissue and plasma are reported → use "Tissue and Plasma".
- If PF cell pellet is reported → use "PF cell pellet".
- If blood + pleural are reported → use "Plasma and Pleural".
- If CSF is mentioned → use "CSF".
- If nothing matches → use "NA".

Use BOTH sources below (GRID and SPACED). When they disagree, prefer the more complete/consistent value:
--- GRID TEXT START ---
{grid_text}
--- GRID TEXT END ---
--- SPACED TEXT START ---
{spaced_text}
--- SPACED TEXT END ---

Return STRICTLY in this format (one block per gene/perturbation):

---
Patient_name: <name>
Case_ID: <id>
Specimen: <choose one of [Plasma, PF & PF cell pellet / Plasma and Pleural / Plasma / Tissue and Plasma / Tissue / Pleural / Tissue and Pleural / CSF / NA]>
Date-reporting_date: <YYYY-MM-DD or NA>
Year_of_NGS: <year or NA>
NGS_Tech_used: <Illumina / Thermofisher / Pac Bio / Nanopore / NA>

TMB (tissue): <High / Low / Intermediate / NA>
TMB (liq): <High / Low / Intermediate / NA>
TMB (pf): <High / Low / Intermediate / NA>
TMB (csf)": "<High / Low / Intermediate / NA>
TMB (others)": "<High / Low / Intermediate / NA>
MSI (tissue): <High / low / stable / NA>
MSI (liq): <High / low / stable / NA>
MSI (pf): <High / low / stable / NA>
MSI (csf)": "<High / low / stable / NA>
MSI (others)": "<High / low / stable / NA>
Telomeric_Allelic_Imbalance (tissue): <Yes / No / NA>
Telomeric_Allelic_Imbalance (liq): <Yes / No / NA>
Telomeric_Allelic_Imbalance (pf): <Yes / No / NA>
Telomeric_Allelic_Imbalance (csf)": "<Yes | No | Not Done | NA>
Telomeric_Allelic_Imbalance (others)": "<Yes | No | Not Done | NA>
Large_Scale_State_Transition (tissue): <Yes / No / NA>
Large_Scale_State_Transition (liq): <Yes / No / NA>
Large_Scale_State_Transition (pf): <Yes / No / NA>
Large_Scale_State_Transition (csf)": "<Yes | No | NA>
Large_Scale_State_Transition (others)": "<Yes | No | NA>
LOH_Score (tissue): <Low / High / NA>
LOH_Score (liq): <Low / High / NA>
LOH_Score (pf): <Low / High / NA>
LOH_Score (csf)": "<Low / High / NA>
LOH_Score (others)": "<Low / High / NA>
HRD_Score (tissue): <Low / High / NA>
HRD_Score (liq): <Low / High / NA>
HRD_Score (pf): <Low / High / NA>
HRD_Score (csf)": "<Low / High / NA>
HRD_Score (others)": "<Low / High / NA>
Tumor_Fraction: <Elevated / Not elevated / NA>
Gene_Tier_4_Name: <name or NA>
Depth_of_Coverage: <value or NA>
RNA_Passing_Filter_Reads: <value or NA>

Gene_name: <name (for fusion - 'Gene A - Gene B')>
Mutation(nucleotide): <mut(n) or NA>
Mutation(protein): <mut(p) or NA>
VAF (tissue): <value or NA>
VAF (liq): <value or NA>
VAF (pf): <value or NA>
VAF (csf)": "<value or NA>
VAF (others)": "<value or NA>
Exon_no: <value or NA>
NM ID: <value or NA>  # for fusion variants, include both partners' NM IDs; else NA
Fusion: <partner gene or NA>
Fusion_Read_Count (tissue): <count or NA>
Fusion_Read_Count (liq): <count or NA>
Fusion_Read_Count (pf): <count or NA>
Fusion_Read_Count (csf)": "<count or NA>
Fusion_Read_Count (others)": "<count or NA>
Amplification: <yes/no>
Amp_read/copy number (tissue)": "<value or NA>
Amp_read/copy number (liq)": "<value or NA>
Amp_read/copy number (pf)": "<value or NA>
Amp_read/copy number (csf)": "<value or NA>
Amp_read/copy number (others)": "<value or NA>
Variant Type: <e.g., SNV / Fusion / INDEL / Deletion / Amplification>
Tier: <Tier level (I/II/III/IV)>

Reporting_company: <name or NA>
Date-Case_discussed: {discussion_date}
---

Return ONLY such blocks, nothing else.
"""

def call_gpt_text(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"GPT call failed: {e}")
        return ""

def parse_list_output(text):
    parsed = []
    for idx, line in enumerate(text.strip().split("\n")):  # <<< changed (track index)
        if line.strip():
            parts = line.strip().split(None, 2)
            if len(parts) == 3:
                serial, name, case_id = parts
            else:
                serial = parts[0] if len(parts) > 0 else ""
                name = parts[1] if len(parts) > 1 else ""
                case_id = parts[2] if len(parts) > 2 else ""
            parsed.append({
                "Serial No.": serial,
                "Patient Name": name,
                "Case ID": case_id,
                "_infile_idx": idx,  # <<< changed
            })
    return parsed

# regex to group the two collated inputs per report  # <<< changed
PAIR_RE = re.compile(r"^(.*)_allign?ed_(spaced|grid)\.txt$", re.IGNORECASE)  # accepts 'alligned' or 'aligned'

def main(input_dir=None, output_dir=None, api_key=None):
    # --- get input/output roots from user or parameters ---
    if input_dir is None:
        input_dir = input("Enter INPUT root folder (collated aligned texts): ").strip()
    if output_dir is None:
        output_dir = input("Enter OUTPUT root folder (GPT outputs): ").strip()
    
    # Initialize client with API key
    if api_key:
        init_client(api_key)
    else:
        try:
            init_client()
        except ValueError as e:
            return {"success": False, "message": str(e)}
    
    if not os.path.isdir(input_dir):
        return {"success": False, "message": f"INPUT_DIR does not exist: {input_dir}"}
    os.makedirs(output_dir, exist_ok=True)
    
    INPUT_DIR = input_dir
    OUTPUT_DIR = output_dir

    try:
        for root, _, files in os.walk(INPUT_DIR):
            rel_root = os.path.relpath(root, INPUT_DIR)
            discussion_date = os.path.basename(os.path.normpath(rel_root))
            output_subdir = os.path.join(OUTPUT_DIR, rel_root)
            os.makedirs(output_subdir, exist_ok=True)

            # --- gather list docs (black/red/blue) & paired report files ---  # <<< changed
            list_records = []
            list_xlsx_path = os.path.join(output_subdir, "list_output.xlsx")  # <<< changed

            # group aligned pairs by base
            pairs = {}  # base -> {"spaced": path?, "grid": path?}  # <<< changed

            for file in files:
                if not file.lower().endswith(".txt"):
                    continue
                low = file.lower()
                # list extraction files (keep existing behavior)
                if any(k in low for k in ["black_list", "red_list", "blue_list"]):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text_input = f.read()
                    prompt = LIST_EXTRACTION_PROMPT + "\n\n" + text_input
                    output = call_gpt_text(prompt)
                    parsed_list = parse_list_output(output)
                    # attach color & filename for ordering  # <<< changed
                    color = "black" if "black_list" in low else ("red" if "red_list" in low else "blue")
                    for entry in parsed_list:
                        entry["Filename"] = file
                        entry["Date-Case_discussed"] = discussion_date
                        entry["_color"] = color
                        list_records.append(entry)
                    continue

                # otherwise, try to match paired aligned inputs  # <<< changed
                m = PAIR_RE.match(file)
                if m:
                    base = m.group(1)
                    style = m.group(2).lower()
                    pairs.setdefault(base, {"spaced": None, "grid": None})
                    pairs[base][style] = os.path.join(root, file)

            # --- process each report pair once (two inputs -> one GPT output) ---  # <<< changed
            for base, d in tqdm(pairs.items(), desc=f"Processing reports in: {rel_root}"):
                out_report_txt = os.path.join(output_subdir, f"{base}_gpt.txt")
                if os.path.exists(out_report_txt):
                    continue  # skip if already parsed

                spaced_path = d.get("spaced")
                grid_path = d.get("grid")

                # read whichever is available
                spaced_text = ""
                grid_text = ""
                if spaced_path and os.path.exists(spaced_path):
                    with open(spaced_path, "r", encoding="utf-8") as f:
                        spaced_text = f.read()
                if grid_path and os.path.exists(grid_path):
                    with open(grid_path, "r", encoding="utf-8") as f:
                        grid_text = f.read()
                if not spaced_text and not grid_text:
                    # nothing to parse for this base
                    continue

                prompt = build_structured_prompt(grid_text=grid_text, spaced_text=spaced_text,
                                                 discussion_date=discussion_date)  # <<< changed
                output = call_gpt_text(prompt)

                with open(out_report_txt, "w", encoding="utf-8") as out:
                    out.write(output)

            # --- write list output as Excel, ordered Black -> Red -> Blue, preserving in-file order ---  # <<< changed
            if list_records:
                color_rank = {"black": 0, "red": 1, "blue": 2}
                # ensure defaults if missing
                for e in list_records:
                    e["_color"] = e.get("_color", "black")
                    e["_infile_idx"] = e.get("_infile_idx", 0)
                ordered = sorted(list_records, key=lambda r: (color_rank.get(r["_color"], 99), r["_infile_idx"]))
                cols = ["Filename", "Date-Case_discussed", "Serial No.", "Patient Name", "Case ID"]
                df = pd.DataFrame(ordered)[cols]
                df.to_excel(list_xlsx_path, index=False)
        
        print("\n🎉 All folders processed. GPT outputs written; list Excel files saved.")
        return {"success": True, "message": f"Successfully processed GPT parsing from {INPUT_DIR}"}
    except Exception as e:
        return {"success": False, "message": f"Error during GPT parsing: {str(e)}"}

if __name__ == "__main__":
    result = main()
    if result:
        print(result["message"])
