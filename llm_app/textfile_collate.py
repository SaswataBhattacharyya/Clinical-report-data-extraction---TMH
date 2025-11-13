import os
import re

# INPUT pattern examples:
#   Some Report_aligned_spaced_page_1.txt
#   Some Report_aligned_grid_page_2.txt
PAGE_RE = re.compile(r"^(.*)_aligned_(spaced|grid)_page_([0-9]+)\.txt$", re.IGNORECASE)

# Controls the word used in the compiled filenames: *_alligned_spaced.txt
OUTPUT_TAG = "alligned"   # change to "aligned" if you prefer

def collate_alignments(input_root: str, output_root: str):
    """
    Walk input_root, mirror its folder tree in output_root, and for each report base:
      - write {base}_{OUTPUT_TAG}_spaced.txt    (collated spaced pages)
      - write {base}_{OUTPUT_TAG}_grid.txt      (collated grid pages)
    Each page block is appended as:
        <page text>\n\nPage No.: {n}\n\n
    """
    for root, _, files in os.walk(input_root):
        # mirror folder structure
        rel_path = os.path.relpath(root, input_root)
        out_dir = os.path.join(output_root, rel_path)
        os.makedirs(out_dir, exist_ok=True)

        # base -> {'spaced': [(page_no, fullpath), ...], 'grid': [...]}
        buckets = {}

        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            m = PAGE_RE.match(fname)
            if not m:
                continue  # not a per-page aligned file

            base = m.group(1).strip()
            style = m.group(2).lower()      # 'spaced' or 'grid'
            page_no = int(m.group(3))

            buckets.setdefault(base, {"spaced": [], "grid": []})
            buckets[base][style].append((page_no, os.path.join(root, fname)))

        # write collated outputs per base & style
        for base, styles in buckets.items():
            for style_name, page_list in styles.items():
                if not page_list:
                    continue
                page_list.sort(key=lambda t: t[0])  # by page number

                out_name = f"{base}_{OUTPUT_TAG}_{style_name}.txt"
                out_path = os.path.join(out_dir, out_name)

                chunks = []
                for page_no, fpath in page_list:
                    with open(fpath, "r", encoding="utf-8") as f:
                        page_text = f.read().rstrip()
                    # <content>\n\nPage No.: n\n\n
                    chunks.append(page_text + "\n\n" + f"Page No.: " + str(page_no) + "\n\n")

                with open(out_path, "w", encoding="utf-8") as out:
                    out.write("".join(chunks))

                print(f"✅ Wrote {out_path}")

def main(input_root=None, output_root=None):
    """Main function for command-line usage or integration"""
    if input_root is None:
        input_root = input("Enter path to INPUT root (per-page aligned texts): ").strip()
    if output_root is None:
        output_root = input("Enter path to OUTPUT root (collated texts): ").strip()
    
    try:
        if not os.path.isdir(input_root):
            return {"success": False, "message": f"Input root does not exist or is not a directory: {input_root}"}
        os.makedirs(output_root, exist_ok=True)
        
        collate_alignments(input_root, output_root)
        return {"success": True, "message": f"Successfully collated text files from {input_root} to {output_root}"}
    except Exception as e:
        return {"success": False, "message": f"Error during text collation: {str(e)}"}

if __name__ == "__main__":
    result = main()
    if result:
        print(result["message"])
