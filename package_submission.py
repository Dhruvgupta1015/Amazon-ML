#!/usr/bin/env python3
"""
package_submission.py — Creates the official Amazon ML Challenge 2026 submission zip.

Structure enforced:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── preprocessor.py
│       │   ├── blocking.py
│       │   ├── feature_extractor.py
│       │   ├── model.py
│       │   ├── metrics.py
│       │   ├── validator.py
│       │   ├── pipeline_runner.py
│       │   └── __init__.py
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Package official Amazon ML 2026 submission zip")
    parser.add_argument("--team-name", default="Resolve_AI_Team", help="Team name for zip filename")
    parser.add_argument("--output-zip", default=None, help="Custom output zip path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    team_name = args.team_name.strip().replace(" ", "_")
    zip_path = Path(args.output_zip) if args.output_zip else root / f"{team_name}_submission.zip"

    mr_path = root / "output" / "matching_results.tsv"
    cp_path = root / "output" / "candidate_pairs.tsv"
    code_dir = root / "code" / "business_entity_resolution"
    doc_path = root / "Documentation_template.md"

    if not mr_path.exists():
        print(f"Error: {mr_path} does not exist. Run pipeline first.")
        sys.exit(1)
    if not cp_path.exists():
        gz_path = root / "output" / "candidate_pairs.tsv.gz"
        if gz_path.exists():
            print(f"Decompressing {gz_path.name} -> {cp_path.name}...")
            import gzip, shutil
            with gzip.open(gz_path, "rb") as f_in, open(cp_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            print(f"Error: {cp_path} does not exist.")
            sys.exit(1)
    if not doc_path.exists():
        print(f"Error: {doc_path} does not exist.")
        sys.exit(1)

    print(f"Packaging {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output folder
        zf.write(mr_path, arcname="output/matching_results.tsv")
        zf.write(cp_path, arcname="output/candidate_pairs.tsv")
        print("  [OK] output/matching_results.tsv")
        print("  [OK] output/candidate_pairs.tsv")

        # 2. Code folder
        src_dir = code_dir / "src"
        for py_file in src_dir.glob("*.py"):
            arcname = f"code/business_entity_resolution/src/{py_file.name}"
            zf.write(py_file, arcname=arcname)
            print(f"  [OK] {arcname}")

        readme_file = code_dir / "README.md"
        if readme_file.exists():
            zf.write(readme_file, arcname="code/business_entity_resolution/README.md")
            print("  [OK] code/business_entity_resolution/README.md")

        req_file = code_dir / "requirements.txt"
        if req_file.exists():
            zf.write(req_file, arcname="code/business_entity_resolution/requirements.txt")
            print("  [OK] code/business_entity_resolution/requirements.txt")

        # 3. Documentation template
        zf.write(doc_path, arcname="Documentation_template.md")
        print("  [OK] Documentation_template.md")

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully created {zip_path.name} ({size_mb:.2f} MB)")
    print("All Amazon ML Challenge 2026 submission packaging rules satisfied!")

if __name__ == "__main__":
    main()
