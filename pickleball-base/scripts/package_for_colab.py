#!/usr/bin/env python3
"""
package_for_colab.py - Packages the clean ball dataset into a zip archive for Google Colab NVIDIA A100 GPU training.
"""
import os
import sys
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(ROOT, "datasets", "ball")
ZIP_OUT = os.path.join(ROOT, "datasets", "ball_clean_dataset.zip")


def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} does not exist. Run pbbase.data.build_datasets first.")
        sys.exit(1)

    print("=" * 80)
    print(" 📦 PACKAGING MERGED BALL DATASET FOR GOOGLE COLAB (A100 GPU)")
    print("=" * 80)
    print(f"  • Source Directory : {DATASET_DIR}")
    print(f"  • Target Archive   : {ZIP_OUT}")

    total_files = 0
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # Include data.yaml configured with relative paths
        yaml_content = "path: ./ball_clean\ntrain: train/images\nval: val/images\ntest: test/images\n\nnc: 1\nnames: ['ball']\n"
        zf.writestr("ball_clean/data.yaml", yaml_content)

        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                folder = os.path.join(DATASET_DIR, split, sub)
                if not os.path.exists(folder):
                    continue
                for fname in sorted(os.listdir(folder)):
                    fpath = os.path.join(folder, fname)
                    # Resolve symlinks so real image bytes are archived
                    real_path = os.path.realpath(fpath)
                    if os.path.isfile(real_path):
                        arcname = os.path.join("ball_clean", split, sub, fname)
                        zf.write(real_path, arcname)
                        total_files += 1
                        if total_files % 1000 == 0:
                            print(f"    Archived {total_files} files...")

    size_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
    print(f"\n[DONE] Successfully created {ZIP_OUT} ({size_mb:.1f} MB, {total_files} files).")
    print("Upload this zip directly to Google Colab or Google Drive to train on T4 GPU.")
    print("=" * 80)


if __name__ == "__main__":
    main()
