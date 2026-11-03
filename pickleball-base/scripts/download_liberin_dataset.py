#!/usr/bin/env python3
"""
download_liberin_dataset.py - Downloads and extracts Roboflow dataset 'liberin-technologies/pickleball-vision/9'
using the verified Roboflow API key.
"""
import json
import os
import sys
import urllib.request
import zipfile

API_KEY = "tuNmM2e2UyIYseofKIbN"
PROJECT_ID = "liberin-technologies/pickleball-vision"
VERSION = "9"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST_DIR = os.path.join(ROOT_DIR, "datasets", "raw", "liberin")
ZIP_DEST = os.path.join(ROOT_DIR, "datasets", "raw", "liberin_v9_yolo26.zip")


def main():
    print("=" * 80)
    print(" 📥 DOWNLOADING ROBOFLOW DATASET: liberin-technologies/pickleball-vision/9 (YOLO26)")
    print("=" * 80)
    print(f"  • API Key            : {API_KEY[:4]}...{API_KEY[-4:]}")
    print(f"  • Destination Dir    : {DEST_DIR}")
    print("=" * 80)

    # 1. Query Roboflow API for the YOLO26 export URL
    export_url = f"https://api.roboflow.com/{PROJECT_ID}/{VERSION}/yolo26?api_key={API_KEY}"
    req = urllib.request.Request(export_url, headers={"User-Agent": "Mozilla/5.0"})

    print("[1/3] Querying Roboflow export endpoint...")
    with urllib.request.urlopen(req) as resp:
        meta = json.loads(resp.read().decode())

    export_info = meta.get("export", {})
    download_link = export_info.get("link")
    archive_size_mb = export_info.get("size", 0.0)

    if not download_link:
        print("Error: Could not retrieve download link. Full response:", meta)
        sys.exit(1)

    print(f"  • Found YOLO26 Archive : {archive_size_mb:.1f} MB")
    print(f"  • Download Link        : {download_link[:40]}...")

    # 2. Download the zip archive
    os.makedirs(os.path.dirname(ZIP_DEST), exist_ok=True)
    print(f"\n[2/3] Downloading archive to {ZIP_DEST}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100.0)
            mb = downloaded / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Downloaded: {mb:.1f} MB / {tot_mb:.1f} MB ({pct:.1f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(download_link, ZIP_DEST, reporthook=reporthook)
    print("\n  • Download complete!")

    # 3. Extract to datasets/raw/liberin/
    print(f"\n[3/3] Extracting archive to {DEST_DIR}...")
    os.makedirs(DEST_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_DEST, "r") as zf:
        zf.extractall(DEST_DIR)

    # Count extracted files
    extracted_imgs = 0
    for root, _, files in os.walk(DEST_DIR):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                extracted_imgs += 1

    print(f"  • Successfully extracted {extracted_imgs} images into {DEST_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
