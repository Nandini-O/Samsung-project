"""
src/extract_enhanced_cache.py
Pre-extracts images from Enhanced Indian Food Parquet shards into an on-disk image cache
indexed by md5_hash: data/processed/images/enhanced_cache/{md5_hash}.jpg

This ensures:
1. Zero parquet reading overhead during multi-epoch PyTorch training.
2. Fast, random-access image loading directly via PIL.
3. 100% hash verification against the stratified manifest.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import hashlib
import time
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/enhanced-indian-food-classification")
CACHE_DIR = Path("data/processed/images/enhanced_cache")
MANIFEST_PATH = Path("data/processed/images/enhanced_indian_food_stratified_manifest.csv")

def extract_parquet_images():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 65)
    print("EXTRACTING ENHANCED INDIAN FOOD IMAGES TO DISK CACHE")
    print(f"Target Directory: {CACHE_DIR.resolve()}")
    print("=" * 65)

    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet shards:")
    for pf in parquet_files:
        print(f"  - {pf.name} ({pf.stat().st_size / (1024**2):.1f} MB)")

    total_extracted = 0
    total_skipped = 0
    t0 = time.time()

    for pf in parquet_files:
        print(f"\nProcessing {pf.name}...")
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            img_bytes = row["image"]["bytes"]
            h = hashlib.md5(img_bytes).hexdigest()
            out_file = CACHE_DIR / f"{h}.jpg"
            if not out_file.exists():
                out_file.write_bytes(img_bytes)
                total_extracted += 1
            else:
                total_skipped += 1

    dt = time.time() - t0
    cached_count = len(list(CACHE_DIR.glob("*.jpg")))
    print("\n" + "=" * 65)
    print(f"Extraction Completed in {dt:.2f}s")
    print(f"Newly written images: {total_extracted:,}")
    print(f"Already cached / duplicate: {total_skipped:,}")
    print(f"Total unique cached images in {CACHE_DIR}: {cached_count:,}")
    print("=" * 65)

    # Verify against manifest
    if MANIFEST_PATH.exists():
        manifest_df = pd.read_csv(MANIFEST_PATH)
        missing_count = 0
        for h in manifest_df["md5_hash"]:
            if not (CACHE_DIR / f"{h}.jpg").exists():
                missing_count += 1
        print(f"Manifest Verification: {len(manifest_df):,} total rows checked.")
        print(f"Missing images: {missing_count} (Must be 0)")
        assert missing_count == 0, f"Error: {missing_count} manifest images missing from cache!"
        print("ALL MANIFEST IMAGES VERIFIED IN DISK CACHE!")

if __name__ == "__main__":
    extract_parquet_images()
