"""
src/dataset_prep_vision.py
Phase 4: Feature Engineering - Vision Dataset Preparation

Prepares reproducible, leakage-free vision training, validation, and test splits:
1. Loads the stratified Enhanced Indian Food manifest (15,404 images across 43 classes).
2. Loads clean FoodLens manifest (2,521 active images, excluding the 2 leakage duplicates).
3. Generates class-to-index mapping (models/class_to_idx.json).
4. Computes inverse-frequency class weights (models/class_weights.json) to handle class imbalance
   (such as Samosa representing 42.4% in FoodLens and Chole Bhature 6.2:1 imbalance in Enhanced).
5. Exports ready-to-train split manifests:
   - data/processed/vision_splits/train_split.csv
   - data/processed/vision_splits/val_split.csv
   - data/processed/vision_splits/test_split.csv
   - data/processed/vision_splits/combined_vision_manifest.csv
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROC_DIR = Path("data/processed")
VISION_DIR = PROC_DIR / "vision_splits"
MODELS_DIR = Path("models")

VISION_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_md5(file_path):
    p = Path(file_path)
    if not p.exists():
        return ""
    return hashlib.md5(p.read_bytes()).hexdigest()


def prepare_vision_datasets():
    print("=" * 65)
    print("PREPARING VISION DATASETS & STRATIFIED SPLITS")
    print("=" * 65)

    enh_manifest_path = PROC_DIR / "images" / "enhanced_indian_food_stratified_manifest.csv"
    fl_manifest_path = PROC_DIR / "images" / "food_lens_manifest.csv"

    if not enh_manifest_path.exists():
        raise FileNotFoundError(f"Missing {enh_manifest_path}. Run image_split_analysis_and_plan.py first.")

    df_enh = pd.read_csv(enh_manifest_path)
    df_fl = pd.read_csv(fl_manifest_path)

    print(f"Loaded Enhanced Indian Food: {len(df_enh):,} images")
    print(f"Loaded FoodLens: {len(df_fl):,} images (total)")

    # 1. Filter FoodLens leakage
    df_fl_active = df_fl[~df_fl["exclude_leakage"]].copy()
    print(f"Active FoodLens (excluding 2 leakage dups): {len(df_fl_active):,} images")

    # Compute genuine MD5 hashes for active FoodLens images
    fl_md5_hashes = [compute_file_md5(p) for p in df_fl_active["path"]]

    # Standardize Enhanced records
    df_enh_clean = pd.DataFrame({
        "image_identifier": df_enh["raw_file"] + ":" + df_enh["md5_hash"].str[:8],
        "image_path": df_enh["raw_file"],
        "class_name": df_enh["label"],
        "split": df_enh["assigned_split"],
        "dataset_source": "enhanced_indian_food",
        "md5_hash": df_enh["md5_hash"]
    })

    # Standardize FoodLens records
    df_fl_clean = pd.DataFrame({
        "image_identifier": df_fl_active["filename"],
        "image_path": df_fl_active["path"],
        "class_name": df_fl_active["class"],
        "split": df_fl_active["split"],
        "dataset_source": "food_lens",
        "md5_hash": fl_md5_hashes
    })

    # 2. Build Enhanced 43-Class Label Dictionary & Weights
    classes = sorted(df_enh_clean["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}

    df_enh_clean["class_idx"] = df_enh_clean["class_name"].map(class_to_idx)

    # Class Weights for WeightedCrossEntropyLoss
    # Formula: w_c = N / (K * N_c)
    n_total = len(df_enh_clean)
    n_classes = len(classes)
    counts = df_enh_clean["class_name"].value_counts()
    
    class_weights = {}
    for c in classes:
        nc = counts[c]
        weight = round(float(n_total / (n_classes * nc)), 4)
        class_weights[c] = {
            "class_idx": class_to_idx[c],
            "count": int(nc),
            "weight": weight
        }

    # Save Class Metadata
    class_idx_path = MODELS_DIR / "class_to_idx.json"
    class_weights_path = MODELS_DIR / "class_weights.json"

    with open(class_idx_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    with open(class_weights_path, "w", encoding="utf-8") as f:
        json.dump(class_weights, f, indent=2)

    print(f"\nClass Metadata Saved:")
    print(f"  - Classes: {n_classes} distinct categories")
    print(f"  - Label Mapping: {class_idx_path}")
    print(f"  - Class Weights: {class_weights_path}")
    print(f"  - Weight range: min={min(v['weight'] for v in class_weights.values()):.2f} (chole_bhature) to "
          f"max={max(v['weight'] for v in class_weights.values()):.2f} (paani_puri)")

    # 3. Export Vision Splits for Enhanced Dataset (Primary 43-Class Benchmark)
    train_df = df_enh_clean[df_enh_clean["split"] == "train"]
    val_df = df_enh_clean[df_enh_clean["split"] == "validation"]
    test_df = df_enh_clean[df_enh_clean["split"] == "test"]

    print(f"\nPrimary Enhanced 43-Class Vision Splits:")
    print(f"  Train:      {len(train_df):,} images ({len(train_df)/n_total*100:.1f}%) | {train_df['class_name'].nunique()} classes")
    print(f"  Validation: {len(val_df):,} images ({len(val_df)/n_total*100:.1f}%) | {val_df['class_name'].nunique()} classes")
    print(f"  Test:       {len(test_df):,} images ({len(test_df)/n_total*100:.1f}%) | {test_df['class_name'].nunique()} classes")

    # Verification: 0% cross-split leakage in Enhanced
    train_hashes = set(train_df["md5_hash"])
    val_hashes = set(val_df["md5_hash"])
    test_hashes = set(test_df["md5_hash"])

    leakage_train_val = len(train_hashes.intersection(val_hashes))
    leakage_train_test = len(train_hashes.intersection(test_hashes))
    leakage_val_test = len(val_hashes.intersection(test_hashes))

    print(f"\nCross-Split Leakage Check in Primary Enhanced Splits:")
    print(f"  Train <-> Val:  {leakage_train_val} shared hashes")
    print(f"  Train <-> Test: {leakage_train_test} shared hashes")
    print(f"  Val <-> Test:   {leakage_val_test} shared hashes")

    if leakage_train_val + leakage_train_test + leakage_val_test > 0:
        raise ValueError("CRITICAL ERROR: Duplicate image leakage detected across Enhanced vision splits!")

    # 4. Process & Verify FoodLens Splits (Secondary / Auxiliary Benchmark)
    fl_train_df = df_fl_clean[df_fl_clean["split"] == "train"]
    fl_val_df = df_fl_clean[df_fl_clean["split"] == "val"]
    fl_test_df = df_fl_clean[df_fl_clean["split"] == "test"]

    fl_train_hashes = set(fl_train_df["md5_hash"])
    fl_val_hashes = set(fl_val_df["md5_hash"])
    fl_test_hashes = set(fl_test_df["md5_hash"])

    fl_leak_tr_val = len(fl_train_hashes.intersection(fl_val_hashes))
    fl_leak_tr_test = len(fl_train_hashes.intersection(fl_test_hashes))
    fl_leak_val_test = len(fl_val_hashes.intersection(fl_test_hashes))

    print(f"\nCross-Split Leakage Check in Active FoodLens Splits (2,521 images):")
    print(f"  Train <-> Val:  {fl_leak_tr_val} shared hashes")
    print(f"  Train <-> Test: {fl_leak_tr_test} shared hashes")
    print(f"  Val <-> Test:   {fl_leak_val_test} shared hashes")

    if fl_leak_tr_val + fl_leak_tr_test + fl_leak_val_test > 0:
        raise ValueError("CRITICAL ERROR: Duplicate image leakage detected across FoodLens vision splits!")

    # Cross-dataset leakage
    cross_dataset_overlap = len(set(df_enh_clean["md5_hash"]).intersection(set(df_fl_clean["md5_hash"])))
    print(f"\nCross-Dataset Leakage (Enhanced vs FoodLens): {cross_dataset_overlap} shared hashes")
    if cross_dataset_overlap > 0:
        raise ValueError("CRITICAL ERROR: Hash collision between Enhanced and FoodLens datasets!")

    # 5. Export Splits
    # A. Primary 43-Class Splits
    train_path = VISION_DIR / "train_split.csv"
    val_path = VISION_DIR / "val_split.csv"
    test_path = VISION_DIR / "test_split.csv"
    enh_train_path = VISION_DIR / "enhanced_train_split.csv"
    enh_val_path = VISION_DIR / "enhanced_val_split.csv"
    enh_test_path = VISION_DIR / "enhanced_test_split.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    train_df.to_csv(enh_train_path, index=False)
    val_df.to_csv(enh_val_path, index=False)
    test_df.to_csv(enh_test_path, index=False)

    # B. Auxiliary FoodLens Splits
    fl_train_path = VISION_DIR / "food_lens_train_split.csv"
    fl_val_path = VISION_DIR / "food_lens_val_split.csv"
    fl_test_path = VISION_DIR / "food_lens_test_split.csv"

    fl_train_df.to_csv(fl_train_path, index=False)
    fl_val_df.to_csv(fl_val_path, index=False)
    fl_test_df.to_csv(fl_test_path, index=False)

    # C. Combined Manifest with Explicit Benchmark Roles
    df_enh_clean["benchmark_role"] = "primary_43class_benchmark"
    df_fl_clean["benchmark_role"] = "auxiliary_transfer_foodlens"

    df_combined = pd.concat([df_enh_clean, df_fl_clean], ignore_index=True)
    combined_path = VISION_DIR / "combined_vision_manifest.csv"
    df_combined.to_csv(combined_path, index=False)

    print(f"\nVision Manifests Exported:")
    print(f"  - Primary Train Split:      {train_path} ({len(train_df):,} images)")
    print(f"  - Primary Val Split:        {val_path} ({len(val_df):,} images)")
    print(f"  - Primary Test Split:       {test_path} ({len(test_df):,} images)")
    print(f"  - FoodLens Train Split:     {fl_train_path} ({len(fl_train_df):,} images)")
    print(f"  - FoodLens Val Split:       {fl_val_path} ({len(fl_val_df):,} images)")
    print(f"  - FoodLens Test Split:      {fl_test_path} ({len(fl_test_df):,} images)")
    print(f"  - Combined Total Manifest:  {combined_path} ({len(df_combined):,} images)")
    print("=" * 65)
    return df_combined


if __name__ == "__main__":
    prepare_vision_datasets()

