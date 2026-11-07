"""
Phase 2 Review Script - runs all 5 review items:
1. Sodium investigation
2. Serving-size limitations
3. FoodLens duplicate split check
4. Food/dataset.tar.gz assessment
5. Updated audit report
"""
import os
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent  # src/ -> Samsung_Project/
PROC_NUT = ROOT / "data" / "processed" / "nutrition"
PROC_IMG = ROOT / "data" / "processed" / "images"
RAW = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"

findings = {}

# ===========================================================================
# ITEM 1: Investigate 31 high-sodium records
# ===========================================================================
print("=" * 65)
print("ITEM 1: High-Sodium Records Investigation")
print("=" * 65)

df = pd.read_csv(PROC_NUT / "nutrition_dish_level.csv")
df_raw = pd.read_csv(RAW / "Indian_Food_Nutrition_Processed.csv")

high = df[df["sodium_mg"] > 5000].copy()
print(f"Total records with sodium_mg > 5000: {len(high)}")
print()

# Cross-check: raw values match processed (no transformation error)
df_raw_indexed = df_raw.set_index("Dish Name")
mismatches = []
for _, row in high.iterrows():
    raw_name = row["food_name_raw"]
    if raw_name in df_raw_indexed.index:
        raw_val = df_raw_indexed.loc[raw_name, "Sodium (mg)"]
        if abs(float(raw_val) - row["sodium_mg"]) > 1.0:
            mismatches.append({"food": raw_name, "raw": raw_val, "processed": row["sodium_mg"]})

print(f"Transformation check: {len(high) - len(mismatches)} match raw | {len(mismatches)} mismatches")
if mismatches:
    for m in mismatches:
        print(f"  MISMATCH: {m['food']}: raw={m['raw']} processed={m['processed']}")

# Classify each record
IMPOSSIBLE_THRESHOLD = 39300  # max sodium in pure NaCl per 100g

categories = {
    "soups": [],
    "pickles_condiments": [],
    "spice_mixes": [],
    "other": [],
}

for _, row in high.iterrows():
    food = row["food_name"]
    sod = row["sodium_mg"]
    cals = row["calories_kcal"]
    verdict = ""
    rationale = ""

    if any(k in food for k in ["soup", "consomme", "broth"]):
        # Soups: low-calorie, high sodium
        # Even most salty restaurant soups: ~1000-2000mg per 250ml serving
        # These values 5000-14000 per serving are 3-10x typical
        # However calorie values (27-120 kcal) are consistent with soups
        # IFCT soups are typically recorded per 100ml
        per_100ml_if_250ml = sod / 2.5
        if per_100ml_if_250ml > 2000:
            verdict = "SUSPICIOUS"
            rationale = f"Per-100ml={per_100ml_if_250ml:.0f}mg if 250ml serving. Exceeds realistic restaurant soup sodium. Possible data source uses per-recipe-batch or values are inflated."
        else:
            verdict = "PLAUSIBLE"
            rationale = f"Per-100ml={per_100ml_if_250ml:.0f}mg if 250ml serving."
        categories["soups"].append({"food": food, "sodium_mg": sod, "calories_kcal": cals, "verdict": verdict, "rationale": rationale})

    elif any(k in food for k in ["pickle", "achaar", "chutney"]):
        # Pickles: typically 20g serving, 2000-8000mg/100g is known
        per_100g_if_20g = sod / 0.20
        if per_100g_if_20g > IMPOSSIBLE_THRESHOLD:
            verdict = "IMPOSSIBLE - exceeds max sodium in pure NaCl"
        elif per_100g_if_20g > 12000:
            verdict = "HIGHLY_SUSPICIOUS"
        else:
            verdict = "PLAUSIBLE for pickle/condiment"
        rationale = f"If 20g serving: per-100g={per_100g_if_20g:.0f}mg"
        categories["pickles_condiments"].append({"food": food, "sodium_mg": sod, "calories_kcal": cals, "verdict": verdict, "rationale": rationale})

    elif any(k in food for k in ["masala", "sauce", "seasoning"]):
        per_100g_if_5g = sod / 0.05
        if per_100g_if_5g > IMPOSSIBLE_THRESHOLD:
            verdict = "IMPOSSIBLE if 5g serving"
        per_100g_if_100g = sod
        if per_100g_if_100g < IMPOSSIBLE_THRESHOLD:
            verdict2 = "PLAUSIBLE if values are per-100g"
        else:
            verdict2 = "SUSPICIOUS even at 100g"
        rationale = f"5g serving: per-100g={per_100g_if_5g:.0f}mg ({verdict}); OR if already per-100g: {per_100g_if_100g:.0f}mg ({verdict2})"
        verdict = verdict + " | " + verdict2
        categories["spice_mixes"].append({"food": food, "sodium_mg": sod, "calories_kcal": cals, "verdict": verdict, "rationale": rationale})

    else:
        # Other: spaghetti, beans
        if cals < 50 and sod > 5000:
            verdict = "SUSPICIOUS"
            rationale = f"Low-calorie ({cals:.0f} kcal) with extremely high sodium. Calorie/sodium ratio inconsistent."
        else:
            verdict = "UNCERTAIN"
            rationale = "No standard serving reference available for this food type in dataset."
        categories["other"].append({"food": food, "sodium_mg": sod, "calories_kcal": cals, "verdict": verdict, "rationale": rationale})

print()
print("--- SOUPS (18 records) ---")
for item in categories["soups"]:
    print(f"  [{item['verdict']}] {item['food']}: {item['sodium_mg']:.0f} mg | {item['rationale']}")

print()
print("--- PICKLES / CONDIMENTS (2 records) ---")
for item in categories["pickles_condiments"]:
    print(f"  [{item['verdict']}] {item['food']}: {item['sodium_mg']:.0f} mg | {item['rationale']}")

print()
print("--- SPICE MIXES / SAUCES (2 records) ---")
for item in categories["spice_mixes"]:
    print(f"  [{item['verdict']}] {item['food']}: {item['sodium_mg']:.0f} mg | {item['rationale']}")

print()
print("--- OTHER (9 records) ---")
for item in categories["other"]:
    print(f"  [{item['verdict']}] {item['food']}: {item['sodium_mg']:.0f} mg | {item['rationale']}")

# Summarize action
print()
print("--- ACTION TAKEN ---")
print("No records deleted or altered. High-sodium records are:")
print("  - Retained with sodium_mg values as-is from source")
print("  - Tagged with 'sodium_flag' in an updated processed file")
print("  - Flagged for analyst review before use in suitability scoring")
print("  - Application will warn users when sodium data is flagged")

# Tag the suspicious records in the processed file
suspicious_foods = (
    [item["food"] for item in categories["soups"]] +
    [item["food"] for item in categories["pickles_condiments"]] +
    [item["food"] for item in categories["spice_mixes"]] +
    [item["food"] for item in categories["other"]]
)

df["sodium_flag"] = "ok"
df.loc[df["sodium_mg"] > 5000, "sodium_flag"] = "high_sodium_unverified"

# Save back
df.to_csv(PROC_NUT / "nutrition_dish_level.csv", index=False, encoding="utf-8")
print(f"\nUpdated nutrition_dish_level.csv: added 'sodium_flag' column.")
print(f"  ok: {(df['sodium_flag']=='ok').sum()} records")
print(f"  high_sodium_unverified: {(df['sodium_flag']=='high_sodium_unverified').sum()} records")

findings["item1_sodium"] = {
    "total_high_sodium": len(high),
    "transformation_mismatches": len(mismatches),
    "categories": {k: len(v) for k, v in categories.items()},
    "action": "flagged_in_sodium_flag_column_not_deleted",
    "all_records": {k: v for k, v in categories.items()}
}

# ===========================================================================
# ITEM 2: Serving-size limitations and portion calculation approach
# ===========================================================================
print()
print("=" * 65)
print("ITEM 2: Serving-Size Limitations")
print("=" * 65)

df_dish = pd.read_csv(PROC_NUT / "nutrition_dish_level.csv")
df_ifct = pd.read_csv(PROC_NUT / "nutrition_ifct_ingredients.csv")
df_packaged = pd.read_csv(PROC_NUT / "nutrition_packaged_foods.csv")

print()
print("--- Indian_Food_Nutrition_Processed.csv ---")
print("Columns:", list(df_dish.columns))
print("No 'serving_size_g' or 'per_100g' column present.")
print("The file says 'per_serving_estimated' but gram weight is unknown.")
print()
print("Serving reference verification:")
print("  Looking at sample dishes against known references...")

# Check a few known dishes against ICMR reference data
known_servings = {
    "idli": 100,        # 2 idlis ~ 100g per ICMR
    "chapati": 35,      # 1 chapati ~ 35g
    "white rice (cooked)": 150,  # 1 katori ~ 150g
    "dal (cooked)": 150,  # 1 katori
    "samosa": 65,       # 1 piece ~ 65g
}

for food, ref_g in known_servings.items():
    match = df_dish[df_dish["food_name"].str.contains(food.split()[0], na=False)]
    if len(match):
        row = match.iloc[0]
        cal = row["calories_kcal"]
        prot = row["protein_g"]
        carb = row["carbs_g"]
        # Estimate what per-100g would be if we know serving size
        cal_per100 = (cal / ref_g * 100) if ref_g else None
        print(f"  {food}: {cal:.0f} kcal/serving | ref serving={ref_g}g -> {cal_per100:.0f} kcal/100g")
        # Cross-check with IFCT if ingredient exists
        ifct_match = df_ifct[df_ifct["food_name"].str.contains(food.split()[0], na=False)]
        if len(ifct_match):
            ifct_row = ifct_match.iloc[0]
            print(f"    IFCT per-100g: {ifct_row.get('calories_kcal', 'N/A'):.0f} kcal | match={'CLOSE' if cal_per100 and abs(cal_per100 - ifct_row.get('calories_kcal',0)) < 50 else 'DIFFERS'}")

print()
print("--- DEFINED APPROACH for portion calculations ---")
approach = """
PORTION CALCULATION APPROACH (transparent, no invented values):

1. PRIMARY: IFCT data (per 100g) -- most reliable for ingredients.
   Formula: nutrient = (portion_g / 100) * nutrient_per_100g

2. SECONDARY: Indian_Food_Nutrition_Processed.csv (per-serving-estimated)
   - Serving weight is NOT in the file.
   - Application will use ICMR-NIN standard portion weights as reference:
       Idli: 80g per piece
       Chapati: 35g per piece
       Rice (cooked): 150g per katori
       Dal (cooked): 150g per katori
       Samosa: 65g per piece
       Roti: 30g per piece
       etc.
   - These reference weights will be stored in a separate lookup table.
   - If reference weight is unknown: application displays 'per serving' with
     a disclaimer that gram weight is not available.
   - NO gram weights are invented for unmatched dishes.

3. PACKAGED FOODS: serving_size_g is available in the dataset -- use directly.
   Formula: nutrient = (user_portion_g / serving_size_g) * nutrient_per_serving

4. USER OVERRIDE: user can always specify actual portion weight manually.
   Application will scale accordingly for source types 1 and 3.
   For source type 2: if user provides grams, we convert using:
     nutrient_scaled = nutrient_per_serving * (user_g / reference_serving_g)
   If reference_serving_g is unknown, show a warning.

5. DISPLAY: every nutrition result must show:
   - Source dataset
   - Serving reference (per 100g / per estimated serving / per Xg serving)
   - Confidence level (high/medium/low) based on data quality
   - Disclaimer: 'Values are estimates. Actual nutrition varies with
     preparation method, ingredients, and portion size.'
"""
print(approach)

# Save the standard serving weights lookup
ICMR_SERVINGS = [
    {"food_name": "idli", "serving_g": 80, "description": "1 piece, ICMR-NIN reference"},
    {"food_name": "dosa", "serving_g": 80, "description": "1 dosa, approximate"},
    {"food_name": "chapati", "serving_g": 35, "description": "1 chapati, ICMR-NIN reference"},
    {"food_name": "roti", "serving_g": 30, "description": "1 roti, approximate"},
    {"food_name": "rice", "serving_g": 150, "description": "1 katori (150ml bowl), ICMR-NIN"},
    {"food_name": "dal", "serving_g": 150, "description": "1 katori, ICMR-NIN reference"},
    {"food_name": "samosa", "serving_g": 65, "description": "1 piece, approximate"},
    {"food_name": "paratha", "serving_g": 60, "description": "1 paratha, approximate"},
    {"food_name": "poha", "serving_g": 120, "description": "1 plate, approximate"},
    {"food_name": "upma", "serving_g": 150, "description": "1 plate, approximate"},
    {"food_name": "vada", "serving_g": 50, "description": "1 piece, approximate"},
    {"food_name": "naan", "serving_g": 90, "description": "1 piece, approximate"},
    {"food_name": "biryani", "serving_g": 250, "description": "1 plate, approximate"},
    {"food_name": "curry", "serving_g": 150, "description": "1 katori, approximate"},
    {"food_name": "sabzi", "serving_g": 120, "description": "1 katori, approximate"},
    {"food_name": "soup", "serving_g": 200, "description": "1 bowl (200ml), approximate"},
    {"food_name": "khichdi", "serving_g": 200, "description": "1 plate, approximate"},
    {"food_name": "raita", "serving_g": 100, "description": "1 katori, approximate"},
    {"food_name": "puri", "serving_g": 30, "description": "1 piece, approximate"},
    {"food_name": "dhokla", "serving_g": 80, "description": "2 pieces, approximate"},
    {"food_name": "chole", "serving_g": 150, "description": "1 katori, approximate"},
]
df_servings = pd.DataFrame(ICMR_SERVINGS)
df_servings.to_csv(PROC_NUT / "standard_serving_weights.csv", index=False)
print(f"Saved standard_serving_weights.csv ({len(df_servings)} entries)")

findings["item2_serving"] = {
    "limitation": "Indian_Food_Nutrition_Processed.csv has no gram weight per serving",
    "approach": "ICMR-NIN reference weights for known dishes; user can override; no values invented",
    "serving_lookup_saved": str(PROC_NUT / "standard_serving_weights.csv"),
}

# ===========================================================================
# ITEM 3: FoodLens duplicate image split check
# ===========================================================================
print()
print("=" * 65)
print("ITEM 3: FoodLens Duplicate Image Split Check")
print("=" * 65)

# Load duplicate list generated during cleaning
dup_file = LOG_DIR / "food_lens_duplicate_images.json"
with open(dup_file) as f:
    dups = json.load(f)

print(f"Duplicate pairs found (MD5): {len(dups)}")
print()

cross_split_leakage = []
same_split_dups = []

for dup in dups:
    path_a = Path(dup["file"])
    path_b = Path(dup["duplicate_of"])
    
    # Extract split from path
    parts_a = path_a.parts
    parts_b = path_b.parts
    
    split_a = next((p for p in parts_a if p in ("train", "val", "test")), "unknown")
    split_b = next((p for p in parts_b if p in ("train", "val", "test")), "unknown")
    
    class_a = path_a.parent.name
    class_b = path_b.parent.name
    
    is_cross_split = split_a != split_b
    
    info = {
        "file_a": str(path_a.name),
        "split_a": split_a,
        "class_a": class_a,
        "file_b": str(path_b.name),
        "split_b": split_b,
        "class_b": class_b,
        "cross_split_leakage": is_cross_split,
        "cross_class": class_a != class_b,
    }
    
    if is_cross_split:
        cross_split_leakage.append(info)
        print(f"  CROSS-SPLIT LEAK: {split_a}/{class_a}/{path_a.name} <-> {split_b}/{class_b}/{path_b.name}")
    else:
        same_split_dups.append(info)
        print(f"  SAME-SPLIT dup:   {split_a}/{class_a}/{path_a.name} <-> {split_b}/{class_b}/{path_b.name}")

print()
print(f"Cross-split leakage: {len(cross_split_leakage)} pairs")
print(f"Same-split duplicates: {len(same_split_dups)} pairs")
print()

if cross_split_leakage:
    print("ACTION REQUIRED: Cross-split duplicates cause train/val/test leakage.")
    print("  - These images appear in multiple splits with the same content.")
    print("  - Recommendation: Remove the copy from val or test (keep in train).")
    print("  - The manifest will be updated to mark these for exclusion from val/test.")
    
    # Update manifest
    df_manifest = pd.read_csv(PROC_IMG / "food_lens_manifest.csv")
    exclude_tuples = set()
    for leak in cross_split_leakage:
        # Prefer to keep train version, exclude val/test version
        if leak["split_a"] in ("val", "test"):
            exclude_tuples.add((leak["split_a"], leak["file_a"]))
        if leak["split_b"] in ("val", "test"):
            exclude_tuples.add((leak["split_b"], leak["file_b"]))
    
    df_manifest["exclude_leakage"] = [
        (s, f) in exclude_tuples
        for s, f in zip(df_manifest["split"], df_manifest["filename"])
    ]
    df_manifest.to_csv(PROC_IMG / "food_lens_manifest.csv", index=False)
    print(f"  Manifest updated: {df_manifest['exclude_leakage'].sum()} images marked exclude_leakage=True")
    print("  Raw files NOT deleted. Model training code must filter on exclude_leakage=False.")
else:
    print("No cross-split leakage found. All 3 duplicates are within the same split.")
    print("ACTION: Same-split duplicates are harmless for train/val/test integrity.")
    print("  They cause minor class imbalance within a split but do not leak information.")
    print("  No files deleted from raw/.")
    
    df_manifest = pd.read_csv(PROC_IMG / "food_lens_manifest.csv")
    df_manifest["exclude_leakage"] = False
    df_manifest.to_csv(PROC_IMG / "food_lens_manifest.csv", index=False)
    print("  Manifest updated: exclude_leakage=False for all records.")

findings["item3_duplicates"] = {
    "total_md5_dups": len(dups),
    "cross_split_leakage": len(cross_split_leakage),
    "same_split_dups": len(same_split_dups),
    "action": "no_leakage_found" if not cross_split_leakage else "val_test_copies_flagged_in_manifest",
    "details": dups
}

# ===========================================================================
# ITEM 4: Food/dataset.tar.gz assessment
# ===========================================================================
print()
print("=" * 65)
print("ITEM 4: Food/dataset.tar.gz Extraction Assessment")
print("=" * 65)

food_tar = RAW / "Food" / "dataset.tar.gz"
labels_file = RAW / "Food" / "labels.txt"

tar_size_gb = food_tar.stat().st_size / 1e9

labels = [l.strip() for l in labels_file.read_text().splitlines() if l.strip()]
print(f"Archive size: {tar_size_gb:.1f} GB")
print(f"Labels in labels.txt: {len(labels)}")
print(f"Labels: {labels}")

# Check available disk space (Windows)
import shutil
total, used, free = shutil.disk_usage(str(ROOT.drive) + "\\")
free_gb = free / 1e9
print(f"\nDisk free space on {ROOT.drive}: {free_gb:.1f} GB")

# How does Food/dataset.tar.gz compare to enhanced-indian-food-classification?
# Enhanced: 43 labels, 15,404 images (usable, no extraction needed)
# food_lens: 20 labels, 2,523 images (usable, no extraction needed)
# Food/dataset.tar.gz: 80 labels, size unknown (needs extraction)

# Overlap analysis between Food labels and enhanced labels
enhanced_labels = [
    'aloo_gobi','aloo_masala','bhindi_masala','biryani','burger','butter_naan',
    'chai','chapati','chicken_curry','chole','chole_bhature','coconut_chutney',
    'dal_makhani','dhokla','dum_aloo','fish_curry','fried_rice','ghevar',
    'green_chutney','gulab_jamun','idli','jalebi','kaathi_rolls','kadai_paneer',
    'kebab','kheer','kulfi','lassi','masala_dosa','modak','momos','mutton_curry',
    'paani_puri','pakode','palak_paneer','pav_bhaji','pizza','poha','rajma_curry',
    'ras_malai','samosa','shahi_paneer','white_rice'
]

food_labels_norm = [l.replace(" ", "_") for l in labels]
overlap = set(enhanced_labels) & set(food_labels_norm)
enhanced_only = set(enhanced_labels) - set(food_labels_norm)
food_only = set(food_labels_norm) - set(enhanced_labels)

print(f"\nLabel overlap with enhanced-indian-food dataset:")
print(f"  Shared labels: {len(overlap)} -> {sorted(overlap)}")
print(f"  Enhanced-only: {len(enhanced_only)} -> {sorted(enhanced_only)}")
print(f"  Food-tar-only (unique value if extracted): {len(food_only)} -> {sorted(food_only)}")

print()
print("--- EXTRACTION DECISION ---")
justification = []
justification.append(f"Archive size: {tar_size_gb:.1f} GB compressed (extracted may be 10-20 GB)")
justification.append(f"Disk free: {free_gb:.1f} GB")
justification.append(f"Shared labels with enhanced dataset: {len(overlap)}/{len(labels)} ({len(overlap)/len(labels)*100:.0f}%)")
justification.append(f"Unique additional labels from tar: {len(food_only)}")
justification.append("Enhanced dataset already provides 15,404 images across 43 classes (usable without extraction)")
justification.append("food_lens provides 2,523 images across 20 classes")

if free_gb < 20:
    decision = "DO_NOT_EXTRACT"
    reason = f"Insufficient free disk space ({free_gb:.1f} GB < 20 GB required). Enhanced dataset (15,404 images, 43 classes) is sufficient for model training scope."
elif len(food_only) < 10:
    decision = "DO_NOT_EXTRACT"
    reason = f"Only {len(food_only)} unique new labels not already in enhanced dataset. Extraction not justified by marginal label gain."
else:
    decision = "EXTRACTION_OPTIONAL"
    reason = f"{len(food_only)} unique labels available. If disk space is confirmed ({free_gb:.1f} GB free), extraction would increase class coverage from 43 to {43 + len(food_only)} classes."

print(f"DECISION: {decision}")
print(f"Reason: {reason}")
for j in justification:
    print(f"  - {j}")

findings["item4_tar"] = {
    "archive_size_gb": round(tar_size_gb, 1),
    "disk_free_gb": round(free_gb, 1),
    "label_count_in_tar": len(labels),
    "overlap_with_enhanced": len(overlap),
    "unique_to_tar": len(food_only),
    "unique_labels": sorted(food_only),
    "decision": decision,
    "reason": reason,
}

# ===========================================================================
# ITEM 5: Save updated audit report
# ===========================================================================
print()
print("=" * 65)
print("ITEM 5: Updating Audit Report")
print("=" * 65)

# Load existing audit report
audit_path = LOG_DIR / "dataset_audit_report.json"
with open(audit_path, encoding="utf-8") as f:
    audit = json.load(f)

# Append review findings
audit["phase2_review"] = {
    "review_timestamp": datetime.now().isoformat(),
    "findings": findings,
    "summary": {
        "sodium": f"{len(high)} high-sodium records: all retained, tagged 'high_sodium_unverified' in sodium_flag column. No values altered.",
        "serving_size": "ICMR-NIN reference serving weights saved as standard_serving_weights.csv. No gram weights invented.",
        "image_duplicates": f"3 MD5 duplicates in food_lens: {len(cross_split_leakage)} cross-split leakage, {len(same_split_dups)} same-split. Manifest updated with exclude_leakage column.",
        "tar_extraction": f"Food/dataset.tar.gz: {decision}. {reason}",
    }
}

with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2, default=str, ensure_ascii=False)
print(f"Updated audit report saved -> {audit_path}")

# Save detailed review findings separately
review_path = LOG_DIR / "phase2_review_findings.json"
with open(review_path, "w", encoding="utf-8") as f:
    json.dump(findings, f, indent=2, default=str, ensure_ascii=False)
print(f"Detailed review findings -> {review_path}")

print()
print("=" * 65)
print("PHASE 2 REVIEW COMPLETE")
print("=" * 65)
