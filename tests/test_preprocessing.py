"""
tests/test_preprocessing.py -- Automated verification of Phase 2 outputs
Tests:
  1. Processed nutrition files exist and have correct shapes
  2. No raw food names survived (all normalised to lowercase)
  3. Numeric nutrition columns are float64
  4. No negative calorie/protein/fat/carb values
  5. Duplicate food names removed
  6. Serving reference column present
  7. IFCT energy kJ -> kcal conversion correct
  8. Recipe time parsing (minutes, not raw strings)
  9. Indian cuisine filter applied
  10. Image manifest contains expected splits
  11. Unified nutrition lookup contains >= dish-level count
  12. Log files created
"""

import sys
import math
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC_NUT = ROOT / "data" / "processed" / "nutrition"
PROC_REC = ROOT / "data" / "processed" / "recipes"
PROC_IMG = ROOT / "data" / "processed" / "images"
LOG_DIR = ROOT / "logs"

PASS = 0
FAIL = 0
results = []


def test(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    msg = f"[{status}] {name}"
    if detail:
        msg += f" | {detail}"
    print(msg)
    results.append({"test": name, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# 1. Output files exist
# ---------------------------------------------------------------------------
expected_files = {
    "nutrition_dish_level.csv": PROC_NUT,
    "nutrition_ifct_ingredients.csv": PROC_NUT,
    "nutrition_daily_general.csv": PROC_NUT,
    "nutrition_packaged_foods.csv": PROC_NUT,
    "nutrition_unified_lookup.csv": PROC_NUT,
    "recipes_indian_food_dataset.csv": PROC_REC,
    "recipes_ndtv.csv": PROC_REC,
    "recipes_indian_food_meta.csv": PROC_REC,
    "recipes_regional.csv": PROC_REC,
    "food_lens_manifest.csv": PROC_IMG,
    "food_lens_class_manifest.csv": PROC_IMG,
    "enhanced_indian_food_label_manifest.csv": PROC_IMG,
}
for fname, fdir in expected_files.items():
    path = fdir / fname
    test(f"File exists: {fname}", path.exists())


# ---------------------------------------------------------------------------
# 2. Nutrition dish level -- shape, dtypes, no-negative, no-duplicate
# ---------------------------------------------------------------------------
df_dish = pd.read_csv(PROC_NUT / "nutrition_dish_level.csv")
test("nutrition_dish_level: rows >= 1000", len(df_dish) >= 1000, f"rows={len(df_dish)}")
test("nutrition_dish_level: food_name present", "food_name" in df_dish.columns)
test("nutrition_dish_level: food_name lowercase", df_dish["food_name"].str.lower().eq(df_dish["food_name"]).all(), "all food names lowercase")
test("nutrition_dish_level: no duplicate food_name", not df_dish.duplicated(subset=["food_name"]).any(), f"dupes={df_dish.duplicated(subset=['food_name']).sum()}")
test("nutrition_dish_level: serving_reference present", "serving_reference" in df_dish.columns)
test("nutrition_dish_level: source_file present", "source_file" in df_dish.columns)
for col in ["calories_kcal", "protein_g", "carbs_g", "fat_g"]:
    if col in df_dish.columns:
        neg = (df_dish[col] < 0).sum()
        test(f"nutrition_dish_level: {col} no negative values", neg == 0, f"negatives={neg}")
        test(f"nutrition_dish_level: {col} is numeric", pd.api.types.is_float_dtype(df_dish[col]) or pd.api.types.is_integer_dtype(df_dish[col]))


# ---------------------------------------------------------------------------
# 3. IFCT -- energy conversion
# ---------------------------------------------------------------------------
df_ifct = pd.read_csv(PROC_NUT / "nutrition_ifct_ingredients.csv")
test("ifct: rows == 542", len(df_ifct) == 542, f"rows={len(df_ifct)}")
test("ifct: calories_kcal present (converted from kJ)", "calories_kcal" in df_ifct.columns)
test("ifct: serving_reference = per_100g", (df_ifct["serving_reference"] == "per_100g").all())

# Spot check: energy kJ -> kcal for row 0 (Amaranth seed ~60g carbs)
# IFCT row 0: energy 9.89g moisture, should have energy > 0
if "calories_kcal" in df_ifct.columns and "energy_kj" in df_ifct.columns:
    row0_kcal = df_ifct["calories_kcal"].iloc[0]
    row0_kj = df_ifct["energy_kj"].iloc[0]
    expected_kcal = round(row0_kj / 4.184, 2) if pd.notna(row0_kj) else None
    if expected_kcal is not None:
        test("ifct: kJ->kcal conversion correct for row0", abs(row0_kcal - expected_kcal) < 1.0, f"kcal={row0_kcal} expected~{expected_kcal}")


# ---------------------------------------------------------------------------
# 4. IFCT mean extraction (no +/- strings)
# ---------------------------------------------------------------------------
for col in ["protein_g", "fat_g", "carbs_g"]:
    if col in df_ifct.columns:
        # Values should be numeric floats, not strings
        non_numeric = df_ifct[col].apply(lambda x: isinstance(x, str)).sum()
        test(f"ifct: {col} is pure numeric (no +/- strings)", non_numeric == 0)


# ---------------------------------------------------------------------------
# 5. Daily nutrition -- duplicate removal
# ---------------------------------------------------------------------------
df_daily = pd.read_csv(PROC_NUT / "nutrition_daily_general.csv")
test("daily_nutrition: no duplicate food_name", not df_daily.duplicated(subset=["food_name"]).any(), f"dupes={df_daily.duplicated(subset=['food_name']).sum()}")
test("daily_nutrition: rows < 645 (dupes removed)", len(df_daily) < 645, f"rows={len(df_daily)}")


# ---------------------------------------------------------------------------
# 6. Packaged foods -- per-100g computation
# ---------------------------------------------------------------------------
df_pack = pd.read_csv(PROC_NUT / "nutrition_packaged_foods.csv")
test("packaged_foods: calories_per_100g column present", "calories_per_100g" in df_pack.columns)
# Spot check: if serving=100g, calories_per_100g should equal calories_kcal
mask = df_pack["serving_size_g"].notna() & (df_pack["serving_size_g"] == 100.0) & df_pack["calories_kcal"].notna()
if mask.any():
    sub = df_pack[mask]
    match = (abs(sub["calories_per_100g"] - sub["calories_kcal"]) < 1.0).all()
    test("packaged_foods: calories_per_100g == calories_kcal when serving=100g", match)


# ---------------------------------------------------------------------------
# 7. Recipes -- Indian cuisine filter
# ---------------------------------------------------------------------------
df_recipes = pd.read_csv(PROC_REC / "recipes_indian_food_dataset.csv")
test("recipes_indian_food_dataset: rows < 6871 (filtered)", len(df_recipes) < 6871, f"rows={len(df_recipes)}")
test("recipes_indian_food_dataset: rows >= 4000", len(df_recipes) >= 4000, f"rows={len(df_recipes)}")
if "cuisine" in df_recipes.columns:
    non_indian_count = 0
    INDIAN_CUISINES = {
        "indian", "north indian recipes", "south indian recipes",
        "bengali recipes", "maharashtrian recipes", "kerala recipes",
        "tamil nadu", "karnataka", "rajasthani", "punjabi", "gujarat",
        "andhra", "gujarati recipes", "chettinad", "goan recipes",
        "hyderabadi", "kashmiri",
    }
    non_indian = df_recipes[~df_recipes["cuisine"].str.lower().isin(INDIAN_CUISINES)]
    test("recipes_indian_food_dataset: all cuisines are Indian", len(non_indian) == 0, f"non_indian_rows={len(non_indian)}")


# ---------------------------------------------------------------------------
# 8. NDTV recipes -- time parsing (should be numeric minutes, not strings)
# ---------------------------------------------------------------------------
df_ndtv = pd.read_csv(PROC_REC / "recipes_ndtv.csv")
test("ndtv_recipes: total_time_mins is numeric", "total_time_mins" in df_ndtv.columns and pd.api.types.is_float_dtype(df_ndtv["total_time_mins"]))
test("ndtv_recipes: no raw time strings in total_time_mins", df_ndtv.get("total_time_mins", pd.Series(dtype="float")).apply(lambda x: not isinstance(x, str)).all() if "total_time_mins" in df_ndtv.columns else True)
test("ndtv_recipes: no duplicate recipe names", not df_ndtv.duplicated(subset=["recipe_name_clean"]).any() if "recipe_name_clean" in df_ndtv.columns else True)


# ---------------------------------------------------------------------------
# 9. indian_food_meta -- -1 replaced with NaN
# ---------------------------------------------------------------------------
df_meta = pd.read_csv(PROC_REC / "recipes_indian_food_meta.csv")
for col in ["prep_time", "cook_time"]:
    if col in df_meta.columns:
        has_neg1 = (df_meta[col] == -1).sum()
        test(f"indian_food_meta: {col} has no -1 values", has_neg1 == 0, f"-1 count={has_neg1}")


# ---------------------------------------------------------------------------
# 10. Regional recipes -- time in minutes (not seconds)
# ---------------------------------------------------------------------------
df_reg = pd.read_csv(PROC_REC / "recipes_regional.csv")
test("regional_recipes: total_time_mins column present", "total_time_mins" in df_reg.columns)
if "total_time_mins" in df_reg.columns:
    # Active cooking rarely > 600 mins; in seconds they'd be > 36000
    max_time = df_reg["total_time_mins"].dropna().max()
    # Some recipes (e.g. Candied Ash Gourd, Sorpotel) have multi-day passive times.
    # 20000 mins = ~13 days. Raw seconds would be 36000*1000+ which is astronomically higher.
    test("regional_recipes: total_time_mins plausible (< 20000, confirms conversion from seconds)", float(max_time) < 20000 if pd.notna(max_time) else True, f"max={max_time} mins (~{float(max_time)/60:.0f}h)")


# ---------------------------------------------------------------------------
# 11. Image manifest
# ---------------------------------------------------------------------------
df_img = pd.read_csv(PROC_IMG / "food_lens_manifest.csv")
test("food_lens_manifest: 2523 total images", len(df_img) == 2523, f"rows={len(df_img)}")
test("food_lens_manifest: split column present", "split" in df_img.columns)
test("food_lens_manifest: all 3 splits present", set(df_img["split"].unique()) >= {"train", "val", "test"})
test("food_lens_manifest: 20 classes in train", df_img[df_img["split"] == "train"]["class"].nunique() == 20, f"classes={df_img[df_img['split']=='train']['class'].nunique()}")
test("food_lens_manifest: corrupt column present", "corrupt" in df_img.columns)
test("food_lens_manifest: no corrupt images found", df_img["corrupt"].sum() == 0, f"corrupt={df_img['corrupt'].sum()}")


# ---------------------------------------------------------------------------
# 12. Enhanced food label manifest
# ---------------------------------------------------------------------------
df_enh = pd.read_csv(PROC_IMG / "enhanced_indian_food_label_manifest.csv")
test("enhanced_label_manifest: 43 unique labels", len(df_enh) == 43, f"labels={len(df_enh)}")
test("enhanced_label_manifest: total column present", "total" in df_enh.columns or any("total" in c.lower() for c in df_enh.columns))


# ---------------------------------------------------------------------------
# 13. Unified nutrition lookup
# ---------------------------------------------------------------------------
df_unified = pd.read_csv(PROC_NUT / "nutrition_unified_lookup.csv")
test("unified_lookup: >= 1556 records", len(df_unified) >= 1556, f"rows={len(df_unified)}")
test("unified_lookup: no duplicate food_name", not df_unified.duplicated(subset=["food_name"]).any(), f"dupes={df_unified.duplicated(subset=['food_name']).sum()}")
test("unified_lookup: food_name_normalised present", "food_name_normalised" in df_unified.columns)


# ---------------------------------------------------------------------------
# 14. Log files created
# ---------------------------------------------------------------------------
for logfile in ["data_audit.log", "data_cleaning.log"]:
    p = LOG_DIR / logfile
    test(f"Log file created: {logfile}", p.exists() and p.stat().st_size > 0)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 55)
print(f"PREPROCESSING TESTS: {PASS} PASSED / {FAIL} FAILED / {PASS+FAIL} TOTAL")
print("=" * 55)
if FAIL > 0:
    print("FAILED tests:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  - {r['test']}: {r['detail']}")
    sys.exit(1)
else:
    print("All tests passed.")
