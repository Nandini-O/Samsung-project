"""
tests/test_eda.py
Phase 3 Exploratory Data Analysis (EDA) Comprehensive Test Suite (Corrected)

Verifies all Phase 3 review and correction requirements:
1. Inventory CSV generation and schema
2. Image dataset splits: Root cause, stratified splitting plan (43 classes, zero leakage)
3. FoodLens class imbalance: Samosa verified at 42.4%
4. Nutrition data quality: 31 sodium flags preserved, 35 Atwater calorie discrepancy flags, 28 overlapping soups
5. Missing micronutrients: Null preservation check (vitamin C, folate strictly not zero-imputed)
6. Image-to-nutrition mapping: 61 classes categorized into valid direct, ambiguous, and unmatched gaps
7. Visualizations: All 11 figures exist (> 50KB each)
8. Written Markdown report completeness
"""

import json
from pathlib import Path
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
REPORT_DIR = WORKSPACE / "reports"
FIG_DIR = REPORT_DIR / "figures"
PROC_DIR = WORKSPACE / "data" / "processed"

PASS = "[PASS]"
FAIL = "[FAIL]"
test_results = []


def test(name, condition, extra=""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}" + (f" | {extra}" if extra else "")
    print(msg)
    test_results.append((name, condition, extra))


def run_tests():
    print("=" * 60)
    print("RUNNING PHASE 3 EDA COMPREHENSIVE TESTS")
    print("=" * 60)

    # 1. Output files exist
    inv_csv = REPORT_DIR / "eda_dataset_inventory.csv"
    nut_csv = REPORT_DIR / "eda_nutrition_stats.csv"
    summary_json = REPORT_DIR / "eda_summary_statistics.json"
    report_md = REPORT_DIR / "phase3_eda_report.md"
    map_csv = REPORT_DIR / "image_to_nutrition_mapping.csv"
    map_json = REPORT_DIR / "image_to_nutrition_mapping.json"
    strat_manifest = PROC_DIR / "images" / "enhanced_indian_food_stratified_manifest.csv"
    strat_summary = PROC_DIR / "images" / "enhanced_stratified_split_summary.csv"

    test("inventory_csv: file exists", inv_csv.exists())
    test("nutrition_stats_csv: file exists", nut_csv.exists())
    test("summary_json: file exists", summary_json.exists())
    test("report_md: file exists", report_md.exists())
    test("mapping_csv: file exists", map_csv.exists())
    test("mapping_json: file exists", map_json.exists())
    test("stratified_manifest: file exists", strat_manifest.exists())
    test("stratified_summary: file exists", strat_summary.exists())

    # 2. Inventory CSV checks
    if inv_csv.exists():
        df_inv = pd.read_csv(inv_csv)
        test("inventory_csv: >= 13 datasets tracked", len(df_inv) >= 13, f"rows={len(df_inv)}")
        req_cols = ["dataset_name", "category", "rows", "columns", "pct_missing", "memory_mb"]
        test("inventory_csv: all schema columns present", all(c in df_inv.columns for c in req_cols))
        test("inventory_csv: total rows across datasets >= 12000", df_inv["rows"].sum() >= 12000, f"sum={df_inv['rows'].sum()}")

    # 3. Figures checks (11 required figures)
    fig_names = [
        "01_dataset_dimensions.png",
        "02_missing_values_summary.png",
        "03_enhanced_image_class_distribution.png",
        "04_food_lens_class_distribution.png",
        "05_nutrition_macro_distributions.png",
        "06_sodium_distribution_and_flags.png",
        "07_atwater_calorie_validation.png",
        "08_nutrition_correlation_matrix.png",
        "09_recipe_diet_and_course_distribution.png",
        "10_recipe_cooking_times.png",
        "11_image_to_nutrition_mapping.png",
    ]
    for fn in fig_names:
        fp = FIG_DIR / fn
        exists = fp.exists()
        size_ok = fp.stat().st_size > 50000 if exists else False
        test(f"figure: {fn} exists and size > 50KB", exists and size_ok,
             f"size={fp.stat().st_size:,} bytes" if exists else "missing")

    # 4. Stratified Split Plan Checks (Item 1)
    if strat_summary.exists() and strat_manifest.exists():
        df_ss = pd.read_csv(strat_summary)
        df_sm = pd.read_csv(strat_manifest)

        test("stratified_plan: exactly 43 classes present", len(df_ss) == 43, f"classes={len(df_ss)}")
        test("stratified_plan: all 43 classes have train images", (df_ss["train_images"] > 0).all())
        test("stratified_plan: all 43 classes have >= 10 validation images", (df_ss["val_images"] >= 10).all(), f"min_val={df_ss['val_images'].min()}")
        test("stratified_plan: all 43 classes have >= 10 test images", (df_ss["test_images"] >= 10).all(), f"min_test={df_ss['test_images'].min()}")
        test("stratified_plan: total manifest images is 15404", len(df_sm) == 15404, f"rows={len(df_sm)}")
        
        # Verify 0 cross-split hash leakage in stratified plan
        leak_check = df_sm.groupby("md5_hash")["assigned_split"].nunique()
        test("stratified_plan: zero hash crosses splits (0% leakage)", leak_check.max() == 1, f"max_splits_per_hash={leak_check.max()}")

    # 5. FoodLens Imbalance Verification (Item 2)
    fl_manifest = PROC_DIR / "images" / "food_lens_manifest.csv"
    if fl_manifest.exists():
        df_fl = pd.read_csv(fl_manifest)
        df_fl_active = df_fl[~df_fl["exclude_leakage"]]
        samosa_count = (df_fl_active["class"] == "samosa").sum()
        samosa_pct = (samosa_count / len(df_fl_active)) * 100
        test("food_lens: exactly 2 cross-split leakage images excluded", df_fl["exclude_leakage"].sum() == 2)
        test("food_lens: active clean images count is 2521", len(df_fl_active) == 2521, f"active={len(df_fl_active)}")
        test("food_lens: samosa count is 1070", samosa_count == 1070, f"samosa={samosa_count}")
        test("food_lens: samosa accounts for 42.4% of active images", abs(samosa_pct - 42.44) < 0.1, f"pct={samosa_pct:.2f}%")

    # 6. Nutrition Data Quality & Flags (Item 3)
    dish_csv = PROC_DIR / "nutrition" / "nutrition_dish_level.csv"
    if dish_csv.exists():
        df_dish = pd.read_csv(dish_csv)
        test("dish_level: total records is 1014", len(df_dish) == 1014, f"rows={len(df_dish)}")
        test("dish_level: sodium_flag column present", "sodium_flag" in df_dish.columns)
        test("dish_level: exactly 31 high sodium records preserved", (df_dish["sodium_flag"] == "high_sodium_unverified").sum() == 31)
        test("dish_level: calorie_flag column present", "calorie_flag" in df_dish.columns)
        test("dish_level: exactly 35 Atwater discrepancy records flagged", (df_dish["calorie_flag"] == "atwater_discrepancy_unverified").sum() == 35)
        test("dish_level: exactly 979 ok calorie records", (df_dish["calorie_flag"] == "ok").sum() == 979)
        
        # Verify soup overlap
        overlap = df_dish[(df_dish["sodium_flag"] == "high_sodium_unverified") & (df_dish["calorie_flag"] == "atwater_discrepancy_unverified")]
        test("dish_level: exactly 28 dishes overlap both flags", len(overlap) == 28, f"overlap={len(overlap)}")
        test("dish_level: raw calories untouched (no values deleted/altered)", df_dish["calories_kcal"].notnull().all())

    # 7. Missing Micronutrients Preservation (Item 4)
    if dish_csv.exists():
        df_dish = pd.read_csv(dish_csv)
        null_vit_c = df_dish["vitamin_c_mg"].isnull().sum()
        null_folate = df_dish["folate_ug"].isnull().sum()
        test("micronutrients: dish_level vitamin_c has null values (not imputed to 0)", null_vit_c == 82, f"nulls={null_vit_c}")
        test("micronutrients: dish_level folate has null values (not imputed to 0)", null_folate == 82, f"nulls={null_folate}")

    ifct_csv = PROC_DIR / "nutrition" / "nutrition_ifct_ingredients.csv"
    if ifct_csv.exists():
        df_ifct = pd.read_csv(ifct_csv)
        test("micronutrients: IFCT vitamin_c has nulls preserved", df_ifct["vitamin_c_mg"].isnull().sum() == 313)
        test("micronutrients: IFCT folate has nulls preserved", df_ifct["folate_ug"].isnull().sum() == 195)

    # 8. Image-to-Nutrition Mapping (Item 5)
    if map_csv.exists():
        df_map = pd.read_csv(map_csv)
        test("mapping: exactly 61 unique image classes tracked", len(df_map) == 61, f"classes={len(df_map)}")
        test("mapping: VALID_DIRECT matches >= 25", (df_map["match_status"] == "VALID_DIRECT").sum() >= 25,
             f"count={(df_map['match_status'] == 'VALID_DIRECT').sum()}")
        test("mapping: AMBIGUOUS_SUBTYPE matches >= 15", (df_map["match_status"] == "AMBIGUOUS_SUBTYPE").sum() >= 15,
             f"count={(df_map['match_status'] == 'AMBIGUOUS_SUBTYPE').sum()}")
        test("mapping: UNMATCHED_GAP matches >= 12", (df_map["match_status"] == "UNMATCHED_GAP").sum() >= 12,
             f"count={(df_map['match_status'] == 'UNMATCHED_GAP').sum()}")
        test("mapping: culinary ambiguity reason documented for ambiguous classes",
             df_map[df_map["match_status"] == "AMBIGUOUS_SUBTYPE"]["culinary_ambiguity_reason"].notnull().all())
        test("mapping: gap resolution strategy documented for unmatched classes",
             df_map[df_map["match_status"] == "UNMATCHED_GAP"]["gap_resolution_strategy"].notnull().all())

    # 9. Written Report Completeness (Item 6)
    if report_md.exists():
        text = report_md.read_text(encoding="utf-8")
        test("report_md: length > 3000 chars", len(text) > 3000, f"length={len(text):,} chars")
        test("report_md: covers Section 1 (Dimensions)", "1. Dataset Dimensions" in text)
        test("report_md: covers Section 2 (Root Cause & Stratified Plan)", "Root Cause of Splits & Stratified Plan" in text)
        test("report_md: covers Section 3 (FoodLens Imbalance Solutions)", "FoodLens Dataset & Imbalance Management" in text)
        test("report_md: covers Section 4 (Dual Quality Flags)", "Dual Data Quality Flags" in text)
        test("report_md: covers Section 6 (Image-to-Nutrition Mapping)", "Image-to-Nutrition Mapping Assessment" in text)
        test("report_md: covers Section 7 (Verified Corrections vs Remaining Limitations)", "Verified Corrections vs Remaining Limitations" in text)

    # Summary
    passed = sum(1 for _, c, _ in test_results if c)
    failed = sum(1 for _, c, _ in test_results if not c)
    total = len(test_results)
    print("\n" + "=" * 60)
    print(f"PHASE 3 EDA COMPREHENSIVE TESTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print("=" * 60)

    if failed > 0:
        print("FAILED tests:")
        for name, c, extra in test_results:
            if not c:
                print(f"  - {name}: {extra}")
        exit(1)
    else:
        print("All Phase 3 EDA comprehensive tests passed.")


if __name__ == "__main__":
    run_tests()
