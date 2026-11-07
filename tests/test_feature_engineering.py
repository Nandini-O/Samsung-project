"""
tests/test_feature_engineering.py
Phase 4 Feature Engineering & Dataset Preparation Test Suite

Verifies:
1. Composite Recipe Nutrition Builder (15 unmatched classes, Atwater consistency < 5%)
2. Ambiguous Subtypes Registry (14 families, 36 discrete profiles, default profiles verified)
3. Master Food & Nutrition Knowledge Base (all classes covered, flags preserved, missing values not zero-filled)
4. Vision Dataset Preparation (43 classes, 70/15/15 splits, zero duplicate leakage, class weights)
5. Nutrient Gap & Suitability Scoring Engine (ICMR RDA profiles, 1.0-10.0 scores, clinical penalties)
6. Recipe Feature Store (5,381 recipes, complexity levels, ingredient tokens)
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

import sys
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

FEATURE_DIR = WORKSPACE / "data" / "processed" / "feature_store"
VISION_DIR = WORKSPACE / "data" / "processed" / "vision_splits"
MODELS_DIR = WORKSPACE / "models"

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
    print("RUNNING PHASE 4 FEATURE ENGINEERING TESTS")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. Composite Recipes (15 unmatched food classes)
    # -------------------------------------------------------------
    comp_csv = FEATURE_DIR / "composite_recipes_nutrition.csv"
    comp_json = FEATURE_DIR / "composite_recipes_nutrition.json"

    test("composite: csv exists", comp_csv.exists())
    test("composite: json exists", comp_json.exists())

    if comp_csv.exists():
        df_comp = pd.read_csv(comp_csv)
        test("composite: exactly 15 classes computed", len(df_comp) == 15, f"count={len(df_comp)}")
        
        required_classes = [
            "aloo_gobi", "aloo_masala", "bhindi_masala", "chole", "chole_bhature",
            "ghevar", "jalebi", "kadai_paneer", "modak", "momos",
            "mutton_curry", "ras_malai", "bisibelebath", "paniyaram", "ven_pongal"
        ]
        for rc in required_classes:
            test(f"composite: class '{rc}' present", rc in df_comp["class_name"].values)

        # Check Atwater consistency
        test("composite: all 15 meet Atwater < 5% error", (df_comp["atwater_discrepancy_pct"] < 5.0).all(),
             f"max_err={df_comp['atwater_discrepancy_pct'].max():.2f}%")
        test("composite: serving sizes are positive", (df_comp["serving_size_g"] > 0).all())

    # -------------------------------------------------------------
    # 2. Ambiguous Subtypes Registry
    # -------------------------------------------------------------
    sub_json = FEATURE_DIR / "ambiguous_subtypes_registry.json"
    sub_csv = FEATURE_DIR / "ambiguous_subtypes_registry.csv"

    test("subtypes: json exists", sub_json.exists())
    test("subtypes: csv exists", sub_csv.exists())

    if sub_csv.exists() and sub_json.exists():
        df_sub = pd.read_csv(sub_csv)
        with open(sub_json) as f:
            reg = json.load(f)

        test("subtypes: 14 parent families registered", len(reg) == 14, f"count={len(reg)}")
        test("subtypes: total discrete sub-profiles >= 30", len(df_sub) >= 30, f"count={len(df_sub)}")
        
        # Verify samosa has distinct potato and keema profiles
        samosa_subs = df_sub[df_sub["parent_class"] == "samosa"]
        test("subtypes: samosa has >= 2 sub-profiles", len(samosa_subs) >= 2)
        pot_cal = samosa_subs[samosa_subs["subtype_id"] == "samosa_potato"]["calories_kcal"].values[0]
        kee_cal = samosa_subs[samosa_subs["subtype_id"] == "samosa_keema"]["calories_kcal"].values[0]
        test("subtypes: samosa potato and keema have distinct calories", pot_cal != kee_cal,
             f"potato={pot_cal}, keema={kee_cal}")

        # Every parent family has exactly 1 default
        defaults_per_parent = df_sub[df_sub["is_default"]].groupby("parent_class").size()
        test("subtypes: every parent family has exactly 1 default", (defaults_per_parent == 1).all() and len(defaults_per_parent) == 14)

    # -------------------------------------------------------------
    # 3. Master Food & Nutrition Knowledge Base
    # -------------------------------------------------------------
    master_csv = FEATURE_DIR / "master_food_nutrition_kb.csv"
    master_json = FEATURE_DIR / "master_food_nutrition_kb.json"

    test("master_kb: csv exists", master_csv.exists())
    test("master_kb: json exists", master_json.exists())

    if master_csv.exists():
        df_kb = pd.read_csv(master_csv)
        test("master_kb: total items >= 50", len(df_kb) >= 50, f"count={len(df_kb)}")
        test("master_kb: match_type column present", "match_type" in df_kb.columns)
        test("master_kb: contains VALID_DIRECT entries", (df_kb["match_type"] == "VALID_DIRECT").sum() >= 25)
        test("master_kb: contains COMPOSITE_RECIPE entries", (df_kb["match_type"] == "COMPOSITE_RECIPE").sum() == 15)
        test("master_kb: contains AMBIGUOUS_SUBTYPE entries", (df_kb["match_type"] == "AMBIGUOUS_SUBTYPE").sum() >= 14)
        
        # Missing values must remain NaN, not 0
        test("master_kb: missing micronutrients remain NaN (not zero-filled)",
             df_kb["calcium_mg"].isnull().any() and df_kb["vitamin_c_mg"].isnull().any())
        test("master_kb: sodium_flag column preserved", "sodium_flag" in df_kb.columns)
        test("master_kb: calorie_flag column preserved", "calorie_flag" in df_kb.columns)

    # -------------------------------------------------------------
    # 4. Vision Dataset Splits & Class Weights
    # -------------------------------------------------------------
    c2idx_path = MODELS_DIR / "class_to_idx.json"
    cweight_path = MODELS_DIR / "class_weights.json"
    train_split = VISION_DIR / "train_split.csv"
    val_split = VISION_DIR / "val_split.csv"
    test_split = VISION_DIR / "test_split.csv"

    test("vision: class_to_idx.json exists", c2idx_path.exists())
    test("vision: class_weights.json exists", cweight_path.exists())
    test("vision: train_split.csv exists", train_split.exists())
    test("vision: val_split.csv exists", val_split.exists())
    test("vision: test_split.csv exists", test_split.exists())

    if train_split.exists() and val_split.exists() and test_split.exists():
        df_tr = pd.read_csv(train_split)
        df_va = pd.read_csv(val_split)
        df_te = pd.read_csv(test_split)

        test("vision: train has 43 classes", df_tr["class_name"].nunique() == 43)
        test("vision: val has 43 classes", df_va["class_name"].nunique() == 43)
        test("vision: test has 43 classes", df_te["class_name"].nunique() == 43)

        # Check total image counts (70% / 15% / 15%)
        tot = len(df_tr) + len(df_va) + len(df_te)
        test("vision: total stratified images is 15404", tot == 15404, f"total={tot}")
        test("vision: train split is 70%", abs(len(df_tr) / tot - 0.70) < 0.01)
        test("vision: val split is 15%", abs(len(df_va) / tot - 0.15) < 0.01)
        test("vision: test split is 15%", abs(len(df_te) / tot - 0.15) < 0.01)

        # Zero leakage check
        tr_h = set(df_tr["md5_hash"])
        va_h = set(df_va["md5_hash"])
        te_h = set(df_te["md5_hash"])
        test("vision: zero leakage train-val", len(tr_h.intersection(va_h)) == 0)
        test("vision: zero leakage train-test", len(tr_h.intersection(te_h)) == 0)
        test("vision: zero leakage val-test", len(va_h.intersection(te_h)) == 0)

    # 4b. FoodLens Auxiliary Splits & Combined Reconciliation Audit
    fl_tr_path = VISION_DIR / "food_lens_train_split.csv"
    fl_va_path = VISION_DIR / "food_lens_val_split.csv"
    fl_te_path = VISION_DIR / "food_lens_test_split.csv"
    comb_manifest = VISION_DIR / "combined_vision_manifest.csv"

    test("vision: food_lens_train_split.csv exists", fl_tr_path.exists())
    test("vision: food_lens_val_split.csv exists", fl_va_path.exists())
    test("vision: food_lens_test_split.csv exists", fl_te_path.exists())
    test("vision: combined_vision_manifest.csv exists", comb_manifest.exists())

    if fl_tr_path.exists() and fl_va_path.exists() and fl_te_path.exists():
        df_fl_tr = pd.read_csv(fl_tr_path)
        df_fl_va = pd.read_csv(fl_va_path)
        df_fl_te = pd.read_csv(fl_te_path)

        fl_tot = len(df_fl_tr) + len(df_fl_va) + len(df_fl_te)
        test("vision: food_lens active total is 2521", fl_tot == 2521, f"total={fl_tot}")
        test("vision: food_lens train split is 1764", len(df_fl_tr) == 1764)
        test("vision: food_lens val split is 375", len(df_fl_va) == 375)
        test("vision: food_lens test split is 382", len(df_fl_te) == 382)

        # Zero leakage check in FoodLens
        fl_tr_h = set(df_fl_tr["md5_hash"])
        fl_va_h = set(df_fl_va["md5_hash"])
        fl_te_h = set(df_fl_te["md5_hash"])
        test("vision: food_lens zero leakage train-val", len(fl_tr_h.intersection(fl_va_h)) == 0)
        test("vision: food_lens zero leakage train-test", len(fl_tr_h.intersection(fl_te_h)) == 0)
        test("vision: food_lens zero leakage val-test", len(fl_va_h.intersection(fl_te_h)) == 0)

        # Unique hash count audit: exactly 2520 unique hashes due to 1 within-train duplicate pair
        fl_all_unique_h = fl_tr_h.union(fl_va_h).union(fl_te_h)
        test("vision: food_lens active unique real hashes is 2520", len(fl_all_unique_h) == 2520, f"unique={len(fl_all_unique_h)}")
        dup_chappati_tr = df_fl_tr[df_fl_tr["image_identifier"].isin(["BUM71V45K4BA.jpg", "YVRKYNVI4KFD.jpg"])]
        test("vision: food_lens within-train duplicate pair verified", len(dup_chappati_tr) == 2 and dup_chappati_tr["md5_hash"].nunique() == 1)

        # Cross-dataset zero leakage check
        if train_split.exists():
            enh_all_h = set(pd.read_csv(train_split)["md5_hash"]).union(
                set(pd.read_csv(val_split)["md5_hash"])
            ).union(set(pd.read_csv(test_split)["md5_hash"]))
            test("vision: cross-dataset zero leakage (Enhanced vs FoodLens)", len(enh_all_h.intersection(fl_all_unique_h)) == 0)

    if comb_manifest.exists():
        df_comb = pd.read_csv(comb_manifest)
        test("vision: combined manifest total is 17925", len(df_comb) == 17925, f"total={len(df_comb)}")
        test("vision: combined manifest has benchmark_role column", "benchmark_role" in df_comb.columns)
        primary_cnt = (df_comb["benchmark_role"] == "primary_43class_benchmark").sum()
        aux_cnt = (df_comb["benchmark_role"] == "auxiliary_transfer_foodlens").sum()
        test("vision: primary 43-class count is 15404", primary_cnt == 15404, f"primary={primary_cnt}")
        test("vision: auxiliary foodlens count is 2521", aux_cnt == 2521, f"aux={aux_cnt}")
        test("vision: reconciliation exact (15404 + 2521 = 17925)", primary_cnt + aux_cnt == 17925)



    # -------------------------------------------------------------
    # 5. Nutrient-Gap & Suitability Scoring Engine
    # -------------------------------------------------------------
    bench_csv = FEATURE_DIR / "food_suitability_benchmarks.csv"
    profiles_json = FEATURE_DIR / "user_health_profiles.json"

    test("scoring: benchmarks csv exists", bench_csv.exists())
    test("scoring: health profiles json exists", profiles_json.exists())

    if bench_csv.exists():
        df_b = pd.read_csv(bench_csv)
        test("scoring: all foods scored", len(df_b) >= 50)
        
        # Check scores bounded in [1.0, 10.0]
        score_cols = [c for c in df_b.columns if c.startswith("score_")]
        test("scoring: >= 5 profiles scored", len(score_cols) >= 5)
        for sc in score_cols:
            test(f"scoring: {sc} bounded in [1.0, 10.0]", (df_b[sc] >= 1.0).all() and (df_b[sc] <= 10.0).all())

        # Clinical condition penalty checks
        # Hypertensive profile should strictly penalize high sodium
        if "score_hypertensive_adult" in df_b.columns:
            flagged_hypertension = df_b[df_b["sodium_flag"] == "high_sodium_unverified"]["score_hypertensive_adult"]
            if len(flagged_hypertension) > 0:
                test("scoring: hypertensive profile caps high-sodium dishes <= 2.0", (flagged_hypertension <= 2.0).all())

        # Diabetic profile should penalize simple sugar desserts (jalebi, ghevar)
        if "score_diabetic_adult" in df_b.columns:
            jalebi_score = df_b[df_b["class_name"] == "jalebi"]["score_diabetic_adult"].values
            if len(jalebi_score) > 0:
                test("scoring: diabetic profile penalizes jalebi (< 5.0)", jalebi_score[0] < 5.0, f"score={jalebi_score[0]}")

        # Missing nutrient values safety check
        from src.nutrient_scoring_engine import compute_suitability_score
        incomplete_food = {
            "calories_kcal": 200.0,
            "carbs_g": 25.0,
            "protein_g": 6.0,
            "fat_g": 8.0,
            "fibre_g": np.nan,
            "sodium_mg": np.nan,
            "calcium_mg": np.nan,
            "iron_mg": np.nan,
            "vitamin_c_mg": np.nan,
            "folate_ug": np.nan
        }
        test_score, test_verdict = compute_suitability_score(incomplete_food, profile_key="adult_male_sedentary")
        test("scoring: handles all-NaN micronutrients without failure", 1.0 <= test_score <= 10.0 and len(test_verdict) > 0,
             f"score={test_score}, verdict={test_verdict}")


    # -------------------------------------------------------------
    # 6. Recipe Recommendation Feature Store
    # -------------------------------------------------------------
    recipe_csv = FEATURE_DIR / "recipe_feature_store.csv"
    test("recipes: feature store csv exists", recipe_csv.exists())

    if recipe_csv.exists():
        df_rec = pd.read_csv(recipe_csv)
        test("recipes: exactly 5381 recipes present", len(df_rec) == 5381, f"count={len(df_rec)}")
        test("recipes: complexity_level column present", "complexity_level" in df_rec.columns)
        test("recipes: complexity levels valid (EASY, MODERATE, ADVANCED)", set(df_rec["complexity_level"].unique()).issubset({"EASY", "MODERATE", "ADVANCED"}))
        test("recipes: ingredient_tokens present and non-empty", (df_rec["ingredient_tokens"].dropna().str.len() > 0).all())

    # Summary
    passed = sum(1 for _, c, _ in test_results if c)
    failed = sum(1 for _, c, _ in test_results if not c)
    total = len(test_results)
    print("\n" + "=" * 60)
    print(f"PHASE 4 FEATURE ENGINEERING TESTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print("=" * 60)

    if failed > 0:
        print("FAILED tests:")
        for name, c, extra in test_results:
            if not c:
                print(f"  - {name}: {extra}")
        exit(1)
    else:
        print("All Phase 4 Feature Engineering tests passed.")


if __name__ == "__main__":
    run_tests()
