"""
src/nutrient_scoring_engine.py
Phase 4: Feature Engineering - Nutrient-Gap & Food Suitability Scoring Engine

Implements evidence-based dietary assessment and food suitability scoring:
1. ICMR-NIN (2020) Recommended Dietary Allowances (RDA) reference profiles:
   - adult_male_sedentary
   - adult_female_sedentary
   - diabetic_adult
   - hypertensive_adult
   - weight_management_adult
2. Multi-dimensional suitability scoring algorithm (Scale: 1.0 to 10.0):
   - Macronutrient balance contribution (Protein/Carb/Fat balance)
   - Dietary fiber & micronutrient bonus (Iron, Calcium, Vitamin C, Folate)
   - Caloric density adjustment
   - Sodium thresholding & safety penalty (including critical penalty for flagged high-sodium dishes)
   - Clinical condition penalties (Diabetes sugar penalty, Hypertension sodium penalty)
3. Computes suitability scores across all master nutrition KB foods for all health profiles.

Outputs:
- data/processed/feature_store/food_suitability_benchmarks.csv
- data/processed/feature_store/user_health_profiles.json
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_DIR = Path("data/processed/feature_store")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# ICMR-NIN 2020 Reference RDA Standards
ICMR_RDA_PROFILES = {
    "adult_male_sedentary": {
        "description": "Standard Adult Indian Male (Sedentary, Reference Weight 65kg)",
        "calories_kcal": 2110.0,
        "protein_g": 54.0,
        "carbs_g": 290.0,
        "fat_g": 25.0,
        "fibre_g": 30.0,
        "sodium_mg_max": 2000.0,
        "calcium_mg": 1000.0,
        "iron_mg": 19.0,
        "vitamin_c_mg": 80.0,
        "folate_ug": 300.0
    },
    "adult_female_sedentary": {
        "description": "Standard Adult Indian Female (Sedentary, Reference Weight 55kg)",
        "calories_kcal": 1660.0,
        "protein_g": 46.0,
        "carbs_g": 230.0,
        "fat_g": 20.0,
        "fibre_g": 25.0,
        "sodium_mg_max": 2000.0,
        "calcium_mg": 1000.0,
        "iron_mg": 29.0,  # Higher iron requirement
        "vitamin_c_mg": 65.0,
        "folate_ug": 220.0
    },
    "diabetic_adult": {
        "description": "Adult with Type-2 Diabetes (Controlled Carbohydrate, High Fiber)",
        "calories_kcal": 1800.0,
        "protein_g": 60.0,
        "carbs_g": 200.0,  # Controlled carbs
        "fat_g": 25.0,
        "fibre_g": 35.0,  # High fiber target
        "sodium_mg_max": 2000.0,
        "calcium_mg": 1000.0,
        "iron_mg": 20.0,
        "vitamin_c_mg": 80.0,
        "folate_ug": 300.0,
        "clinical_condition": "DIABETES"
    },
    "hypertensive_adult": {
        "description": "Adult with Hypertension (DASH Diet, Low Sodium)",
        "calories_kcal": 1900.0,
        "protein_g": 55.0,
        "carbs_g": 260.0,
        "fat_g": 25.0,
        "fibre_g": 30.0,
        "sodium_mg_max": 1500.0,  # Strict low sodium target
        "calcium_mg": 1200.0,
        "iron_mg": 20.0,
        "vitamin_c_mg": 80.0,
        "folate_ug": 300.0,
        "clinical_condition": "HYPERTENSION"
    },
    "weight_management": {
        "description": "Adult with Weight Management / Obesity Goals (Caloric Deficit, High Satiety)",
        "calories_kcal": 1500.0,  # Caloric restriction
        "protein_g": 65.0,  # High protein for lean mass preservation
        "carbs_g": 180.0,
        "fat_g": 20.0,
        "fibre_g": 35.0,  # High satiety fiber
        "sodium_mg_max": 2000.0,
        "calcium_mg": 1000.0,
        "iron_mg": 20.0,
        "vitamin_c_mg": 80.0,
        "folate_ug": 300.0,
        "clinical_condition": "WEIGHT_LOSS"
    }
}


def compute_suitability_score(food_row, profile_key="adult_male_sedentary"):
    """
    Computes evidence-based Food Suitability Score out of 10 for a single food item
    against a specific user profile based on ICMR-NIN nutritional standards.
    """
    profile = ICMR_RDA_PROFILES.get(profile_key, ICMR_RDA_PROFILES["adult_male_sedentary"])
    
    cal = float(food_row.get("calories_kcal", 0))
    carbs = float(food_row.get("carbs_g", 0))
    protein = float(food_row.get("protein_g", 0))
    fat = float(food_row.get("fat_g", 0))
    fibre = float(food_row.get("fibre_g", 0)) if pd.notnull(food_row.get("fibre_g")) else 1.0
    sodium = float(food_row.get("sodium_mg", 0)) if pd.notnull(food_row.get("sodium_mg")) else 200.0
    sod_flag = str(food_row.get("sodium_flag", "ok"))
    cal_flag = str(food_row.get("calorie_flag", "ok"))
    
    # Base score
    score = 7.0
    
    # 1. Protein Contribution (Target: >= 10-15% of energy from protein)
    if cal > 0:
        protein_cal_pct = (protein * 4) / cal * 100
        if protein_cal_pct >= 18.0:
            score += 1.2
        elif protein_cal_pct >= 12.0:
            score += 0.6
        elif protein_cal_pct < 6.0:
            score -= 0.6
            
    # 2. Dietary Fiber & Satiety
    if fibre >= 4.0:
        score += 1.0
    elif fibre >= 2.5:
        score += 0.5
    elif fibre < 1.0 and carbs > 30.0:
        score -= 0.6  # Refined carb penalty
        
    # 3. Healthy Fat vs Excessive Saturated/Frying Fat
    if cal > 0:
        fat_cal_pct = (fat * 9) / cal * 100
        if fat_cal_pct > 50.0:
            score -= 1.4  # Excessive fat penalty (e.g. deep fried sweets/bhature)
        elif 20.0 <= fat_cal_pct <= 35.0:
            score += 0.4  # Balanced healthy fat bonus
            
    # 4. Caloric Density & Portion Balance
    meal_cal_target = profile["calories_kcal"] * 0.33  # ~1/3 daily calories per main meal
    if cal > (meal_cal_target * 1.3):
        score -= 0.8  # Heavy meal deduction
        
    # 5. Sodium Penalty & Safety Flag Thresholding
    meal_sodium_target = profile["sodium_mg_max"] * 0.33
    if sodium > (meal_sodium_target * 1.5):
        score -= 1.2
    elif sodium > meal_sodium_target:
        score -= 0.6
        
    # CRITICAL: Flagged High Sodium (>5000mg unverified)
    if sod_flag == "high_sodium_unverified" or sodium > 5000.0:
        score -= 3.5  # Severe penalty for unverified extreme sodium
        
    # 6. Clinical Condition Specific Modifiers
    cond = profile.get("clinical_condition", None)
    if cond == "DIABETES":
        if carbs > 40.0:
            score -= 1.5  # High carb load penalty
        if fibre >= 4.0:
            score += 0.8  # Fiber bonus for glycemic blunting
        if "sweet" in str(food_row.get("display_name", "")).lower() or "jalebi" in str(food_row.get("class_name", "")).lower():
            score -= 3.0  # Simple sugar dessert penalty
            
    elif cond == "HYPERTENSION":
        if sodium > 500.0:
            score -= 2.0  # Heightened sodium penalty
        if sod_flag == "high_sodium_unverified":
            score = min(score, 2.0)  # Cap score at 2.0/10 for hypertensive users
            
    elif cond == "WEIGHT_LOSS":
        if cal > 350.0:
            score -= 1.5  # Caloric excess penalty
        if protein >= 10.0 and fibre >= 3.0:
            score += 1.0  # High satiety bonus
            
    # Clip final score strictly to [1.0, 10.0]
    final_score = round(max(1.0, min(10.0, score)), 1)
    
    # Qualitative Suitability Verdict
    if final_score >= 8.0:
        verdict = "HIGHLY_RECOMMENDED"
    elif final_score >= 6.5:
        verdict = "HEALTHY_CHOICE"
    elif final_score >= 4.5:
        verdict = "MODERATE_PORTION_CONTROL"
    else:
        verdict = "NOT_RECOMMENDED_LIMIT_INTAKE"
        
    return final_score, verdict


def evaluate_master_benchmarks():
    print("=" * 65)
    print("EVALUATING FOOD SUITABILITY BENCHMARKS ACROSS ALL PROFILES")
    print("=" * 65)

    kb_path = FEATURE_DIR / "master_food_nutrition_kb.csv"
    if not kb_path.exists():
        raise FileNotFoundError(f"Missing {kb_path}. Run master_nutrition_kb.py first.")

    df_kb = pd.read_csv(kb_path)
    print(f"Loaded {len(df_kb)} master food items.")

    benchmark_rows = []

    for _, row in df_kb.iterrows():
        c_name = row["class_name"]
        disp_name = row["display_name"]
        
        bench_entry = {
            "food_id": row["food_id"],
            "class_name": c_name,
            "display_name": disp_name,
            "match_type": row["match_type"],
            "calories_kcal": row["calories_kcal"],
            "protein_g": row["protein_g"],
            "carbs_g": row["carbs_g"],
            "fat_g": row["fat_g"],
            "sodium_mg": row["sodium_mg"],
            "sodium_flag": row["sodium_flag"],
            "calorie_flag": row["calorie_flag"],
        }
        
        # Calculate suitability score for each user profile
        for p_key in ICMR_RDA_PROFILES.keys():
            score, verdict = compute_suitability_score(row, profile_key=p_key)
            bench_entry[f"score_{p_key}"] = score
            bench_entry[f"verdict_{p_key}"] = verdict
            
        benchmark_rows.append(bench_entry)

    df_bench = pd.DataFrame(benchmark_rows)

    # Export
    bench_csv = FEATURE_DIR / "food_suitability_benchmarks.csv"
    profiles_json = FEATURE_DIR / "user_health_profiles.json"

    df_bench.to_csv(bench_csv, index=False)
    with open(profiles_json, "w", encoding="utf-8") as f:
        json.dump(ICMR_RDA_PROFILES, f, indent=2)

    print(f"\nSuitability Scoring Benchmarks Evaluated:")
    print(f"  - Total foods scored: {len(df_bench)}")
    print(f"  - Evaluated against {len(ICMR_RDA_PROFILES)} health profiles")
    print(f"  - Average Adult Male Score: {df_bench['score_adult_male_sedentary'].mean():.2f} / 10.0")
    print(f"  - Highest Scoring Foods: {', '.join(df_bench.sort_values(by='score_adult_male_sedentary', ascending=False)['class_name'].head(4))}")
    print(f"  - Lowest Scoring Foods:  {', '.join(df_bench.sort_values(by='score_adult_male_sedentary', ascending=True)['class_name'].head(4))}")
    
    print(f"\nOutputs saved:")
    print(f"  - Benchmark CSV: {bench_csv}")
    print(f"  - Health Profiles JSON: {profiles_json}")
    print("=" * 65)
    return df_bench


if __name__ == "__main__":
    evaluate_master_benchmarks()
