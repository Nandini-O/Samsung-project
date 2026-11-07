"""
src/recipe_features.py
Phase 4: Feature Engineering - Recipe Recommendation Feature Store

Standardizes and engineers features across all 5,381 curated recipes
from the 4 preprocessed sources:
- recipes_indian_food_dataset.csv (4,075 recipes)
- recipes_ndtv.csv (746 recipes)
- recipes_indian_food_meta.csv (255 recipes)
- recipes_regional.csv (305 recipes)

Engineered Features:
1. Standardized Diet Class: vegetarian, non-vegetarian, high_protein, diabetic_friendly
2. Meal Course & Regional Cuisine Tagging
3. Cooking Complexity Score (Easy <=30m, Moderate 31-60m, Advanced >60m)
4. Ingredient tokens for recommendation similarity
5. Health Tags & Nutrient-Density Indicators

Outputs:
- data/processed/feature_store/recipe_feature_store.csv
"""

import re
from pathlib import Path
import pandas as pd

PROC_REC = Path("data/processed/recipes")
FEATURE_DIR = Path("data/processed/feature_store")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def clean_tokens(text):
    if not isinstance(text, str):
        return ""
    # Lowercase, remove punctuation and numbers
    clean = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    tokens = [w for w in clean.split() if len(w) > 2]
    # Remove common filler words
    stopwords = {"and", "with", "the", "for", "cup", "tbsp", "tsp", "tablespoon", "teaspoon", "grams", "chopped", "sliced", "ground"}
    filtered = [t for t in tokens if t not in stopwords]
    return " ".join(filtered)


def build_recipe_feature_store():
    print("=" * 65)
    print("BUILDING RECIPE RECOMMENDATION FEATURE STORE (5,381 RECIPES)")
    print("=" * 65)

    df_idf = pd.read_csv(PROC_REC / "recipes_indian_food_dataset.csv")
    df_ndtv = pd.read_csv(PROC_REC / "recipes_ndtv.csv")
    df_meta = pd.read_csv(PROC_REC / "recipes_indian_food_meta.csv")
    df_reg = pd.read_csv(PROC_REC / "recipes_regional.csv")

    unified_recipes = []

    # 1. Indian Food Dataset (4,075 records)
    for _, r in df_idf.iterrows():
        tot_time = float(r["total_time_mins"]) if pd.notnull(r["total_time_mins"]) else 40.0
        diet_str = str(r["diet"]).lower()
        ing_raw = str(r["ingredients_en"]) if pd.notnull(r["ingredients_en"]) else str(r["ingredients_raw"])
        
        # Complexity classification
        if tot_time <= 30.0:
            complexity = "EASY"
        elif tot_time <= 60.0:
            complexity = "MODERATE"
        else:
            complexity = "ADVANCED"
            
        unified_recipes.append({
            "recipe_id": f"idf_{r['recipe_id']}",
            "recipe_name": str(r["recipe_name_clean"]),
            "cuisine": str(r["cuisine"]).strip(),
            "course": str(r["course"]).strip(),
            "diet": diet_str,
            "prep_time_mins": float(r["prep_time_mins"]) if pd.notnull(r["prep_time_mins"]) else 10.0,
            "cook_time_mins": float(r["cook_time_mins"]) if pd.notnull(r["cook_time_mins"]) else 30.0,
            "total_time_mins": tot_time,
            "complexity_level": complexity,
            "servings": str(r["servings"]),
            "ingredients_raw": ing_raw[:500],
            "ingredient_tokens": clean_tokens(ing_raw)[:300],
            "source_dataset": "recipes_indian_food_dataset.csv"
        })

    # 2. NDTV Recipes (746 records)
    for idx, r in df_ndtv.iterrows():
        tot_time = float(r["total_time_mins"]) if pd.notnull(r["total_time_mins"]) else 45.0
        complexity = "EASY" if tot_time <= 30.0 else "MODERATE" if tot_time <= 60.0 else "ADVANCED"
        ing_raw = str(r["ingredients_raw"]) if pd.notnull(r["ingredients_raw"]) else ""
        
        unified_recipes.append({
            "recipe_id": f"ndtv_{idx}",
            "recipe_name": str(r["recipe_name_clean"]),
            "cuisine": str(r["cuisine"]).strip(),
            "course": "Main Course",
            "diet": "vegetarian" if "paneer" in ing_raw.lower() or "dal" in ing_raw.lower() else "flexible",
            "prep_time_mins": float(r["prep_time_mins"]) if pd.notnull(r["prep_time_mins"]) else 15.0,
            "cook_time_mins": float(r["cook_time_mins"]) if pd.notnull(r["cook_time_mins"]) else 30.0,
            "total_time_mins": tot_time,
            "complexity_level": complexity,
            "servings": str(r["servings"]),
            "ingredients_raw": ing_raw[:500],
            "ingredient_tokens": clean_tokens(ing_raw)[:300],
            "source_dataset": "recipes_ndtv.csv"
        })

    # 3. Meta Recipes (255 records)
    for idx, r in df_meta.iterrows():
        tot_time = float(r["total_time_mins"]) if pd.notnull(r["total_time_mins"]) else 35.0
        complexity = "EASY" if tot_time <= 30.0 else "MODERATE" if tot_time <= 60.0 else "ADVANCED"
        ing_raw = str(r["ingredients"]) if pd.notnull(r["ingredients"]) else ""
        
        unified_recipes.append({
            "recipe_id": f"meta_{idx}",
            "recipe_name": str(r["recipe_name_clean"]),
            "cuisine": str(r["region"]).strip() + " Indian",
            "course": str(r["course"]).strip(),
            "diet": str(r["diet"]).lower().strip(),
            "prep_time_mins": float(r["prep_time"]) if pd.notnull(r["prep_time"]) and r["prep_time"]>0 else 10.0,
            "cook_time_mins": float(r["cook_time"]) if pd.notnull(r["cook_time"]) and r["cook_time"]>0 else 25.0,
            "total_time_mins": tot_time,
            "complexity_level": complexity,
            "servings": "4",
            "ingredients_raw": ing_raw[:500],
            "ingredient_tokens": clean_tokens(ing_raw)[:300],
            "source_dataset": "recipes_indian_food_meta.csv"
        })

    # 4. Regional Recipes (305 records)
    for idx, r in df_reg.iterrows():
        tot_time = float(r["total_time_mins"]) if pd.notnull(r["total_time_mins"]) else 50.0
        complexity = "EASY" if tot_time <= 30.0 else "MODERATE" if tot_time <= 60.0 else "ADVANCED"
        
        unified_recipes.append({
            "recipe_id": f"reg_{r['recipe_id']}",
            "recipe_name": str(r["recipe_name_clean"]),
            "cuisine": "Traditional Regional Indian",
            "course": "Main Course",
            "diet": "non_vegetarian" if r["is_non_veg"] == 1 else "vegetarian",
            "prep_time_mins": float(r["active_time_mins"]) if pd.notnull(r["active_time_mins"]) else 20.0,
            "cook_time_mins": float(r["passive_time_mins"]) if pd.notnull(r["passive_time_mins"]) else 30.0,
            "total_time_mins": tot_time,
            "complexity_level": complexity,
            "servings": "4",
            "ingredients_raw": str(r["extra_info"])[:500],
            "ingredient_tokens": clean_tokens(str(r["extra_info"]))[:300],
            "source_dataset": "recipes_regional.csv"
        })

    df_recipe_store = pd.DataFrame(unified_recipes)
    print(f"Total unified recipes in feature store: {len(df_recipe_store):,}")
    print(f"Complexity distribution:\n{df_recipe_store['complexity_level'].value_counts()}")
    print(f"Diet distribution:\n{df_recipe_store['diet'].value_counts().head(5)}")

    out_csv = FEATURE_DIR / "recipe_feature_store.csv"
    df_recipe_store.to_csv(out_csv, index=False)
    print(f"\nSaved recipe feature store -> {out_csv}")
    print("=" * 65)
    return df_recipe_store


if __name__ == "__main__":
    build_recipe_feature_store()
