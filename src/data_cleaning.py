"""
data_cleaning.py -- NutriSense AI Data Preprocessing

Cleans and standardises:
  1. Indian_Food_Nutrition_Processed.csv  -> primary nutrition DB (per serving, dish-level)
  2. IndianFoodCompositionTable_In_CSV_Format.csv -> ingredient-level IFCT data (per 100g)
  3. daily_food_nutrition_dataset.csv     -> generic food nutrition (per serving)
  4. packaged_foods_india.csv             -> packaged food nutrition (per serving/100g)
  5. IndianFoodDatasetCSV.csv             -> recipe + ingredient dataset
  6. INDIAN FOOD RECIPE.csv              -> NDTV recipe dataset
  7. indian_food.csv                      -> Indian food metadata
  8. recipes.csv                          -> regional recipe dataset
  9. food_lens image dataset             -> validates image files, generates class manifest

Rules:
  - Original files in data/raw/ are NEVER modified.
  - Cleaned outputs go to data/processed/nutrition/ and data/processed/recipes/.
  - All preprocessing decisions are logged to logs/data_cleaning.log.
  - No values are invented for missing data; NaN is preserved and documented.
"""

import re
import sys
import json
import shutil
import hashlib
import logging
from io import StringIO
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC_NUT = ROOT / "data" / "processed" / "nutrition"
PROC_REC = ROOT / "data" / "processed" / "recipes"
PROC_IMG = ROOT / "data" / "processed" / "images"
LOG_DIR = ROOT / "logs"

for d in (PROC_NUT, PROC_REC, PROC_IMG, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "data_cleaning.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared text utilities
# ---------------------------------------------------------------------------

def normalise_food_name(name: str) -> str:
    """Lowercase, strip, collapse whitespace, remove trailing punctuation."""
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^\w\s\-\/\(\)]", "", name)
    return name


def safe_read_csv(path, **kwargs):
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"Cannot read {path}")


def log_decision(msg: str):
    log.info("  [DECISION] %s", msg)


# ---------------------------------------------------------------------------
# 1. Indian_Food_Nutrition_Processed.csv
#    Source: curated dish-level nutrition, inferred per-serving
# ---------------------------------------------------------------------------

def clean_nutrition_processed():
    src = RAW / "Indian_Food_Nutrition_Processed.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)

    original_shape = df.shape
    log.info("  Original shape: %s", original_shape)

    # ---- 1. Rename columns for consistency ----
    col_map = {
        "Dish Name": "food_name",
        "Calories (kcal)": "calories_kcal",
        "Carbohydrates (g)": "carbs_g",
        "Protein (g)": "protein_g",
        "Fats (g)": "fat_g",
        "Free Sugar (g)": "free_sugar_g",
        "Fibre (g)": "fibre_g",
        "Sodium (mg)": "sodium_mg",
        "Calcium (mg)": "calcium_mg",
        "Iron (mg)": "iron_mg",
        "Vitamin C (mg)": "vitamin_c_mg",
        "Folate (µg)": "folate_ug",
    }
    df = df.rename(columns=col_map)
    log_decision("Renamed columns to snake_case with explicit units.")

    # ---- 2. Normalise food names ----
    df["food_name_raw"] = df["food_name"]
    df["food_name"] = df["food_name"].apply(normalise_food_name)

    # ---- 3. Remove completely blank rows ----
    n_before = len(df)
    df = df.dropna(subset=["food_name"])
    df = df[df["food_name"] != ""]
    n_after = len(df)
    if n_before != n_after:
        log_decision(f"Dropped {n_before - n_after} rows with blank food_name.")

    # ---- 4. Convert numeric cols, coerce bad values to NaN ----
    numeric_cols = [c for c in df.columns if c not in ("food_name", "food_name_raw")]
    for col in numeric_cols:
        before_miss = df[col].isnull().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after_miss = df[col].isnull().sum()
        if after_miss > before_miss:
            log_decision(f"{col}: {after_miss - before_miss} new NaN after coercion (non-numeric values).")

    # ---- 5. Range checks: flag suspicious values ----
    flags = []
    checks = {
        "calories_kcal": (0, 1500),
        "protein_g": (0, 100),
        "carbs_g": (0, 200),
        "fat_g": (0, 100),
        "fibre_g": (0, 50),
        "sodium_mg": (0, 5000),
        "iron_mg": (0, 30),
    }
    for col, (lo, hi) in checks.items():
        if col in df.columns:
            bad = df[(df[col] < lo) | (df[col] > hi)]
            if len(bad):
                flags.append({"column": col, "out_of_range": len(bad), "examples": bad["food_name"].head(3).tolist()})
                log_decision(f"{col}: {len(bad)} values outside [{lo}, {hi}] -- flagged but retained.")

    # ---- 6. Duplicates ----
    n_dup = df.duplicated(subset=["food_name"]).sum()
    if n_dup:
        log_decision(f"Found {n_dup} duplicate food_names -- keeping first occurrence.")
        df = df.drop_duplicates(subset=["food_name"], keep="first")

    # ---- 7. Add metadata ----
    df["source_file"] = "Indian_Food_Nutrition_Processed.csv"
    df["serving_reference"] = "per_serving_estimated"
    df["data_quality_note"] = np.where(
        df[["vitamin_c_mg", "folate_ug"]].isnull().any(axis=1),
        "vitamin_c or folate missing",
        "complete"
    )

    out = PROC_NUT / "nutrition_dish_level.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df.shape)
    if flags:
        flag_path = LOG_DIR / "nutrition_dish_level_flags.json"
        with open(flag_path, "w", encoding="utf-8") as fh:
            json.dump(flags, fh, indent=2)
        log.info("  Range flags saved -> %s", flag_path)
    return df


# ---------------------------------------------------------------------------
# 2. IndianFoodCompositionTable_In_CSV_Format.csv
#    Source: IFCT2017 -- ingredient-level, per 100g
# ---------------------------------------------------------------------------

def clean_ifct():
    src = RAW / "IndianFoodCompositionTable_In_CSV_Format.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)
    log.info("  Original shape: %s  cols=%s", df.shape, len(df.columns))

    # The IFCT has very complex column names like "Protein [g]|protcnt|PROXIMATE PRINCIPLES..."
    # We keep only the core nutritional columns relevant to the app.
    KEEP_PATTERNS = {
        "code": "food_code",
        "name": "food_name",
        "Moisture [g]": "moisture_g",
        "Protein [g]": "protein_g",
        "Total Fat [g]": "fat_g",
        "Carbohydrate [g]": "carbs_g",
        "Energy [kJ]": "energy_kj",
        "Dietary Fibre - Total [g]": "fibre_total_g",
        "Thiamine(B1) [mg]": "thiamine_b1_mg",
        "Riboflavin (B2) [mg]": "riboflavin_b2_mg",
        "Niacin (B3) [mg]": "niacin_b3_mg",
        "Total Ascorbic Acid [mg]": "vitamin_c_mg",
        "Total Folates (B9) [ug]": "folate_ug",
        "Calcium [mg]": "calcium_mg",
        "Iron [mg]": "iron_mg",
        "Phosphorus [mg]": "phosphorus_mg",
        "Zinc [mg]": "zinc_mg",
        "Total Saturated Fatty Acids (TSFA) [mg]": "saturated_fat_mg",
        "Cholesterol [mg]": "cholesterol_mg",
    }

    # Match columns by prefix
    col_mapping = {}
    for pattern, new_name in KEEP_PATTERNS.items():
        matched = [c for c in df.columns if c.startswith(pattern) or c == pattern]
        if matched:
            col_mapping[matched[0]] = new_name

    df_slim = df[list(col_mapping.keys())].rename(columns=col_mapping)
    log_decision(f"Selected {len(col_mapping)} core nutrient columns from {len(df.columns)} total IFCT columns.")

    # Normalise food names
    df_slim["food_name_raw"] = df_slim["food_name"]
    df_slim["food_name"] = df_slim["food_name"].apply(normalise_food_name)

    # IFCT values contain mean +/- std e.g. "9.20+-0.40" -- extract mean only
    def extract_mean(val):
        if pd.isnull(val):
            return np.nan
        s = str(val).strip()
        m = re.match(r"^([\d\.]+)", s)
        if m:
            return float(m.group(1))
        return np.nan

    numeric_cols = [c for c in df_slim.columns if c not in ("food_code", "food_name", "food_name_raw")]
    for col in numeric_cols:
        df_slim[col] = df_slim[col].apply(extract_mean)

    log_decision("Extracted mean from 'mean+-std' formatted IFCT values; std discarded.")

    # Energy: IFCT uses kJ, convert to kcal (1 kcal = 4.184 kJ)
    if "energy_kj" in df_slim.columns:
        df_slim["calories_kcal"] = (df_slim["energy_kj"] / 4.184).round(2)
        log_decision("Converted energy_kj to calories_kcal (1 kcal = 4.184 kJ).")

    # Remove rows with no name
    df_slim = df_slim[df_slim["food_name"].str.len() > 0]

    # Duplicates
    n_dup = df_slim.duplicated(subset=["food_name"]).sum()
    if n_dup:
        log_decision(f"Dropped {n_dup} duplicate food_names in IFCT.")
        df_slim = df_slim.drop_duplicates(subset=["food_name"], keep="first")

    df_slim["source_file"] = "IndianFoodCompositionTable_In_CSV_Format.csv"
    df_slim["serving_reference"] = "per_100g"

    out = PROC_NUT / "nutrition_ifct_ingredients.csv"
    df_slim.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df_slim.shape)
    return df_slim


# ---------------------------------------------------------------------------
# 3. daily_food_nutrition_dataset.csv
#    Generic food nutrition data, per serving
# ---------------------------------------------------------------------------

def clean_daily_nutrition():
    src = RAW / "daily_food_nutrition_dataset.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)
    log.info("  Original shape: %s", df.shape)

    col_map = {
        "Food_Item": "food_name",
        "Category": "category",
        "Calories (kcal)": "calories_kcal",
        "Protein (g)": "protein_g",
        "Carbohydrates (g)": "carbs_g",
        "Fat (g)": "fat_g",
        "Fiber (g)": "fibre_g",
        "Sugars (g)": "sugars_g",
        "Sodium (mg)": "sodium_mg",
        "Cholesterol (mg)": "cholesterol_mg",
        "Meal_Type": "meal_type",
        "Water_Intake (ml)": "water_ml",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["food_name_raw"] = df["food_name"]
    df["food_name"] = df["food_name"].apply(normalise_food_name)

    numeric_cols = [c for c in df.columns if c not in ("food_name", "food_name_raw", "category", "meal_type")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_blank = df["food_name"].eq("").sum()
    if n_blank:
        log_decision(f"Dropped {n_blank} blank food_name rows.")
        df = df[df["food_name"] != ""]

    n_dup = df.duplicated(subset=["food_name"]).sum()
    if n_dup:
        log_decision(f"Dropped {n_dup} duplicate food_name rows in daily_food_nutrition.")
        df = df.drop_duplicates(subset=["food_name"], keep="first")

    df["source_file"] = "daily_food_nutrition_dataset.csv"
    df["serving_reference"] = "per_serving"

    out = PROC_NUT / "nutrition_daily_general.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df.shape)
    return df


# ---------------------------------------------------------------------------
# 4. packaged_foods_india.csv
#    Packaged food nutrition -- per serving (Serving_Size_g column present)
# ---------------------------------------------------------------------------

def clean_packaged_foods():
    src = RAW / "packaged_foods_india.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)
    log.info("  Original shape: %s", df.shape)

    col_map = {
        "S.No": "serial_no",
        "Item name": "food_name",
        "Brand_Name": "brand",
        "Category": "category",
        "Sub_Category": "sub_category",
        "Ingredients": "ingredients",
        "Serving_Size_g": "serving_size_g",
        "Calories_kcal": "calories_kcal",
        "Carbohydrates_g": "carbs_g",
        "Proteins_g": "protein_g",
        "Total_Fat_g": "fat_g",
        "Saturated_Fat_g": "saturated_fat_g",
        "Trans_Fat_g": "trans_fat_g",
        "Sugar_g": "sugar_g",
        "Sodium_mg": "sodium_mg",
        "Dietary_Fiber_g": "fibre_g",
        "Cholesterol_mg": "cholesterol_mg",
        "Calcium_mg": "calcium_mg",
        "Vitamin": "vitamin",
        "Iron_mg": "iron_mg",
        "Price_INR": "price_inr",
        "Source_of_Data": "source",
        "Collection_Date": "collection_date",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df = df.drop(columns=["Collected_By", "Remarks"], errors="ignore")

    df["food_name_raw"] = df["food_name"]
    df["food_name"] = df["food_name"].apply(normalise_food_name)

    numeric_cols = [
        "serving_size_g", "calories_kcal", "carbs_g", "protein_g", "fat_g",
        "saturated_fat_g", "trans_fat_g", "sugar_g", "sodium_mg", "fibre_g",
        "cholesterol_mg", "calcium_mg", "iron_mg", "price_inr",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Flag records without serving size
    n_no_serving = df["serving_size_g"].isnull().sum() if "serving_size_g" in df.columns else 0
    if n_no_serving:
        log_decision(f"packaged_foods: {n_no_serving} rows missing serving_size_g -- retained with NaN.")

    # Compute per-100g calories for comparability
    if "serving_size_g" in df.columns and "calories_kcal" in df.columns:
        mask = df["serving_size_g"].notna() & (df["serving_size_g"] > 0)
        df.loc[mask, "calories_per_100g"] = (df.loc[mask, "calories_kcal"] / df.loc[mask, "serving_size_g"] * 100).round(2)
        log_decision("Computed calories_per_100g = (calories_kcal / serving_size_g) * 100 where serving_size_g > 0.")

    n_dup = df.duplicated(subset=["food_name", "brand"], keep="first").sum()
    if n_dup:
        log_decision(f"Dropped {n_dup} duplicate (food_name, brand) rows in packaged_foods.")
        df = df.drop_duplicates(subset=["food_name", "brand"], keep="first")

    df["source_file"] = "packaged_foods_india.csv"
    df["serving_reference"] = "per_serving_size_g"

    out = PROC_NUT / "nutrition_packaged_foods.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df.shape)
    return df


# ---------------------------------------------------------------------------
# 5. IndianFoodDatasetCSV.csv  -- Recipe dataset (Kaggle, multilingual)
# ---------------------------------------------------------------------------

def clean_indian_food_dataset():
    src = RAW / "IndianFoodDatasetCSV.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)
    log.info("  Original shape: %s", df.shape)

    # Keep only Indian cuisines
    INDIAN_CUISINES = {
        "Indian", "North Indian Recipes", "South Indian Recipes",
        "Bengali Recipes", "Maharashtrian Recipes", "Kerala Recipes",
        "Tamil Nadu", "Karnataka", "Rajasthani", "Punjabi", "Gujarat",
        "Andhra", "Gujarati Recipes", "Chettinad", "Goan Recipes",
        "Hyderabadi", "Kashmiri",
    }
    n_before = len(df)
    df = df[df["Cuisine"].isin(INDIAN_CUISINES)]
    log_decision(f"Filtered to Indian cuisines: {n_before} -> {len(df)} rows.")

    col_map = {
        "Srno": "recipe_id",
        "RecipeName": "recipe_name",
        "TranslatedRecipeName": "recipe_name_en",
        "Ingredients": "ingredients_raw",
        "TranslatedIngredients": "ingredients_en",
        "PrepTimeInMins": "prep_time_mins",
        "CookTimeInMins": "cook_time_mins",
        "TotalTimeInMins": "total_time_mins",
        "Servings": "servings",
        "Cuisine": "cuisine",
        "Course": "course",
        "Diet": "diet",
        "Instructions": "instructions",
        "TranslatedInstructions": "instructions_en",
        "URL": "source_url",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Normalise recipe names
    df["recipe_name_clean"] = df["recipe_name_en"].fillna(df["recipe_name"]).apply(normalise_food_name)

    # Drop rows with missing ingredients
    n_miss_ing = df["ingredients_raw"].isnull().sum()
    if n_miss_ing:
        log_decision(f"Dropping {n_miss_ing} rows with missing ingredients.")
        df = df.dropna(subset=["ingredients_raw"])

    # Time columns
    for col in ["prep_time_mins", "cook_time_mins", "total_time_mins", "servings"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalise diet labels
    diet_map = {
        "Vegetarian": "vegetarian",
        "High Protein Vegetarian": "vegetarian_high_protein",
        "Non Vegeterian": "non_vegetarian",
        "Eggetarian": "eggetarian",
        "Diabetic Friendly": "diabetic_friendly",
        "High Protein Non Vegetarian": "non_vegetarian_high_protein",
        "No Onion No Garlic (Sattvic)": "sattvic",
        "Vegan": "vegan",
        "Gluten Free": "gluten_free",
        "Sugar Free Diet": "sugar_free",
    }
    if "diet" in df.columns:
        df["diet"] = df["diet"].map(diet_map).fillna(df["diet"])

    # Duplicates
    n_dup = df.duplicated(subset=["recipe_name_clean"]).sum()
    if n_dup:
        log_decision(f"Dropped {n_dup} duplicate recipe names.")
        df = df.drop_duplicates(subset=["recipe_name_clean"], keep="first")

    df["source_file"] = "IndianFoodDatasetCSV.csv"

    out = PROC_REC / "recipes_indian_food_dataset.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df.shape)
    return df


# ---------------------------------------------------------------------------
# 6. INDIAN FOOD RECIPE.csv  -- NDTV recipe dataset
# ---------------------------------------------------------------------------

def clean_ndtv_recipes():
    src = RAW / "INDIAN FOOD RECIPE.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)
    log.info("  Original shape: %s", df.shape)

    col_map = {
        "RECIPE TITLE": "recipe_name",
        "TOTAL COOKING TIME": "total_time_raw",
        "PREPARATION TIME": "prep_time_raw",
        "COOKING TIME": "cook_time_raw",
        "RECIPE SERVINGS": "servings",
        "LEVEL": "difficulty",
        "INGREDIENTS": "ingredients_raw",
        "RECIPE LINK": "source_url",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["recipe_name_clean"] = df["recipe_name"].apply(normalise_food_name)

    # Parse time strings like "10 mins", "1 hr 30 mins"
    def parse_time_mins(s):
        if pd.isnull(s):
            return np.nan
        s = str(s).lower().strip()
        hrs = re.search(r"(\d+)\s*hr", s)
        mins = re.search(r"(\d+)\s*min", s)
        total = 0
        if hrs:
            total += int(hrs.group(1)) * 60
        if mins:
            total += int(mins.group(1))
        return float(total) if total > 0 else np.nan

    for col in ["total_time_raw", "prep_time_raw", "cook_time_raw"]:
        if col in df.columns:
            new_col = col.replace("_raw", "_mins")
            df[new_col] = df[col].apply(parse_time_mins)
    log_decision("Parsed time strings ('10 mins', '1 hr 30 mins') into integer minutes.")

    df["servings"] = pd.to_numeric(df["servings"], errors="coerce")

    # Log missing values (retain rows with missing times -- they're still useful)
    for col in ["prep_time_mins", "cook_time_mins", "total_time_mins", "servings"]:
        if col in df.columns:
            n = df[col].isnull().sum()
            if n:
                log_decision(f"ndtv_recipes: {n} rows missing {col} -- retained with NaN.")

    # Duplicates
    n_dup = df.duplicated(subset=["recipe_name_clean"]).sum()
    if n_dup:
        log_decision(f"Dropped {n_dup} duplicate recipe names in NDTV recipes.")
        df = df.drop_duplicates(subset=["recipe_name_clean"], keep="first")

    df["source_file"] = "INDIAN FOOD RECIPE.csv"
    df["cuisine"] = "indian"

    out = PROC_REC / "recipes_ndtv.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df.shape)
    return df


# ---------------------------------------------------------------------------
# 7. indian_food.csv -- metadata: diet, prep/cook time, course, region
# ---------------------------------------------------------------------------

def clean_indian_food_metadata():
    src = RAW / "indian_food.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)
    log.info("  Original shape: %s", df.shape)

    df["recipe_name_clean"] = df["name"].apply(normalise_food_name)

    for col in ["prep_time", "cook_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # -1 in prep/cook time indicates unknown
    df.loc[df["prep_time"] == -1, "prep_time"] = np.nan
    df.loc[df["cook_time"] == -1, "cook_time"] = np.nan
    log_decision("Replaced -1 in prep_time / cook_time with NaN (indicates unknown).")

    # Total time
    df["total_time_mins"] = df["prep_time"].add(df["cook_time"], fill_value=0)

    # region: one missing -- retain as NaN
    n_miss_region = df["region"].isnull().sum()
    if n_miss_region:
        log_decision(f"indian_food.csv: {n_miss_region} rows with missing region -- retained with NaN.")

    n_dup = df.duplicated(subset=["recipe_name_clean"]).sum()
    if n_dup:
        log_decision(f"Dropped {n_dup} duplicates in indian_food.csv.")
        df = df.drop_duplicates(subset=["recipe_name_clean"], keep="first")

    df["source_file"] = "indian_food.csv"

    out = PROC_REC / "recipes_indian_food_meta.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df.shape)
    return df


# ---------------------------------------------------------------------------
# 8. recipes.csv -- regional recipe dataset with cooking times in seconds
# ---------------------------------------------------------------------------

def clean_regional_recipes():
    src = RAW / "recipes.csv"
    log.info("=== Cleaning: %s ===", src.name)
    df = safe_read_csv(src)
    log.info("  Original shape: %s", df.shape)

    col_map = {
        "Recipe Id": "recipe_id",
        "Recipe Name English": "recipe_name",
        "Recipe Name Local": "recipe_name_local",
        "Is Non Veg": "is_non_veg",
        "Total Active Time": "active_time_sec",
        "Total Passive Time": "passive_time_sec",
        "Total Steps": "total_steps",
        "Warning": "warning",
        "Extra Info": "extra_info",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    df["recipe_name_clean"] = df["recipe_name"].apply(normalise_food_name)

    # Times are in seconds -- convert to minutes
    for col in ["active_time_sec", "passive_time_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "active_time_sec" in df.columns:
        df["active_time_mins"] = (df["active_time_sec"] / 60).round(1)
    if "passive_time_sec" in df.columns:
        df["passive_time_mins"] = (df["passive_time_sec"] / 60).round(1)
    if "active_time_mins" in df.columns and "passive_time_mins" in df.columns:
        df["total_time_mins"] = df["active_time_mins"].add(df["passive_time_mins"], fill_value=0)
    log_decision("Converted active/passive times from seconds to minutes.")

    # Extra info: 281/305 rows missing -- retain
    n_miss_extra = df["extra_info"].isnull().sum() if "extra_info" in df.columns else 0
    if n_miss_extra:
        log_decision(f"recipes.csv: {n_miss_extra} rows missing extra_info -- retained with NaN.")

    df["source_file"] = "recipes.csv"
    df["diet"] = df["is_non_veg"].map({True: "non_vegetarian", False: "vegetarian"}) if "is_non_veg" in df.columns else "unknown"

    n_dup = df.duplicated(subset=["recipe_id"]).sum() if "recipe_id" in df.columns else 0
    if n_dup:
        log_decision(f"Dropped {n_dup} duplicate recipe_id rows.")
        df = df.drop_duplicates(subset=["recipe_id"], keep="first")

    out = PROC_REC / "recipes_regional.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    log.info("  Saved -> %s  (shape: %s)", out, df.shape)
    return df


# ---------------------------------------------------------------------------
# 9. food_lens image dataset -- validate images, generate class manifest
# ---------------------------------------------------------------------------

def audit_and_manifest_food_lens():
    src = RAW / "food_lens" / "images.cv_rdi4no0tot5qp4nyjvgq6" / "data"
    log.info("=== Image Audit: food_lens ===")

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    manifest = []
    corrupt = []
    duplicates_found = []
    seen_hashes = {}
    total = 0
    valid = 0

    # Try importing Pillow for deeper validation
    try:
        from PIL import Image as PILImage, UnidentifiedImageError
        pil_available = True
    except ImportError:
        pil_available = False
        log.warning("  Pillow not installed -- skipping pixel-level corruption check.")

    for split in ["train", "val", "test"]:
        split_dir = src / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.replace("food indian_food ", "").replace("_", " ").strip()
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                total += 1
                size = img_path.stat().st_size
                is_corrupt = False
                width = height = None

                if size == 0:
                    is_corrupt = True
                    corrupt.append({"path": str(img_path), "reason": "zero_byte"})
                elif pil_available:
                    try:
                        with PILImage.open(img_path) as im:
                            im.verify()
                        with PILImage.open(img_path) as im:
                            width, height = im.size
                        valid += 1
                    except Exception as e:
                        is_corrupt = True
                        corrupt.append({"path": str(img_path), "reason": str(e)[:80]})
                else:
                    valid += 1

                # Near-duplicate detection via MD5 hash (file-level only)
                if not is_corrupt and size < 5_000_000:  # skip hashing huge files
                    try:
                        h = hashlib.md5(img_path.read_bytes()).hexdigest()
                        if h in seen_hashes:
                            duplicates_found.append({"file": str(img_path), "duplicate_of": seen_hashes[h]})
                        else:
                            seen_hashes[h] = str(img_path)
                    except Exception:
                        pass

                manifest.append({
                    "split": split,
                    "class": cls_name,
                    "filename": img_path.name,
                    "path": str(img_path),
                    "size_bytes": size,
                    "width": width,
                    "height": height,
                    "corrupt": is_corrupt,
                })

    manifest_df = pd.DataFrame(manifest)
    manifest_out = PROC_IMG / "food_lens_manifest.csv"
    manifest_df.to_csv(manifest_out, index=False)
    log.info("  Total images: %d | Valid: %d | Corrupt: %d | Duplicates: %d", total, valid, len(corrupt), len(duplicates_found))
    log.info("  Manifest saved -> %s", manifest_out)

    if corrupt:
        corrupt_out = LOG_DIR / "food_lens_corrupt_images.json"
        with open(corrupt_out, "w") as fh:
            json.dump(corrupt, fh, indent=2)
        log.info("  Corrupt image list -> %s", corrupt_out)
        log_decision(f"Found {len(corrupt)} corrupt images. They are listed in {corrupt_out} but NOT deleted from raw/.")

    if duplicates_found:
        dup_out = LOG_DIR / "food_lens_duplicate_images.json"
        with open(dup_out, "w") as fh:
            json.dump(duplicates_found, fh, indent=2)
        log.info("  Duplicate image list -> %s", dup_out)
        log_decision(f"Found {len(duplicates_found)} exact-duplicate images (MD5). Listed but NOT deleted from raw/.")

    # Class manifest for model training
    class_manifest = (
        manifest_df[~manifest_df["corrupt"]]
        .groupby(["split", "class"])
        .size()
        .reset_index(name="image_count")
    )
    class_manifest_out = PROC_IMG / "food_lens_class_manifest.csv"
    class_manifest.to_csv(class_manifest_out, index=False)
    log.info("  Class manifest -> %s", class_manifest_out)

    return manifest_df


# ---------------------------------------------------------------------------
# 10. enhanced-indian-food-classification parquet -- generate label manifest
# ---------------------------------------------------------------------------

def generate_enhanced_label_manifest():
    src = RAW / "enhanced-indian-food-classification"
    log.info("=== Label Manifest: enhanced-indian-food-classification ===")

    files = sorted(src.glob("*.parquet"))
    rows = []
    for f in files:
        split_key = f.stem.split("-")[0]
        try:
            df = pd.read_parquet(f, columns=["label"])
            counts = df["label"].value_counts()
            for lbl, cnt in counts.items():
                rows.append({"split": split_key, "label": lbl, "count": int(cnt)})
        except Exception as e:
            log.warning("  Skipped %s: %s", f.name, e)

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        pivot = manifest.pivot_table(index="label", columns="split", values="count", aggfunc="sum").fillna(0).astype(int)
        pivot["total"] = pivot.sum(axis=1)
        out = PROC_IMG / "enhanced_indian_food_label_manifest.csv"
        pivot.reset_index().to_csv(out, index=False)
        log.info("  Label manifest -> %s  (%d unique labels)", out, len(pivot))
    return manifest


# ---------------------------------------------------------------------------
# 11. Build unified nutrition lookup table
#     Merge dish-level + IFCT ingredient data into one searchable CSV
# ---------------------------------------------------------------------------

def build_unified_nutrition_db(df_dish, df_ifct):
    log.info("=== Building unified nutrition lookup ===")

    # Standardise columns to merge
    CORE_COLS = ["food_name", "food_name_raw", "calories_kcal", "protein_g", "carbs_g", "fat_g", "fibre_g", "sodium_mg", "calcium_mg", "iron_mg", "source_file", "serving_reference"]

    def subset(df):
        return df[[c for c in CORE_COLS if c in df.columns]].copy()

    dfs = [subset(df_dish), subset(df_ifct)]
    merged = pd.concat(dfs, ignore_index=True)

    # Prefer dish-level data over IFCT for duplicate food names
    merged = merged.drop_duplicates(subset=["food_name"], keep="first")

    merged["food_name_normalised"] = merged["food_name"].apply(normalise_food_name)

    out = PROC_NUT / "nutrition_unified_lookup.csv"
    merged.to_csv(out, index=False, encoding="utf-8")
    log.info("  Unified lookup -> %s  (%d records)", out, len(merged))
    log_decision(f"Unified nutrition DB: dish-level takes priority over IFCT for {len(df_dish)} dishes. IFCT adds {len(merged) - len(df_dish)} ingredient entries.")
    return merged


# ---------------------------------------------------------------------------
# Run all cleaning steps
# ---------------------------------------------------------------------------

def run_all():
    start = datetime.now()
    log.info("\n" + "=" * 60)
    log.info("NutriSense AI -- Data Cleaning Pipeline")
    log.info("Started: %s", start.isoformat())
    log.info("=" * 60)

    results = {}

    try:
        results["nutrition_dish_level"] = clean_nutrition_processed()
    except Exception as e:
        log.error("FAILED: clean_nutrition_processed -- %s", e)
        results["nutrition_dish_level"] = None

    try:
        results["ifct"] = clean_ifct()
    except Exception as e:
        log.error("FAILED: clean_ifct -- %s", e)
        results["ifct"] = None

    try:
        results["daily_nutrition"] = clean_daily_nutrition()
    except Exception as e:
        log.error("FAILED: clean_daily_nutrition -- %s", e)
        results["daily_nutrition"] = None

    try:
        results["packaged_foods"] = clean_packaged_foods()
    except Exception as e:
        log.error("FAILED: clean_packaged_foods -- %s", e)
        results["packaged_foods"] = None

    try:
        results["recipes_full"] = clean_indian_food_dataset()
    except Exception as e:
        log.error("FAILED: clean_indian_food_dataset -- %s", e)
        results["recipes_full"] = None

    try:
        results["recipes_ndtv"] = clean_ndtv_recipes()
    except Exception as e:
        log.error("FAILED: clean_ndtv_recipes -- %s", e)
        results["recipes_ndtv"] = None

    try:
        results["recipes_meta"] = clean_indian_food_metadata()
    except Exception as e:
        log.error("FAILED: clean_indian_food_metadata -- %s", e)
        results["recipes_meta"] = None

    try:
        results["recipes_regional"] = clean_regional_recipes()
    except Exception as e:
        log.error("FAILED: clean_regional_recipes -- %s", e)
        results["recipes_regional"] = None

    try:
        results["food_lens_manifest"] = audit_and_manifest_food_lens()
    except Exception as e:
        log.error("FAILED: audit_and_manifest_food_lens -- %s", e)
        results["food_lens_manifest"] = None

    try:
        generate_enhanced_label_manifest()
    except Exception as e:
        log.error("FAILED: generate_enhanced_label_manifest -- %s", e)

    # Unified nutrition DB
    if results.get("nutrition_dish_level") is not None and results.get("ifct") is not None:
        try:
            build_unified_nutrition_db(results["nutrition_dish_level"], results["ifct"])
        except Exception as e:
            log.error("FAILED: build_unified_nutrition_db -- %s", e)

    elapsed = (datetime.now() - start).total_seconds()
    log.info("\n" + "=" * 60)
    log.info("Cleaning complete in %.1f seconds", elapsed)
    log.info("Output directories:")
    log.info("  Nutrition: %s", PROC_NUT)
    log.info("  Recipes:   %s", PROC_REC)
    log.info("  Images:    %s", PROC_IMG)
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    run_all()
