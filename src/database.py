"""
src/database.py
===============
Phase 10 — NutriSense AI Samsung Project
Backend Database Integration & SQLite Repository

Provides:
 1. SQLite schema initialization for user profiles, foods, nutrients, recipes,
    ingredients, allergens, meal logs, meal plans, scores history, and user feedback.
 2. Seamless automatic data migration & seeding from existing CSV/JSON knowledge bases
    (master_food_nutrition_kb.csv, recipe_feature_store.csv, profiles.json).
 3. Parameterized SQL queries preventing injection, with input validation, transaction
    management, and robust error handling.
 4. Complete Meal Diary CRUD (logging meals, retrieving history, calculating daily totals
    and progress toward ICMR-NIN RDA targets).
 5. Persistence of AI-generated meal plans, recommendation feedback, and scoring audits.
 6. Clean repository abstraction enabling end-to-end data flow between vision inference,
    nutrition analysis, profiling, scoring, and recommendation systems.

Author: NutriSense AI Team
Phase: 10
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Workspace Paths & Database Configuration
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_DB_DIR = _WORKSPACE_ROOT / "data" / "database"
_DEFAULT_DB_FILE = _DB_DIR / "nutrisense.db"

_PROCESSED_FEATURE_DIR = _WORKSPACE_ROOT / "data" / "processed" / "feature_store"
_FOOD_KB_CSV = _PROCESSED_FEATURE_DIR / "master_food_nutrition_kb.csv"
_RECIPES_CSV = _PROCESSED_FEATURE_DIR / "recipe_feature_store.csv"
_LEGACY_PROFILES_JSON = _WORKSPACE_ROOT / "data" / "user_profiles" / "profiles.json"


def _now_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def get_db_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolves target database path, creating parent directory if necessary."""
    p = Path(custom_path) if custom_path else _DEFAULT_DB_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """
    Creates and configures an SQLite database connection.
    Enables WAL mode for concurrency, enforces foreign keys, and sets Row factory.
    """
    path = get_db_path(db_path)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


# ---------------------------------------------------------------------------
# Database Schema Definitions
# ---------------------------------------------------------------------------

SCHEMA_STATEMENTS = [
    # 1. User Profiles
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        height_cm REAL,
        weight_kg REAL,
        dietary_goal TEXT,
        activity_level TEXT,
        dietary_preference TEXT,
        health_conditions TEXT,
        allergies TEXT,
        morning_routine TEXT,
        budget TEXT,
        cooking_time_minutes INTEGER,
        breakfast_preference TEXT,
        dietary_notes TEXT,
        daily_calorie_target REAL,
        is_active INTEGER DEFAULT 0,
        profile_version INTEGER DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,

    # 2. Master Food Nutrition Knowledge Base
    """
    CREATE TABLE IF NOT EXISTS foods (
        food_id TEXT PRIMARY KEY,
        class_name TEXT UNIQUE NOT NULL,
        display_name TEXT NOT NULL,
        match_type TEXT,
        serving_size_g REAL,
        serving_unit TEXT,
        calories_kcal REAL NOT NULL,
        carbs_g REAL NOT NULL,
        protein_g REAL NOT NULL,
        fat_g REAL NOT NULL,
        fibre_g REAL,
        sodium_mg REAL,
        calcium_mg REAL,
        iron_mg REAL,
        vitamin_c_mg REAL,
        folate_ug REAL,
        sodium_flag TEXT,
        calorie_flag TEXT,
        data_source TEXT,
        confidence TEXT,
        has_subtypes INTEGER DEFAULT 0,
        created_at TEXT NOT NULL
    );
    """,

    # 3. Curated Indian & Regional Recipes
    """
    CREATE TABLE IF NOT EXISTS recipes (
        recipe_id TEXT PRIMARY KEY,
        recipe_name TEXT NOT NULL,
        cuisine TEXT,
        course TEXT,
        diet TEXT,
        prep_time_mins REAL,
        cook_time_mins REAL,
        total_time_mins REAL,
        complexity_level TEXT,
        servings INTEGER,
        ingredients_raw TEXT,
        ingredient_tokens TEXT,
        source_dataset TEXT,
        created_at TEXT NOT NULL
    );
    """,

    # 4. Recipe Ingredients Breakdown
    """
    CREATE TABLE IF NOT EXISTS recipe_ingredients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recipe_id TEXT NOT NULL,
        ingredient_name TEXT NOT NULL,
        raw_text TEXT,
        FOREIGN KEY (recipe_id) REFERENCES recipes(recipe_id) ON DELETE CASCADE
    );
    """,

    # 5. Master Allergens
    """
    CREATE TABLE IF NOT EXISTS allergens (
        allergen_id TEXT PRIMARY KEY,
        allergen_name TEXT UNIQUE NOT NULL,
        description TEXT
    );
    """,

    # 6. Food Allergens Mapping
    """
    CREATE TABLE IF NOT EXISTS food_allergens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_name TEXT NOT NULL,
        allergen_name TEXT NOT NULL,
        is_declared INTEGER DEFAULT 1,
        FOREIGN KEY (class_name) REFERENCES foods(class_name) ON DELETE CASCADE
    );
    """,

    # 7. Meal Logs (Daily Diet Diary)
    """
    CREATE TABLE IF NOT EXISTS meal_logs (
        log_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        meal_date TEXT NOT NULL,
        meal_time TEXT,
        meal_slot TEXT NOT NULL,
        food_name TEXT NOT NULL,
        recipe_id TEXT,
        portion_g REAL NOT NULL,
        portion_description TEXT,
        calories REAL NOT NULL,
        protein_g REAL NOT NULL,
        carbs_g REAL NOT NULL,
        fat_g REAL NOT NULL,
        fibre_g REAL,
        sodium_mg REAL,
        suitability_score REAL,
        source_type TEXT,
        image_path TEXT,
        notes TEXT,
        logged_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
    );
    """,

    # 8. Persisted Meal Plans (Daily & Weekly)
    """
    CREATE TABLE IF NOT EXISTS meal_plans (
        plan_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        plan_type TEXT NOT NULL,
        target_date TEXT,
        target_calories REAL,
        plan_payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE CASCADE
    );
    """,

    # 9. User Feedback & Model Corrections
    """
    CREATE TABLE IF NOT EXISTS user_feedback (
        feedback_id TEXT PRIMARY KEY,
        user_id TEXT,
        feedback_type TEXT NOT NULL,
        item_reference TEXT,
        rating INTEGER,
        user_corrected_name TEXT,
        comment TEXT,
        submitted_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE SET NULL
    );
    """,

    # 10. Suitability Scores History & Audit Log
    """
    CREATE TABLE IF NOT EXISTS scores_history (
        score_id TEXT PRIMARY KEY,
        user_id TEXT,
        food_name TEXT NOT NULL,
        suitability_score REAL NOT NULL,
        evaluation_summary TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id) ON DELETE SET NULL
    );
    """,

    # Performance Indexes
    "CREATE INDEX IF NOT EXISTS idx_meal_logs_user_date ON meal_logs(user_id, meal_date);",
    "CREATE INDEX IF NOT EXISTS idx_meal_logs_slot ON meal_logs(meal_slot);",
    "CREATE INDEX IF NOT EXISTS idx_foods_class ON foods(class_name);",
    "CREATE INDEX IF NOT EXISTS idx_recipes_course ON recipes(course);",
    "CREATE INDEX IF NOT EXISTS idx_recipes_diet ON recipes(diet);"
]


def init_db(db_path: Optional[Union[str, Path]] = None) -> bool:
    """Initializes all database tables and indexes."""
    conn = get_connection(db_path)
    try:
        with conn:
            for stmt in SCHEMA_STATEMENTS:
                conn.execute(stmt)
        return True
    except sqlite3.Error as e:
        print(f"ERROR: Database initialization failed: {e}")
        return False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Initial Data Seeding & Migration Helpers
# ---------------------------------------------------------------------------

def seed_initial_data(db_path: Optional[Union[str, Path]] = None) -> Dict[str, int]:
    """
    Populates database from existing CSV/JSON files if tables are empty.
    Preserves any existing data without overwriting.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    seeded_counts = {"foods": 0, "recipes": 0, "allergens": 0, "profiles": 0}

    try:
        with conn:
            # 1. Seed Master Foods Knowledge Base
            cur = conn.execute("SELECT COUNT(*) FROM foods")
            if cur.fetchone()[0] == 0 and _FOOD_KB_CSV.exists():
                import pandas as pd
                df_food = pd.read_csv(_FOOD_KB_CSV)
                now_str = _now_iso()
                for _, r in df_food.iterrows():
                    conn.execute("""
                        INSERT INTO foods (
                            food_id, class_name, display_name, match_type, serving_size_g,
                            serving_unit, calories_kcal, carbs_g, protein_g, fat_g,
                            fibre_g, sodium_mg, calcium_mg, iron_mg, vitamin_c_mg,
                            folate_ug, sodium_flag, calorie_flag, data_source,
                            confidence, has_subtypes, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        str(r.get("food_id") or uuid.uuid4().hex[:12]),
                        str(r.get("class_name", "")),
                        str(r.get("display_name", "")),
                        str(r.get("match_type", "VALID_DIRECT")),
                        float(r.get("serving_size_g")) if pd.notnull(r.get("serving_size_g")) else 100.0,
                        str(r.get("serving_unit", "100g")),
                        float(r.get("calories_kcal", 0.0)),
                        float(r.get("carbs_g", 0.0)),
                        float(r.get("protein_g", 0.0)),
                        float(r.get("fat_g", 0.0)),
                        float(r.get("fibre_g")) if pd.notnull(r.get("fibre_g")) else None,
                        float(r.get("sodium_mg")) if pd.notnull(r.get("sodium_mg")) else None,
                        float(r.get("calcium_mg")) if pd.notnull(r.get("calcium_mg")) else None,
                        float(r.get("iron_mg")) if pd.notnull(r.get("iron_mg")) else None,
                        float(r.get("vitamin_c_mg")) if pd.notnull(r.get("vitamin_c_mg")) else None,
                        float(r.get("folate_ug")) if pd.notnull(r.get("folate_ug")) else None,
                        str(r.get("sodium_flag", "ok")),
                        str(r.get("calorie_flag", "ok")),
                        str(r.get("data_source", "ICMR_IFCT_2017")),
                        str(r.get("confidence", "HIGH")),
                        1 if bool(r.get("has_subtypes", False)) else 0,
                        now_str
                    ))
                    seeded_counts["foods"] += 1

            # 2. Seed Master Recipes
            cur = conn.execute("SELECT COUNT(*) FROM recipes")
            if cur.fetchone()[0] == 0 and _RECIPES_CSV.exists():
                import pandas as pd
                df_rec = pd.read_csv(_RECIPES_CSV)
                now_str = _now_iso()
                for _, r in df_rec.iterrows():
                    rec_id = str(r.get("recipe_id") or uuid.uuid4().hex[:12])
                    conn.execute("""
                        INSERT INTO recipes (
                            recipe_id, recipe_name, cuisine, course, diet,
                            prep_time_mins, cook_time_mins, total_time_mins,
                            complexity_level, servings, ingredients_raw,
                            ingredient_tokens, source_dataset, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        rec_id,
                        str(r.get("recipe_name", "")),
                        str(r.get("cuisine", "Indian")),
                        str(r.get("course", "Main Course")),
                        str(r.get("diet", "vegetarian")),
                        float(r.get("prep_time_mins")) if pd.notnull(r.get("prep_time_mins")) else 15.0,
                        float(r.get("cook_time_mins")) if pd.notnull(r.get("cook_time_mins")) else 20.0,
                        float(r.get("total_time_mins")) if pd.notnull(r.get("total_time_mins")) else 35.0,
                        str(r.get("complexity_level", "MODERATE")),
                        int(r.get("servings")) if pd.notnull(r.get("servings")) else 4,
                        str(r.get("ingredients_raw", "")),
                        str(r.get("ingredient_tokens", "")),
                        str(r.get("source_dataset", "recipes_indian_food_dataset.csv")),
                        now_str
                    ))
                    seeded_counts["recipes"] += 1

            # 3. Seed Master Allergens
            cur = conn.execute("SELECT COUNT(*) FROM allergens")
            if cur.fetchone()[0] == 0:
                allergens_list = [
                    ("gluten", "Gluten (Wheat, Barley, Rye, Semolina, Maida)"),
                    ("dairy", "Dairy Products (Milk, Paneer, Ghee, Butter, Curd, Cream)"),
                    ("nuts", "Tree Nuts & Peanuts (Cashew, Almond, Peanut, Walnut)"),
                    ("soy", "Soy & Soy Derivatives (Soya chunks, Tofu, Soy Sauce)"),
                    ("seafood", "Fish, Shellfish, Prawns, Shrimp, Crab"),
                    ("eggs", "Poultry Eggs & Egg whites"),
                    ("sesame", "Sesame seeds (Til) & Tahini"),
                    ("mustard", "Mustard seeds (Sarson, Rai) & Mustard oil")
                ]
                for a_id, desc in allergens_list:
                    conn.execute(
                        "INSERT INTO allergens (allergen_id, allergen_name, description) VALUES (?, ?, ?)",
                        (a_id, a_id.title(), desc)
                    )
                    seeded_counts["allergens"] += 1

            # 4. Migrate Existing JSON Profiles if Database is empty
            cur = conn.execute("SELECT COUNT(*) FROM user_profiles")
            if cur.fetchone()[0] == 0 and _LEGACY_PROFILES_JSON.exists():
                try:
                    with open(_LEGACY_PROFILES_JSON, "r", encoding="utf-8") as f:
                        profiles_data = json.load(f)
                    active_id = profiles_data.get("active_profile_id")
                    for uid, p in profiles_data.get("profiles", {}).items():
                        is_active_val = 1 if uid == active_id else 0
                        conn.execute("""
                            INSERT INTO user_profiles (
                                user_id, display_name, age, gender, height_cm, weight_kg,
                                dietary_goal, activity_level, dietary_preference,
                                health_conditions, allergies, morning_routine, budget,
                                cooking_time_minutes, breakfast_preference, dietary_notes,
                                daily_calorie_target, is_active, profile_version,
                                created_at, updated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            uid,
                            str(p.get("display_name", "User")),
                            p.get("age"),
                            p.get("gender", "prefer_not_to_say"),
                            p.get("height_cm"),
                            p.get("weight_kg"),
                            p.get("dietary_goal", "general_wellness"),
                            p.get("activity_level", "sedentary"),
                            p.get("dietary_preference", "vegetarian"),
                            json.dumps(p.get("health_conditions", [])),
                            json.dumps(p.get("allergies", [])),
                            p.get("morning_routine", "regular"),
                            p.get("budget", "medium"),
                            p.get("cooking_time_minutes", 30),
                            json.dumps(p.get("breakfast_preference", [])),
                            p.get("dietary_notes", ""),
                            p.get("daily_calorie_target", 2000.0),
                            is_active_val,
                            p.get("profile_version", 1),
                            p.get("created_at") or _now_iso(),
                            p.get("updated_at") or _now_iso()
                        ))
                        seeded_counts["profiles"] += 1
                except Exception as e:
                    print(f"WARNING: Could not migrate legacy profiles.json: {e}")

    except sqlite3.Error as e:
        print(f"ERROR: Seeding database failed: {e}")
    finally:
        conn.close()

    return seeded_counts


# ---------------------------------------------------------------------------
# User Profile Database Operations (Repository Pattern)
# ---------------------------------------------------------------------------

def db_save_profile(profile_dict: Dict[str, Any], db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Inserts or updates a user profile with parameterized SQL."""
    init_db(db_path)
    conn = get_connection(db_path)

    uid = profile_dict.get("user_id") or uuid.uuid4().hex
    display_name = str(profile_dict.get("display_name", "")).strip()
    if not display_name:
        raise ValueError("Profile display_name is required and cannot be empty.")

    now = _now_iso()
    created_at = profile_dict.get("created_at") or now
    updated_at = now
    version = int(profile_dict.get("profile_version", 1))

    try:
        with conn:
            # Check if this should be active
            cur = conn.execute("SELECT COUNT(*) FROM user_profiles WHERE is_active = 1")
            has_active = cur.fetchone()[0] > 0
            is_active = 1 if (not has_active or bool(profile_dict.get("is_active"))) else 0

            conn.execute("""
                INSERT INTO user_profiles (
                    user_id, display_name, age, gender, height_cm, weight_kg,
                    dietary_goal, activity_level, dietary_preference,
                    health_conditions, allergies, morning_routine, budget,
                    cooking_time_minutes, breakfast_preference, dietary_notes,
                    daily_calorie_target, is_active, profile_version,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    age = excluded.age,
                    gender = excluded.gender,
                    height_cm = excluded.height_cm,
                    weight_kg = excluded.weight_kg,
                    dietary_goal = excluded.dietary_goal,
                    activity_level = excluded.activity_level,
                    dietary_preference = excluded.dietary_preference,
                    health_conditions = excluded.health_conditions,
                    allergies = excluded.allergies,
                    morning_routine = excluded.morning_routine,
                    budget = excluded.budget,
                    cooking_time_minutes = excluded.cooking_time_minutes,
                    breakfast_preference = excluded.breakfast_preference,
                    dietary_notes = excluded.dietary_notes,
                    daily_calorie_target = excluded.daily_calorie_target,
                    is_active = excluded.is_active,
                    profile_version = user_profiles.profile_version + 1,
                    updated_at = excluded.updated_at
            """, (
                uid,
                display_name,
                profile_dict.get("age"),
                profile_dict.get("gender", "prefer_not_to_say"),
                profile_dict.get("height_cm"),
                profile_dict.get("weight_kg"),
                profile_dict.get("dietary_goal", "general_wellness"),
                profile_dict.get("activity_level", "sedentary"),
                profile_dict.get("dietary_preference", "vegetarian"),
                json.dumps(profile_dict.get("health_conditions", [])),
                json.dumps(profile_dict.get("allergies", [])),
                profile_dict.get("morning_routine", "regular"),
                profile_dict.get("budget", "medium"),
                profile_dict.get("cooking_time_minutes", 30),
                json.dumps(profile_dict.get("breakfast_preference", [])),
                profile_dict.get("dietary_notes", ""),
                profile_dict.get("daily_calorie_target", 2000.0),
                is_active,
                version,
                created_at,
                updated_at
            ))

        return db_get_profile(uid, db_path=db_path)  # type: ignore
    finally:
        conn.close()


def db_get_profile(user_id: str, db_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Retrieves single profile by user_id with parsed JSON arrays."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return None

        p = dict(row)
        p["health_conditions"] = json.loads(p.get("health_conditions") or "[]")
        p["allergies"] = json.loads(p.get("allergies") or "[]")
        p["breakfast_preference"] = json.loads(p.get("breakfast_preference") or "[]")
        p["is_active"] = bool(p.get("is_active"))
        return p
    finally:
        conn.close()


def db_list_profiles(db_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Lists all saved user profiles."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM user_profiles ORDER BY updated_at DESC")
        results = []
        for row in cur.fetchall():
            p = dict(row)
            p["health_conditions"] = json.loads(p.get("health_conditions") or "[]")
            p["allergies"] = json.loads(p.get("allergies") or "[]")
            p["breakfast_preference"] = json.loads(p.get("breakfast_preference") or "[]")
            p["is_active"] = bool(p.get("is_active"))
            results.append(p)
        return results
    finally:
        conn.close()


def db_delete_profile(user_id: str, db_path: Optional[Union[str, Path]] = None) -> bool:
    """Deletes profile by ID."""
    conn = get_connection(db_path)
    try:
        with conn:
            cur = conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            return cur.rowcount > 0
    finally:
        conn.close()


def db_get_active_profile(db_path: Optional[Union[str, Path]] = None) -> Optional[Dict[str, Any]]:
    """Returns the currently active profile, or the most recent profile if none marked active."""
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM user_profiles WHERE is_active = 1 LIMIT 1")
        row = cur.fetchone()
        if not row:
            # Fallback to most recently updated profile
            cur = conn.execute("SELECT * FROM user_profiles ORDER BY updated_at DESC LIMIT 1")
            row = cur.fetchone()
        if not row:
            return None

        p = dict(row)
        p["health_conditions"] = json.loads(p.get("health_conditions") or "[]")
        p["allergies"] = json.loads(p.get("allergies") or "[]")
        p["breakfast_preference"] = json.loads(p.get("breakfast_preference") or "[]")
        p["is_active"] = bool(p.get("is_active"))
        return p
    finally:
        conn.close()


def db_set_active_profile(user_id: str, db_path: Optional[Union[str, Path]] = None) -> bool:
    """Marks specified profile as active and deactivates all others."""
    conn = get_connection(db_path)
    try:
        with conn:
            # Verify profile exists
            cur = conn.execute("SELECT user_id FROM user_profiles WHERE user_id = ?", (user_id,))
            if not cur.fetchone():
                return False
            conn.execute("UPDATE user_profiles SET is_active = 0")
            conn.execute("UPDATE user_profiles SET is_active = 1 WHERE user_id = ?", (user_id,))
            return True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Meal Diary & Nutrition History Operations
# ---------------------------------------------------------------------------

VALID_MEAL_SLOTS = {"breakfast", "lunch", "snack", "dinner"}

def log_meal(meal_data: Dict[str, Any], db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Records a logged meal entry in SQLite with input validation and parameterized SQL.

    Parameters
    ----------
    meal_data : dict
        Contains user_id, food_name, portion_g, calories, protein_g, carbs_g, fat_g,
        meal_slot, optional meal_date, suitability_score, notes.
    """
    init_db(db_path)
    conn = get_connection(db_path)

    # Validate inputs
    user_id = str(meal_data.get("user_id", "")).strip()
    if not user_id:
        raise ValueError("Parameter 'user_id' is required for meal logging.")

    food_name = str(meal_data.get("food_name", "")).strip()
    if not food_name:
        raise ValueError("Parameter 'food_name' is required for meal logging.")

    try:
        portion_g = float(meal_data.get("portion_g", 0))
    except (ValueError, TypeError):
        raise ValueError("Parameter 'portion_g' must be a valid numeric amount.")
    if portion_g <= 0 or portion_g > 5000:
        raise ValueError("Parameter 'portion_g' must be between 1 and 5000 grams.")

    meal_slot = str(meal_data.get("meal_slot", "lunch")).lower().strip()
    if meal_slot not in VALID_MEAL_SLOTS:
        meal_slot = "lunch"

    try:
        calories = float(meal_data.get("calories", 0.0))
        protein_g = float(meal_data.get("protein_g", 0.0))
        carbs_g = float(meal_data.get("carbs_g", 0.0))
        fat_g = float(meal_data.get("fat_g", 0.0))
    except (ValueError, TypeError):
        raise ValueError("Nutrient values (calories, protein, carbs, fat) must be numeric.")

    if calories < 0 or protein_g < 0 or carbs_g < 0 or fat_g < 0:
        raise ValueError("Nutrient amounts cannot be negative.")

    fibre_g = float(meal_data["fibre_g"]) if meal_data.get("fibre_g") is not None else None
    sodium_mg = float(meal_data["sodium_mg"]) if meal_data.get("sodium_mg") is not None else None
    score = float(meal_data["suitability_score"]) if meal_data.get("suitability_score") is not None else None

    log_id = meal_data.get("log_id") or uuid.uuid4().hex
    now = datetime.now(timezone.utc)
    meal_date = str(meal_data.get("meal_date") or now.strftime("%Y-%m-%d")).strip()
    meal_time = str(meal_data.get("meal_time") or now.strftime("%H:%M:%S")).strip()
    logged_at = _now_iso()

    try:
        with conn:
            conn.execute("""
                INSERT INTO meal_logs (
                    log_id, user_id, meal_date, meal_time, meal_slot,
                    food_name, recipe_id, portion_g, portion_description,
                    calories, protein_g, carbs_g, fat_g, fibre_g, sodium_mg,
                    suitability_score, source_type, image_path, notes, logged_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id,
                user_id,
                meal_date,
                meal_time,
                meal_slot,
                food_name,
                meal_data.get("recipe_id"),
                portion_g,
                meal_data.get("portion_description", f"{portion_g}g serving"),
                calories,
                protein_g,
                carbs_g,
                fat_g,
                fibre_g,
                sodium_mg,
                score,
                meal_data.get("source_type", "manual_entry"),
                meal_data.get("image_path"),
                meal_data.get("notes", ""),
                logged_at
            ))

        # Retrieve inserted row
        cur = conn.execute("SELECT * FROM meal_logs WHERE log_id = ?", (log_id,))
        return dict(cur.fetchone())
    finally:
        conn.close()


def get_meal_history(
    user_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    meal_slot: Optional[str] = None,
    limit: int = 100,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves filtered meal log entries for a given user.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM meal_logs WHERE user_id = ?"
        params: List[Any] = [user_id]

        if start_date:
            query += " AND meal_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND meal_date <= ?"
            params.append(end_date)
        if meal_slot:
            query += " AND meal_slot = ?"
            params.append(meal_slot.lower().strip())

        query += " ORDER BY meal_date DESC, meal_time DESC LIMIT ?"
        params.append(limit)

        cur = conn.execute(query, tuple(params))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def delete_meal_log(
    log_id: str,
    user_id: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> bool:
    """Deletes a meal log entry by log_id (and optional user_id guard)."""
    conn = get_connection(db_path)
    try:
        with conn:
            if user_id:
                cur = conn.execute("DELETE FROM meal_logs WHERE log_id = ? AND user_id = ?", (log_id, user_id))
            else:
                cur = conn.execute("DELETE FROM meal_logs WHERE log_id = ?", (log_id,))
            return cur.rowcount > 0
    finally:
        conn.close()


def get_meal_totals(
    user_id: str,
    date_str: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Calculates aggregated daily nutrition totals and percentage of daily targets.
    """
    init_db(db_path)
    target_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    conn = get_connection(db_path)
    try:
        # Get active user profile to check calorie & RDA targets
        profile = db_get_profile(user_id, db_path=db_path) or {}
        calorie_target = float(profile.get("daily_calorie_target") or 2000.0)

        cur = conn.execute("""
            SELECT
                COUNT(*) as meal_count,
                COALESCE(SUM(calories), 0.0) as total_calories,
                COALESCE(SUM(protein_g), 0.0) as total_protein_g,
                COALESCE(SUM(carbs_g), 0.0) as total_carbs_g,
                COALESCE(SUM(fat_g), 0.0) as total_fat_g,
                COALESCE(SUM(fibre_g), 0.0) as total_fibre_g,
                COALESCE(SUM(sodium_mg), 0.0) as total_sodium_mg,
                COALESCE(AVG(suitability_score), 0.0) as avg_suitability_score
            FROM meal_logs
            WHERE user_id = ? AND meal_date = ?
        """, (user_id, target_date))
        agg = dict(cur.fetchone())

        # Break down by meal slot
        cur_slots = conn.execute("""
            SELECT meal_slot, COUNT(*) as count, SUM(calories) as slot_calories
            FROM meal_logs
            WHERE user_id = ? AND meal_date = ?
            GROUP BY meal_slot
        """, (user_id, target_date))
        slots_breakdown = {row["meal_slot"]: {"count": row["count"], "calories": round(row["slot_calories"], 1)} for row in cur_slots.fetchall()}

        tot_cals = round(agg["total_calories"], 1)
        calorie_progress_pct = round((tot_cals / calorie_target) * 100.0, 1) if calorie_target > 0 else 0.0

        return {
            "user_id": user_id,
            "date": target_date,
            "meal_count": agg["meal_count"],
            "totals": {
                "calories": tot_cals,
                "protein_g": round(agg["total_protein_g"], 1),
                "carbs_g": round(agg["total_carbs_g"], 1),
                "fat_g": round(agg["total_fat_g"], 1),
                "fibre_g": round(agg["total_fibre_g"], 1),
                "sodium_mg": round(agg["total_sodium_mg"], 1),
                "avg_suitability_score": round(agg["avg_suitability_score"], 1)
            },
            "target": {
                "daily_calorie_target": calorie_target,
                "calorie_progress_pct": calorie_progress_pct,
                "calories_remaining": max(0.0, round(calorie_target - tot_cals, 1))
            },
            "slots": slots_breakdown
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Meal Plan Persistence & Feedback Operations
# ---------------------------------------------------------------------------

def save_meal_plan_db(
    user_id: str,
    plan_type: str,
    plan_payload: Dict[str, Any],
    target_date: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None
) -> str:
    """Stores a generated daily or weekly meal plan in SQLite."""
    init_db(db_path)
    conn = get_connection(db_path)
    plan_id = uuid.uuid4().hex
    target_date = target_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_cals = float(plan_payload.get("target_calories") or plan_payload.get("daily_calorie_target") or 2000.0)

    try:
        with conn:
            conn.execute("""
                INSERT INTO meal_plans (
                    plan_id, user_id, plan_type, target_date, target_calories,
                    plan_payload, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                plan_id,
                user_id,
                plan_type,
                target_date,
                target_cals,
                json.dumps(plan_payload, ensure_ascii=False),
                _now_iso()
            ))
        return plan_id
    finally:
        conn.close()


def get_latest_meal_plan_db(
    user_id: str,
    plan_type: str = "daily",
    db_path: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent saved meal plan for a user."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("""
            SELECT * FROM meal_plans
            WHERE user_id = ? AND plan_type = ?
            ORDER BY created_at DESC LIMIT 1
        """, (user_id, plan_type))
        row = cur.fetchone()
        if not row:
            return None
        res = dict(row)
        res["plan_payload"] = json.loads(res["plan_payload"])
        return res
    finally:
        conn.close()


def record_feedback_db(
    feedback_data: Dict[str, Any],
    db_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Records user feedback, ratings, and vision prediction corrections."""
    init_db(db_path)
    conn = get_connection(db_path)
    f_id = feedback_data.get("feedback_id") or uuid.uuid4().hex
    now_str = _now_iso()

    try:
        with conn:
            conn.execute("""
                INSERT INTO user_feedback (
                    feedback_id, user_id, feedback_type, item_reference,
                    rating, user_corrected_name, comment, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                f_id,
                feedback_data.get("user_id"),
                feedback_data.get("feedback_type", "general"),
                feedback_data.get("item_reference", ""),
                int(feedback_data.get("rating", 5)) if feedback_data.get("rating") else None,
                feedback_data.get("user_corrected_name"),
                feedback_data.get("comment", ""),
                now_str
            ))
        return {
            "feedback_id": f_id,
            "status": "recorded",
            "submitted_at": now_str
        }
    finally:
        conn.close()


def get_feedback_list_db(
    limit: int = 50,
    db_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """Retrieves recent user feedback entries."""
    conn = get_connection(db_path)
    try:
        cur = conn.execute("SELECT * FROM user_feedback ORDER BY submitted_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_database_stats(db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Returns database size, file status, and table row counts."""
    init_db(db_path)
    path = get_db_path(db_path)
    conn = get_connection(db_path)
    tables = ["user_profiles", "foods", "recipes", "allergens", "meal_logs", "meal_plans", "user_feedback"]
    counts = {}

    try:
        for t in tables:
            cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
            counts[t] = cur.fetchone()[0]

        file_size_kb = round(os.path.getsize(str(path)) / 1024, 1) if path.exists() else 0.0

        return {
            "database_file": str(path),
            "exists": path.exists(),
            "size_kb": file_size_kb,
            "row_counts": counts,
            "status": "healthy"
        }
    finally:
        conn.close()
