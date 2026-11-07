"""
user_profile_manager.py
=======================
Phase 8 — NutriSense AI Samsung Project
User Profile Management System

Provides complete CRUD operations, validation, BMI calculation, ICMR-NIN profile
mapping, personalized breakfast suggestions, and personalized nutritional context
for users of the NutriSense AI platform.

All health-related guidance is educational and lifestyle-oriented.
See PHASE8_DISCLAIMER for full disclaimer text.

Author: NutriSense AI Team
Phase: 8
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

PHASE8_DISCLAIMER: str = (
    "NutriSense AI Phase 8 — User Profile Management. "
    "All nutritional guidance, calorie targets, and health-related information "
    "provided by this module are for general educational and lifestyle purposes only. "
    "They do NOT constitute medical advice, diagnosis, or treatment. "
    "Health conditions entered by the user are self-reported and are used solely to "
    "personalise dietary suggestions. Always consult a qualified healthcare professional "
    "or registered dietitian before making significant changes to your diet, especially "
    "if you have a medical condition. NutriSense AI accepts no liability for decisions "
    "made based on the information provided by this module."
)

BMI_LIMITATION_NOTE: str = (
    "BMI is a population-level screening tool and does not account for muscle mass, "
    "bone density, age, gender, or ethnic background. "
    "It should not be used as the sole indicator of health."
)

# ---------------------------------------------------------------------------
# Allowed-value enumerations
# ---------------------------------------------------------------------------

ALLOWED_GENDER = {"male", "female", "other", "prefer_not_to_say"}

ALLOWED_DIETARY_GOAL = {
    "general_wellness",
    "weight_loss",
    "muscle_gain",
    "diabetes_management",
    "hypertension_management",
    "heart_health",
    "balanced_nutrition",
}

ALLOWED_ACTIVITY_LEVEL = {
    "sedentary",
    "lightly_active",
    "moderately_active",
    "very_active",
    "extremely_active",
}

ALLOWED_DIETARY_PREFERENCE = {
    "vegetarian",
    "vegan",
    "non_vegetarian",
    "eggetarian",
    "jain",
    "satvik",
    "no_restriction",
}

ALLOWED_ALLERGIES = {
    "gluten", "dairy", "nuts", "soy", "seafood", "eggs",
    "sesame", "mustard", "none",
}

ALLOWED_HEALTH_CONDITIONS = {
    "diabetes", "hypertension", "high_cholesterol", "obesity", "none",
}

ALLOWED_BUDGET = {"low", "medium", "high"}

ALLOWED_MORNING_ROUTINE = {"early_bird", "regular", "late_riser"}

ALLOWED_BREAKFAST_PREFERENCE = {
    "light", "protein_rich", "traditional_indian", "quick",
    "cooked", "no_preference",
}

# ---------------------------------------------------------------------------
# ICMR-NIN calorie / protein reference targets (approximate adult values)
# Used for portion guidance calculations
# ---------------------------------------------------------------------------

_ICMR_PROFILES: Dict[str, Dict] = {
    "diabetic_adult": {
        "calories": 1800.0,
        "protein_g": 65.0,
        "description": "Moderate-energy, low-glycaemic diet for diabetes management",
    },
    "hypertensive_adult": {
        "calories": 1900.0,
        "protein_g": 60.0,
        "description": "DASH-aligned, sodium-controlled diet for hypertension management",
    },
    "weight_management": {
        "calories": 1600.0,
        "protein_g": 70.0,
        "description": "Calorie-controlled diet targeting gradual, sustainable weight loss",
    },
    "adult_female_sedentary": {
        "calories": 1900.0,
        "protein_g": 55.0,
        "description": "Reference diet for sedentary adult female (ICMR-NIN 2020)",
    },
    "adult_male_sedentary": {
        "calories": 2320.0,
        "protein_g": 60.0,
        "description": "Reference diet for sedentary adult male (ICMR-NIN 2020)",
    },
}

# Activity-level calorie multipliers (PAL-based)
_ACTIVITY_MULTIPLIER: Dict[str, float] = {
    "sedentary": 1.0,
    "lightly_active": 1.12,
    "moderately_active": 1.27,
    "very_active": 1.45,
    "extremely_active": 1.60,
}

# Goal-based calorie adjustment (delta from base)
_GOAL_CALORIE_DELTA: Dict[str, float] = {
    "general_wellness": 0.0,
    "weight_loss": -300.0,
    "muscle_gain": +250.0,
    "diabetes_management": -100.0,
    "hypertension_management": 0.0,
    "heart_health": -100.0,
    "balanced_nutrition": 0.0,
}

# ---------------------------------------------------------------------------
# Breakfast suggestion bank
# ---------------------------------------------------------------------------
# Each entry: food_name, routine, dietary_restrictions (preferences that CAN eat it),
#             estimated_calories, reason
_BREAKFAST_BANK: List[Dict] = [
    # Early bird
    {
        "food_name": "Idli with sambar",
        "routine": "early_bird",
        "compatible_preferences": {
            "vegetarian", "vegan", "eggetarian", "jain", "satvik", "no_restriction",
        },
        "estimated_calories": 150,
        "reason": "Light, steamed, low-fat, easy to digest — ideal for an early morning.",
        "goal_boost": {"general_wellness", "weight_loss", "balanced_nutrition", "diabetes_management"},
    },
    {
        "food_name": "Poha (flattened rice) with peanuts",
        "routine": "early_bird",
        "compatible_preferences": {
            "vegetarian", "eggetarian", "no_restriction",
        },
        "estimated_calories": 250,
        "reason": "Quick to prepare, iron-rich, moderate carbohydrate — great for early risers.",
        "goal_boost": {"general_wellness", "weight_loss", "balanced_nutrition"},
    },
    {
        "food_name": "Upma",
        "routine": "early_bird",
        "compatible_preferences": {
            "vegetarian", "eggetarian", "jain", "satvik", "no_restriction",
        },
        "estimated_calories": 220,
        "reason": "Semolina-based, filling without being heavy, good source of B-vitamins.",
        "goal_boost": {"general_wellness", "balanced_nutrition"},
    },
    {
        "food_name": "Moong dal chilla",
        "routine": "early_bird",
        "compatible_preferences": {
            "vegetarian", "vegan", "eggetarian", "jain", "satvik", "no_restriction",
        },
        "estimated_calories": 200,
        "reason": "High-protein lentil pancake — excellent for early-morning muscle fuelling.",
        "goal_boost": {"muscle_gain", "weight_loss", "diabetes_management"},
    },
    # Regular
    {
        "food_name": "Dosa with coconut chutney",
        "routine": "regular",
        "compatible_preferences": {
            "vegetarian", "vegan", "eggetarian", "no_restriction",
        },
        "estimated_calories": 300,
        "reason": "Fermented batter improves gut health; low in fat when prepared on a flat tawa.",
        "goal_boost": {"general_wellness", "balanced_nutrition", "heart_health"},
    },
    {
        "food_name": "Whole wheat paratha with curd",
        "routine": "regular",
        "compatible_preferences": {
            "vegetarian", "eggetarian", "no_restriction",
        },
        "estimated_calories": 380,
        "reason": "Complex carbohydrates + probiotic curd for sustained energy through the morning.",
        "goal_boost": {"muscle_gain", "general_wellness"},
    },
    {
        "food_name": "Besan chilla (chickpea flour pancake)",
        "routine": "regular",
        "compatible_preferences": {
            "vegetarian", "vegan", "eggetarian", "jain", "satvik", "no_restriction",
        },
        "estimated_calories": 230,
        "reason": "High protein, low glycaemic index — an excellent vegetarian protein breakfast.",
        "goal_boost": {"weight_loss", "muscle_gain", "diabetes_management"},
    },
    {
        "food_name": "Oats upma with vegetables",
        "routine": "regular",
        "compatible_preferences": {
            "vegetarian", "vegan", "eggetarian", "no_restriction",
        },
        "estimated_calories": 260,
        "reason": "Oats provide beta-glucan fibre, supportive of heart health and cholesterol management.",
        "goal_boost": {"heart_health", "weight_loss", "hypertension_management", "balanced_nutrition"},
    },
    # Late riser
    {
        "food_name": "Aloo paratha with pickle and curd",
        "routine": "late_riser",
        "compatible_preferences": {
            "vegetarian", "eggetarian", "no_restriction",
        },
        "estimated_calories": 420,
        "reason": "Substantial late-morning meal with carbohydrates and protein to bridge to lunch.",
        "goal_boost": {"general_wellness", "muscle_gain"},
    },
    {
        "food_name": "Chole (chickpea curry) with roti",
        "routine": "late_riser",
        "compatible_preferences": {
            "vegetarian", "vegan", "eggetarian", "no_restriction",
        },
        "estimated_calories": 400,
        "reason": "Protein- and fibre-rich legume meal; keeps you satiated until dinner.",
        "goal_boost": {"muscle_gain", "balanced_nutrition", "general_wellness"},
    },
    {
        "food_name": "Rajma rice (kidney bean curry with rice)",
        "routine": "late_riser",
        "compatible_preferences": {
            "vegetarian", "vegan", "eggetarian", "no_restriction",
        },
        "estimated_calories": 450,
        "reason": "Complete protein combination; high fibre aids digestion and satiety.",
        "goal_boost": {"muscle_gain", "balanced_nutrition"},
    },
    {
        "food_name": "Bread poha",
        "routine": "late_riser",
        "compatible_preferences": {
            "vegetarian", "eggetarian", "no_restriction",
        },
        "estimated_calories": 280,
        "reason": "Quick yet filling; retains the comfort of poha while using pantry staples.",
        "goal_boost": {"general_wellness", "balanced_nutrition"},
    },
]

# Allergen map — mirrors ALLERGEN_MAP in inference_pipeline for consistency
ALLERGEN_MAP: Dict[str, str] = {
    "gluten": "gluten",
    "dairy": "dairy",
    "nuts": "tree_nuts",
    "soy": "soy",
    "seafood": "fish_shellfish",
    "eggs": "eggs",
    "sesame": "sesame",
    "mustard": "mustard",
}

# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_PROFILES_DIR = _WORKSPACE_ROOT / "data" / "user_profiles"
_PROFILES_FILE = _PROFILES_DIR / "profiles.json"
PROFILES_DIR = _PROFILES_DIR
PROFILES_FILE = _PROFILES_FILE


def _ensure_storage() -> None:
    """Create the profiles directory and file if they do not exist."""
    _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    if not _PROFILES_FILE.exists():
        try:
            _PROFILES_FILE.write_text(
                json.dumps({"profiles": {}, "active_profile_id": None}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass


def _read_db() -> Dict:
    """
    Read the profiles JSON database from disk.

    Returns
    -------
    dict
        Parsed database dictionary with keys ``profiles`` and
        ``active_profile_id``.  Returns an empty valid structure on error.
    """
    _ensure_storage()
    try:
        text = _PROFILES_FILE.read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, json.JSONDecodeError):
        return {"profiles": {}, "active_profile_id": None}


def _write_db(db: Dict) -> bool:
    """
    Write the profiles database to disk atomically (best-effort).

    Parameters
    ----------
    db : dict
        The full database dictionary to persist.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on write error.
    """
    try:
        _PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        _PROFILES_FILE.write_text(
            json.dumps(db, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except OSError:
        return False


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_profile_data(data: dict) -> Tuple[bool, List[str]]:
    """
    Validate a profile data dictionary against all field constraints.

    Parameters
    ----------
    data : dict
        Raw profile data to validate.  ``user_id``, ``created_at``,
        ``updated_at``, and ``profile_version`` are managed internally and
        may be absent.

    Returns
    -------
    tuple[bool, list[str]]
        ``(True, [])`` when all fields are valid.
        ``(False, ["error message", ...])`` when one or more fields fail.

    Notes
    -----
    ``dietary_notes`` is silently truncated to 500 characters; a warning
    string is added to the error list (non-fatal) so callers can inform
    the user.
    """
    errors: List[str] = []

    # --- display_name (required) -------------------------------------------
    display_name = data.get("display_name")
    if display_name is None:
        errors.append("display_name is required.")
    else:
        display_name = str(display_name).strip()
        if not (2 <= len(display_name) <= 50):
            errors.append(
                f"display_name must be between 2 and 50 characters "
                f"(got {len(display_name)})."
            )

    # --- age (optional) -------------------------------------------------------
    if "age" in data and data["age"] is not None:
        age = data["age"]
        if not isinstance(age, int) or isinstance(age, bool):
            errors.append("age must be an integer.")
        elif not (1 <= age <= 120):
            errors.append(f"age must be between 1 and 120 (got {age}).")

    # --- gender (optional) ---------------------------------------------------
    if "gender" in data and data["gender"] is not None:
        if data["gender"] not in ALLOWED_GENDER:
            errors.append(
                f"gender must be one of {sorted(ALLOWED_GENDER)} "
                f"(got '{data['gender']}')."
            )

    # --- height_cm (optional) ------------------------------------------------
    if "height_cm" in data and data["height_cm"] is not None:
        h = data["height_cm"]
        if not isinstance(h, (int, float)) or isinstance(h, bool):
            errors.append("height_cm must be a number.")
        elif not (50.0 <= float(h) <= 250.0):
            errors.append(f"height_cm must be between 50 and 250 cm (got {h}).")

    # --- weight_kg (optional) ------------------------------------------------
    if "weight_kg" in data and data["weight_kg"] is not None:
        w = data["weight_kg"]
        if not isinstance(w, (int, float)) or isinstance(w, bool):
            errors.append("weight_kg must be a number.")
        elif not (20.0 <= float(w) <= 300.0):
            errors.append(f"weight_kg must be between 20 and 300 kg (got {w}).")

    # --- dietary_goal (optional enum) ----------------------------------------
    if "dietary_goal" in data and data["dietary_goal"] is not None:
        if data["dietary_goal"] not in ALLOWED_DIETARY_GOAL:
            errors.append(
                f"dietary_goal must be one of {sorted(ALLOWED_DIETARY_GOAL)} "
                f"(got '{data['dietary_goal']}')."
            )

    # --- activity_level (optional enum) --------------------------------------
    if "activity_level" in data and data["activity_level"] is not None:
        if data["activity_level"] not in ALLOWED_ACTIVITY_LEVEL:
            errors.append(
                f"activity_level must be one of {sorted(ALLOWED_ACTIVITY_LEVEL)} "
                f"(got '{data['activity_level']}')."
            )

    # --- dietary_preference (optional enum) ----------------------------------
    if "dietary_preference" in data and data["dietary_preference"] is not None:
        if data["dietary_preference"] not in ALLOWED_DIETARY_PREFERENCE:
            errors.append(
                f"dietary_preference must be one of {sorted(ALLOWED_DIETARY_PREFERENCE)} "
                f"(got '{data['dietary_preference']}')."
            )

    # --- allergies (optional list) -------------------------------------------
    if "allergies" in data and data["allergies"] is not None:
        allergies = data["allergies"]
        if not isinstance(allergies, list):
            errors.append("allergies must be a list.")
        else:
            invalid = [a for a in allergies if a not in ALLOWED_ALLERGIES]
            if invalid:
                errors.append(
                    f"Invalid allergy values: {invalid}. "
                    f"Allowed: {sorted(ALLOWED_ALLERGIES)}."
                )

    # --- health_conditions (optional list) -----------------------------------
    if "health_conditions" in data and data["health_conditions"] is not None:
        hc = data["health_conditions"]
        if not isinstance(hc, list):
            errors.append("health_conditions must be a list.")
        else:
            invalid = [c for c in hc if c not in ALLOWED_HEALTH_CONDITIONS]
            if invalid:
                errors.append(
                    f"Invalid health_condition values: {invalid}. "
                    f"Allowed: {sorted(ALLOWED_HEALTH_CONDITIONS)}."
                )

    # --- budget (optional enum) ----------------------------------------------
    if "budget" in data and data["budget"] is not None:
        if data["budget"] not in ALLOWED_BUDGET:
            errors.append(
                f"budget must be one of {sorted(ALLOWED_BUDGET)} "
                f"(got '{data['budget']}')."
            )

    # --- max_cooking_time_minutes (optional) ---------------------------------
    if "max_cooking_time_minutes" in data and data["max_cooking_time_minutes"] is not None:
        mct = data["max_cooking_time_minutes"]
        if not isinstance(mct, int) or isinstance(mct, bool):
            errors.append("max_cooking_time_minutes must be an integer.")
        elif not (5 <= mct <= 180):
            errors.append(
                f"max_cooking_time_minutes must be between 5 and 180 "
                f"(got {mct})."
            )

    # --- avoid_ingredients (optional list of free-text strings) --------------
    if "avoid_ingredients" in data and data["avoid_ingredients"] is not None:
        ai = data["avoid_ingredients"]
        if not isinstance(ai, list):
            errors.append("avoid_ingredients must be a list.")
        else:
            if len(ai) > 20:
                errors.append(
                    f"avoid_ingredients can contain at most 20 items "
                    f"(got {len(ai)})."
                )
            long_items = [
                item for item in ai
                if isinstance(item, str) and len(item) > 50
            ]
            if long_items:
                errors.append(
                    f"Each avoid_ingredient must be at most 50 characters. "
                    f"Offending items: {long_items}."
                )
            non_strings = [item for item in ai if not isinstance(item, str)]
            if non_strings:
                errors.append(
                    f"All items in avoid_ingredients must be strings. "
                    f"Non-string items found: {non_strings}."
                )

    # --- dietary_notes (optional, truncated to 500 chars) --------------------
    if "dietary_notes" in data and data["dietary_notes"] is not None:
        dn = str(data["dietary_notes"]).strip()
        if len(dn) > 500:
            # Non-fatal — caller should truncate
            errors.append(
                f"dietary_notes exceeds 500 characters and will be truncated "
                f"to 500 (original length: {len(dn)})."
            )

    # --- morning_routine_preference (optional enum) --------------------------
    if "morning_routine_preference" in data and data["morning_routine_preference"] is not None:
        if data["morning_routine_preference"] not in ALLOWED_MORNING_ROUTINE:
            errors.append(
                f"morning_routine_preference must be one of "
                f"{sorted(ALLOWED_MORNING_ROUTINE)} "
                f"(got '{data['morning_routine_preference']}')."
            )

    # --- breakfast_preference (optional list) --------------------------------
    if "breakfast_preference" in data and data["breakfast_preference"] is not None:
        bp = data["breakfast_preference"]
        if not isinstance(bp, list):
            errors.append("breakfast_preference must be a list.")
        else:
            invalid = [b for b in bp if b not in ALLOWED_BREAKFAST_PREFERENCE]
            if invalid:
                errors.append(
                    f"Invalid breakfast_preference values: {invalid}. "
                    f"Allowed: {sorted(ALLOWED_BREAKFAST_PREFERENCE)}."
                )

    # Determine overall validity (truncation warning is non-fatal)
    fatal_errors = [
        e for e in errors
        if "will be truncated" not in e
    ]
    is_valid = len(fatal_errors) == 0
    return is_valid, errors


def _sanitise_data(data: dict) -> dict:
    """
    Apply sanitisation rules to incoming profile data before saving.

    * Strips whitespace from ``display_name``.
    * Truncates ``dietary_notes`` to 500 characters.
    * Converts numeric fields to the correct Python types.

    Parameters
    ----------
    data : dict
        Raw profile data.

    Returns
    -------
    dict
        Sanitised copy of the data.
    """
    cleaned = dict(data)

    if "display_name" in cleaned and cleaned["display_name"] is not None:
        cleaned["display_name"] = str(cleaned["display_name"]).strip()

    if "dietary_notes" in cleaned and cleaned["dietary_notes"] is not None:
        dn = str(cleaned["dietary_notes"]).strip()
        cleaned["dietary_notes"] = dn[:500]

    if "height_cm" in cleaned and cleaned["height_cm"] is not None:
        cleaned["height_cm"] = float(cleaned["height_cm"])

    if "weight_kg" in cleaned and cleaned["weight_kg"] is not None:
        cleaned["weight_kg"] = float(cleaned["weight_kg"])

    if "age" in cleaned and cleaned["age"] is not None:
        cleaned["age"] = int(cleaned["age"])

    if "max_cooking_time_minutes" in cleaned and cleaned["max_cooking_time_minutes"] is not None:
        cleaned["max_cooking_time_minutes"] = int(cleaned["max_cooking_time_minutes"])

    return cleaned


# ---------------------------------------------------------------------------
# BMI
# ---------------------------------------------------------------------------

def calculate_bmi(height_cm: float, weight_kg: float) -> dict:
    """
    Calculate BMI and return a structured result with category and disclaimer.

    Parameters
    ----------
    height_cm : float
        Height in centimetres (must be > 0).
    weight_kg : float
        Weight in kilograms (must be > 0).

    Returns
    -------
    dict
        Keys:

        * ``value`` (float): BMI rounded to 1 decimal place.
        * ``category`` (str): One of ``"Underweight"``, ``"Normal"``,
          ``"Overweight"``, ``"Obese"``.
        * ``limitation_note`` (str): Standard BMI limitation disclaimer.

    Raises
    ------
    ValueError
        If height_cm or weight_kg are not positive numbers.

    Examples
    --------
    >>> calculate_bmi(170, 65)
    {'value': 22.5, 'category': 'Normal', 'limitation_note': '...'}
    """
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("height_cm and weight_kg must both be positive numbers.")

    height_m = height_cm / 100.0
    bmi_value = weight_kg / (height_m ** 2)
    bmi_rounded = round(bmi_value, 1)

    if bmi_rounded < 18.5:
        category = "Underweight"
    elif bmi_rounded < 25.0:
        category = "Normal"
    elif bmi_rounded < 30.0:
        category = "Overweight"
    else:
        category = "Obese"

    return {
        "value": bmi_rounded,
        "category": category,
        "limitation_note": BMI_LIMITATION_NOTE,
    }


# ---------------------------------------------------------------------------
# ICMR-NIN profile mapping
# ---------------------------------------------------------------------------

def _map_icmr_profile(profile: dict) -> str:
    """
    Map a user profile to an ICMR-NIN reference profile key.

    Priority order:

    1. Diabetes (goal or condition) → ``"diabetic_adult"``
    2. Hypertension (goal or condition) → ``"hypertensive_adult"``
    3. Weight-loss goal → ``"weight_management"``
    4. Female gender (no clinical condition above) → ``"adult_female_sedentary"``
    5. Default → ``"adult_male_sedentary"``

    Parameters
    ----------
    profile : dict
        A validated user profile dictionary.

    Returns
    -------
    str
        One of the keys in :data:`_ICMR_PROFILES`.
    """
    goal = profile.get("dietary_goal", "")
    conditions = set(profile.get("health_conditions") or [])

    if goal == "diabetes_management" or "diabetes" in conditions:
        return "diabetic_adult"

    if goal == "hypertension_management" or "hypertension" in conditions:
        return "hypertensive_adult"

    if goal == "weight_loss":
        return "weight_management"

    if profile.get("gender") == "female":
        return "adult_female_sedentary"

    return "adult_male_sedentary"


# ---------------------------------------------------------------------------
# Breakfast suggestions
# ---------------------------------------------------------------------------

def get_breakfast_suggestions(profile: dict) -> List[dict]:
    """
    Return up to 3 personalised Indian breakfast suggestions.

    Suggestions are filtered by:

    * ``morning_routine_preference`` (early_bird / regular / late_riser)
    * ``dietary_preference`` (vegetarian, vegan, etc.)
    * ``dietary_goal`` (ranked by goal alignment)

    Parameters
    ----------
    profile : dict
        A validated user profile dictionary.

    Returns
    -------
    list[dict]
        Up to 3 dicts with keys ``food_name``, ``reason``,
        ``estimated_calories``.
    """
    routine = profile.get("morning_routine_preference", "regular")
    pref = profile.get("dietary_preference", "no_restriction")
    goal = profile.get("dietary_goal", "general_wellness")

    # Filter by routine and dietary compatibility
    candidates = [
        item for item in _BREAKFAST_BANK
        if item["routine"] == routine and pref in item["compatible_preferences"]
    ]

    # If no candidates for routine, fall back to "regular"
    if not candidates:
        candidates = [
            item for item in _BREAKFAST_BANK
            if item["routine"] == "regular" and pref in item["compatible_preferences"]
        ]

    # Sort: goal-boosted items first
    def _rank(item: dict) -> int:
        return 0 if goal in item.get("goal_boost", set()) else 1

    candidates.sort(key=_rank)

    result = []
    for item in candidates[:3]:
        result.append(
            {
                "food_name": item["food_name"],
                "reason": item["reason"],
                "estimated_calories": item["estimated_calories"],
            }
        )

    return result


# ---------------------------------------------------------------------------
# Personalised context
# ---------------------------------------------------------------------------

def _build_portion_guidance(profile: dict, daily_calories: float) -> str:
    """
    Build a human-readable portion guidance string.

    Parameters
    ----------
    profile : dict
        Validated user profile.
    daily_calories : float
        Computed daily calorie target.

    Returns
    -------
    str
        A single guidance sentence suitable for display.
    """
    goal = profile.get("dietary_goal", "general_wellness")
    activity = profile.get("activity_level", "sedentary")

    # Approximate main-meal share: 30-35 % of daily calories
    lower = round(daily_calories * 0.30)
    upper = round(daily_calories * 0.35)

    activity_label = activity.replace("_", " ")
    return (
        f"Based on your activity level ({activity_label}) and goal "
        f"({goal.replace('_', ' ')}), aim for approximately "
        f"{lower}–{upper} kcal per main meal. "
        "Distribute the remainder across breakfast, snacks, and an evening meal. "
        "Actual needs vary — listen to your body and consult a dietitian for personalised advice."
    )


def _build_allergen_avoid_list(profile: dict) -> List[str]:
    """
    Convert user allergy selections to standardised allergen codes.

    Parameters
    ----------
    profile : dict
        Validated user profile.

    Returns
    -------
    list[str]
        Standardised allergen codes from :data:`ALLERGEN_MAP`.
    """
    allergies = profile.get("allergies") or []
    avoid = []
    for allergy in allergies:
        if allergy == "none":
            continue
        code = ALLERGEN_MAP.get(allergy)
        if code:
            avoid.append(code)
    return avoid


def get_personalized_context(user_id_or_profile: Union[str, dict]) -> dict:
    """
    Compute and return a full personalised nutritional context for a user.

    Accepts either a ``user_id`` string (profile is loaded from storage) or
    a fully-formed profile dict (useful for preview before saving).

    Parameters
    ----------
    user_id_or_profile : str or dict
        Either the string UUID of an existing profile or a profile dict.

    Returns
    -------
    dict
        Keys:

        * ``icmr_profile_key`` (str)
        * ``daily_calorie_target`` (float)
        * ``daily_protein_target`` (float)
        * ``bmi_info`` (dict or None)
        * ``suggested_breakfast`` (list[dict])
        * ``portion_guidance`` (str)
        * ``allergen_avoid_list`` (list[str])
        * ``personalization_disclaimer`` (str)

    Raises
    ------
    ValueError
        If a ``user_id`` is provided but the profile is not found.
    TypeError
        If the argument is neither a string nor a dict.
    """
    if isinstance(user_id_or_profile, str):
        profile = get_profile(user_id_or_profile)
        if profile is None:
            raise ValueError(
                f"No profile found with user_id '{user_id_or_profile}'."
            )
    elif isinstance(user_id_or_profile, dict):
        profile = user_id_or_profile
    else:
        raise TypeError(
            "user_id_or_profile must be a user_id string or a profile dict."
        )

    # --- ICMR profile mapping ------------------------------------------------
    icmr_key = _map_icmr_profile(profile)
    icmr_ref = _ICMR_PROFILES[icmr_key]

    # --- Calorie & protein targets -------------------------------------------
    base_calories = icmr_ref["calories"]
    activity = profile.get("activity_level", "sedentary")
    goal = profile.get("dietary_goal", "general_wellness")

    activity_mult = _ACTIVITY_MULTIPLIER.get(activity, 1.0)
    goal_delta = _GOAL_CALORIE_DELTA.get(goal, 0.0)

    # Apply activity multiplier then goal delta
    daily_calories = round((base_calories * activity_mult) + goal_delta, 1)
    daily_protein = icmr_ref["protein_g"]

    # Protein boost for muscle gain
    if goal == "muscle_gain":
        daily_protein = round(daily_protein * 1.3, 1)

    # --- BMI -----------------------------------------------------------------
    bmi_info: Optional[dict] = None
    h = profile.get("height_cm")
    w = profile.get("weight_kg")
    if h and w:
        try:
            bmi_info = calculate_bmi(float(h), float(w))
        except ValueError:
            bmi_info = None

    # --- Breakfast suggestions -----------------------------------------------
    suggested_breakfast = get_breakfast_suggestions(profile)

    # --- Portion guidance ----------------------------------------------------
    portion_guidance = _build_portion_guidance(profile, daily_calories)

    # --- Allergen list -------------------------------------------------------
    allergen_avoid_list = _build_allergen_avoid_list(profile)

    # --- Disclaimer ----------------------------------------------------------
    personalization_disclaimer = (
        "These nutritional targets are estimates derived from ICMR-NIN 2020 "
        "reference values adjusted for your self-reported activity level and goal. "
        "They are intended as lifestyle guidance only and do NOT constitute "
        "personalised medical nutrition therapy. Health conditions are self-reported. "
        "Consult a registered dietitian or physician for clinical dietary advice. "
        f"{PHASE8_DISCLAIMER}"
    )

    return {
        "icmr_profile_key": icmr_key,
        "daily_calorie_target": daily_calories,
        "daily_protein_target": daily_protein,
        "bmi_info": bmi_info,
        "suggested_breakfast": suggested_breakfast,
        "portion_guidance": portion_guidance,
        "allergen_avoid_list": allergen_avoid_list,
        "personalization_disclaimer": personalization_disclaimer,
    }


# ---------------------------------------------------------------------------
# Public CRUD API
# ---------------------------------------------------------------------------

def create_profile(data: dict) -> dict:
    """
    Validate and persist a new user profile.

    A new UUID is generated for ``user_id``.  The profile is saved to the
    JSON database immediately.

    Parameters
    ----------
    data : dict
        Profile field values.  ``user_id``, ``created_at``, ``updated_at``,
        and ``profile_version`` must NOT be supplied — they are managed
        internally.

    Returns
    -------
    dict
        The saved profile dict including all auto-generated fields.

    Raises
    ------
    ValueError
        If validation fails.  The exception message lists all validation
        errors joined by semicolons.
    """
    is_valid, errors = validate_profile_data(data)
    if not is_valid:
        raise ValueError(
            "Profile validation failed: " + "; ".join(
                e for e in errors if "will be truncated" not in e
            )
        )

    cleaned = _sanitise_data(data)
    now = _now_iso()

    profile: dict = {
        "user_id": str(uuid.uuid4()),
        "display_name": cleaned["display_name"],
        "age": cleaned.get("age"),
        "gender": cleaned.get("gender"),
        "height_cm": cleaned.get("height_cm"),
        "weight_kg": cleaned.get("weight_kg"),
        "dietary_goal": cleaned.get("dietary_goal"),
        "activity_level": cleaned.get("activity_level"),
        "dietary_preference": cleaned.get("dietary_preference"),
        "allergies": cleaned.get("allergies", []),
        "health_conditions": cleaned.get("health_conditions", []),
        "budget": cleaned.get("budget"),
        "max_cooking_time_minutes": cleaned.get("max_cooking_time_minutes"),
        "avoid_ingredients": cleaned.get("avoid_ingredients", []),
        "dietary_notes": cleaned.get("dietary_notes"),
        "morning_routine_preference": cleaned.get("morning_routine_preference"),
        "breakfast_preference": cleaned.get("breakfast_preference", []),
        "created_at": now,
        "updated_at": now,
        "profile_version": 1,
    }

    db = _read_db()
    db["profiles"][profile["user_id"]] = profile

    # Set as active if no active profile exists
    if db.get("active_profile_id") is None:
        db["active_profile_id"] = profile["user_id"]

    _write_db(db)
    return dict(profile)


def update_profile(user_id: str, data: dict) -> dict:
    """
    Validate and merge new data into an existing profile.

    Only the fields present in ``data`` are updated.  ``user_id``,
    ``created_at``, and ``profile_version`` cannot be overwritten.

    Parameters
    ----------
    user_id : str
        UUID of the profile to update.
    data : dict
        Fields to update with their new values.

    Returns
    -------
    dict
        The updated profile dict.

    Raises
    ------
    ValueError
        If the profile does not exist or validation fails.
    """
    db = _read_db()
    if user_id not in db["profiles"]:
        raise ValueError(f"No profile found with user_id '{user_id}'.")

    existing = dict(db["profiles"][user_id])

    # Merge incoming data over existing (excluding managed fields)
    merged_candidate = dict(existing)
    protected = {"user_id", "created_at", "updated_at", "profile_version"}
    for key, value in data.items():
        if key not in protected:
            merged_candidate[key] = value

    is_valid, errors = validate_profile_data(merged_candidate)
    if not is_valid:
        raise ValueError(
            "Profile update validation failed: " + "; ".join(
                e for e in errors if "will be truncated" not in e
            )
        )

    merged_candidate = _sanitise_data(merged_candidate)
    merged_candidate["updated_at"] = _now_iso()
    merged_candidate["profile_version"] = existing.get("profile_version", 1) + 1
    # Preserve original managed fields
    merged_candidate["user_id"] = existing["user_id"]
    merged_candidate["created_at"] = existing["created_at"]

    db["profiles"][user_id] = merged_candidate
    _write_db(db)
    return dict(merged_candidate)


def get_profile(user_id: str) -> Optional[dict]:
    """
    Retrieve a full profile by its UUID.

    Parameters
    ----------
    user_id : str
        UUID of the profile to retrieve.

    Returns
    -------
    dict or None
        The profile dict, or ``None`` if not found.
    """
    db = _read_db()
    profile = db["profiles"].get(user_id)
    return dict(profile) if profile else None


def delete_profile(user_id: str) -> bool:
    """
    Delete a profile from the database.

    If the deleted profile was the active profile, ``active_profile_id`` is
    reset to ``None``.

    Parameters
    ----------
    user_id : str
        UUID of the profile to delete.

    Returns
    -------
    bool
        ``True`` if the profile was deleted, ``False`` if it did not exist.
    """
    db = _read_db()
    if user_id not in db["profiles"]:
        return False

    del db["profiles"][user_id]

    if db.get("active_profile_id") == user_id:
        # Promote the first remaining profile to active, or set None
        remaining = list(db["profiles"].keys())
        db["active_profile_id"] = remaining[0] if remaining else None

    return _write_db(db)


def list_profiles() -> List[dict]:
    """
    Return a summary list of all stored profiles.

    Only a subset of fields is included to keep the response lightweight:
    ``user_id``, ``display_name``, ``dietary_goal``, ``dietary_preference``,
    ``created_at``, ``updated_at``, ``profile_version``.

    Returns
    -------
    list[dict]
        List of profile summary dicts, ordered by ``created_at`` ascending.
    """
    db = _read_db()
    summary_fields = {
        "user_id",
        "display_name",
        "dietary_goal",
        "dietary_preference",
        "activity_level",
        "created_at",
        "updated_at",
        "profile_version",
    }
    summaries = []
    for profile in db["profiles"].values():
        summary = {k: v for k, v in profile.items() if k in summary_fields}
        summaries.append(summary)

    summaries.sort(key=lambda p: p.get("created_at", ""))
    return summaries


def get_active_profile() -> Optional[dict]:
    """
    Retrieve the currently active profile.

    Returns
    -------
    dict or None
        The active profile dict, or ``None`` if no active profile is set.
    """
    db = _read_db()
    active_id = db.get("active_profile_id")
    if not active_id:
        return None
    return get_profile(active_id)


def set_active_profile(user_id: str) -> bool:
    """
    Set the active profile by user_id.

    Parameters
    ----------
    user_id : str
        UUID of the profile to make active.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if the profile does not exist.
    """
    db = _read_db()
    if user_id not in db["profiles"]:
        return False
    db["active_profile_id"] = user_id
    return _write_db(db)


# ---------------------------------------------------------------------------
# Module self-test (run with: python -m src.user_profile_manager)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("NutriSense AI — Phase 8: User Profile Manager Self-Test")
    print("=" * 70)

    # 1. Validation test
    print("\n[1] Validation test — missing display_name:")
    ok, errs = validate_profile_data({})
    print(f"    Valid: {ok}, Errors: {errs}")

    # 2. Create profile
    print("\n[2] Creating a sample profile...")
    sample = {
        "display_name": "Priya Sharma",
        "age": 32,
        "gender": "female",
        "height_cm": 162.0,
        "weight_kg": 65.0,
        "dietary_goal": "weight_loss",
        "activity_level": "moderately_active",
        "dietary_preference": "vegetarian",
        "allergies": ["dairy"],
        "health_conditions": ["none"],
        "budget": "medium",
        "max_cooking_time_minutes": 30,
        "avoid_ingredients": ["refined sugar"],
        "dietary_notes": "Prefer South Indian food. Avoid very spicy dishes.",
        "morning_routine_preference": "early_bird",
        "breakfast_preference": ["light", "traditional_indian"],
    }
    profile = create_profile(sample)
    print(f"    Created user_id: {profile['user_id']}")
    print(f"    display_name: {profile['display_name']}")
    print(f"    profile_version: {profile['profile_version']}")

    # 3. BMI
    print("\n[3] BMI calculation:")
    bmi = calculate_bmi(profile["height_cm"], profile["weight_kg"])
    print(f"    BMI: {bmi['value']} — {bmi['category']}")

    # 4. Update profile
    print("\n[4] Updating profile weight...")
    updated = update_profile(profile["user_id"], {"weight_kg": 62.0})
    print(f"    New weight_kg: {updated['weight_kg']}")
    print(f"    profile_version: {updated['profile_version']}")

    # 5. Personalised context
    print("\n[5] Personalised context:")
    ctx = get_personalized_context(profile["user_id"])
    print(f"    ICMR profile key: {ctx['icmr_profile_key']}")
    print(f"    Daily calorie target: {ctx['daily_calorie_target']} kcal")
    print(f"    Daily protein target: {ctx['daily_protein_target']} g")
    print(f"    Allergen avoid list: {ctx['allergen_avoid_list']}")
    print(f"    BMI info: {ctx['bmi_info']}")
    print(f"    Breakfast suggestions:")
    for s in ctx["suggested_breakfast"]:
        print(f"      - {s['food_name']} (~{s['estimated_calories']} kcal): {s['reason']}")
    print(f"    Portion guidance: {ctx['portion_guidance']}")

    # 6. List profiles
    print("\n[6] Profile list:")
    for p in list_profiles():
        print(f"    {p['user_id'][:8]}... — {p['display_name']}")

    # 7. Active profile
    print("\n[7] Active profile display_name:", get_active_profile()["display_name"])

    # 8. Delete
    print("\n[8] Deleting profile...")
    deleted = delete_profile(profile["user_id"])
    print(f"    Deleted: {deleted}")
    print(f"    Active profile after delete: {get_active_profile()}")

    print("\n" + "=" * 70)
    print("Self-test complete.")
    print("=" * 70)
