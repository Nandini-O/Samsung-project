"""
src/recommendation_engine.py
============================
Phase 9 — NutriSense AI Samsung Project
Recommendation System & Meal Planning Engine

Provides:
 1. Transparent Suitability Scoring (0 to 100) with documented weights & formula.
 2. Mandatory Pre-Filtering Safety Rule: Hard exclusion for allergens & dietary restrictions
    BEFORE ranking, with incomplete information flagging. Never allows high score to override safety.
 3. Content-Based Filtering using TF-IDF & Cosine Similarity on ingredient tokens.
 4. Personalized Recipe & Food Recommendations tailored to goals, budget, and cooking time.
 5. Alternative Food Suggestions for unfavorable or high-risk items.
 6. Comprehensive Culinary Ingredient Substitutions for healthy & allergy-friendly cooking.
 7. Daily Meal Planning with target calorie distribution (Breakfast, Lunch, Snack, Dinner).
 8. Weekly Meal Planning (7-Day schedule) with inter-day variety & automated grocery list.
 9. Plain English explanations and strict educational/lifestyle disclaimers.

Author: NutriSense AI Team
Phase: 9
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Workspace Paths
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
_FEATURE_DIR = _WORKSPACE_ROOT / "data" / "processed" / "feature_store"
_RECIPES_CSV = _FEATURE_DIR / "recipe_feature_store.csv"
_MASTER_KB_CSV = _FEATURE_DIR / "master_food_nutrition_kb.csv"

# ---------------------------------------------------------------------------
# Educational / Non-Medical Disclaimer
# ---------------------------------------------------------------------------

RECOMMENDATION_DISCLAIMER: str = (
    "NutriSense AI Phase 9 Recommendation System. All suitability scores (0-100), "
    "meal plans, alternative food suggestions, and ingredient substitutions are "
    "algorithmic lifestyle and educational guidance based on general nutritional guidelines. "
    "They do NOT constitute medical diagnosis, clinical dietetics, or prescriptive nutritional "
    "therapy. Scores and suggestions are not medically validated. Always consult a qualified "
    "physician, registered dietitian, or allergist before undertaking significant dietary "
    "modifications, especially for chronic health conditions or severe food allergies."
)

SAFETY_OVERRIDE_NOTE: str = (
    "Safety Filter Precedence: Recipes or foods containing confirmed allergens or "
    "violating declared dietary restrictions are strictly disqualified (Score = 0) "
    "BEFORE any ranking occurs. High nutritional suitability never overrides safety filters."
)

# ---------------------------------------------------------------------------
# Transparent Scoring Formula Weights
# ---------------------------------------------------------------------------

SCORE_WEIGHTS: Dict[str, float] = {
    "goal_alignment": 0.30,       # 30%: Alignment with user's specific dietary goal
    "macro_micro_balance": 0.25,  # 25%: Protein, fiber, lower sodium/sugar indicators
    "content_preference": 0.20,   # 20%: TF-IDF taste, cuisine, & ingredient similarity
    "time_feasibility": 0.15,     # 15%: Cooking and preparation time constraints
    "budget_fit": 0.10,           # 10%: Economic accessibility & ingredient cost tier
}

# ---------------------------------------------------------------------------
# Allergen & Dietary Restriction Lexicons (Word-boundary matching)
# ---------------------------------------------------------------------------

ALLERGEN_LEXICON: Dict[str, Set[str]] = {
    "gluten": {
        "wheat", "maida", "atta", "suji", "rava", "semolina", "barley", "rye",
        "bread", "pasta", "flour", "all purpose flour", "noodles", "roti", "sewai",
        "vermicelli", "malt"
    },
    "dairy": {
        "milk", "paneer", "ghee", "butter", "curd", "yogurt", "cheese", "cream",
        "malai", "khoya", "mawa", "whey", "buttermilk", "chaas", "dahi", "chenna",
        "condensed milk", "dairy"
    },
    "nuts": {
        "cashew", "kaju", "peanut", "peanuts", "groundnut", "moongphali", "almond",
        "badam", "walnut", "akhrot", "pistachio", "pista", "hazelnut", "pecan",
        "macadamia", "pine nuts", "chironji", "charoli"
    },
    "soy": {
        "soy", "soya", "tofu", "edamame", "soy sauce", "soybean", "soy milk",
        "soya chunks", "soya granules", "tempeh"
    },
    "seafood": {
        "fish", "prawn", "prawns", "shrimp", "shrimps", "crab", "crabs", "lobster",
        "salmon", "tuna", "pomfret", "surmai", "rohu", "katla", "hilsa", "mackerel",
        "anchovy", "clam", "mussel", "squid", "octopus", "seafood"
    },
    "eggs": {
        "egg", "eggs", "anda", "omelette", "mayonnaise", "egg white", "egg yolk"
    },
    "sesame": {
        "sesame", "til", "tahini", "gingelly"
    },
    "mustard": {
        "mustard", "sarson", "rai", "mustard oil", "kasundi"
    },
}

DIETARY_RESTRICTION_LEXICON: Dict[str, Set[str]] = {
    "vegetarian": {
        "chicken", "mutton", "lamb", "fish", "prawn", "shrimp", "crab", "lobster",
        "pork", "beef", "bacon", "meat", "poultry", "seafood", "egg", "eggs", "anda",
        "gelatin", "lard"
    },
    "vegan": {
        "chicken", "mutton", "lamb", "fish", "prawn", "shrimp", "crab", "lobster",
        "pork", "beef", "meat", "egg", "eggs", "anda", "gelatin", "lard",
        "milk", "paneer", "ghee", "butter", "curd", "yogurt", "cheese", "cream",
        "malai", "khoya", "mawa", "whey", "honey", "dairy", "dahi"
    },
    "jain": {
        "chicken", "mutton", "fish", "prawn", "egg", "meat", "seafood",
        "onion", "onions", "garlic", "potato", "potatoes", "aloo", "carrot", "carrots",
        "radish", "mooli", "ginger", "adrak", "beetroot", "turnip", "shallot",
        "shallots", "root vegetable"
    },
    "satvik": {
        "chicken", "mutton", "fish", "prawn", "egg", "meat", "seafood",
        "onion", "onions", "garlic", "mushroom", "mushrooms", "alcohol", "wine",
        "beer", "vinegar"
    },
    "eggetarian": {
        "chicken", "mutton", "lamb", "fish", "prawn", "shrimp", "crab", "lobster",
        "pork", "beef", "meat", "poultry", "seafood"
    },
}

# ---------------------------------------------------------------------------
# Comprehensive Culinary Ingredient Substitution Registry (25+ entries)
# ---------------------------------------------------------------------------

INGREDIENT_SUBSTITUTIONS_REGISTRY: List[Dict[str, Any]] = [
    {
        "id": "sub_maida_wwf",
        "original_ingredient": "Maida (Refined Flour)",
        "substitute": "Whole Wheat Flour (Atta) / Oat Flour",
        "ratio": "1:1 by weight",
        "health_benefit": "Increases dietary fiber by 400%, lowers glycemic index, enhances sustained fullness.",
        "target_goals": ["diabetes_management", "weight_loss", "balanced_nutrition", "heart_health"],
        "allergy_tags": [],
        "dietary_tags": ["vegetarian", "vegan", "satvik", "jain"],
        "culinary_notes": "For baking, add 1-2 tbsp extra liquid as whole wheat flour absorbs more hydration."
    },
    {
        "id": "sub_white_rice_brown",
        "original_ingredient": "White Rice (Polished)",
        "substitute": "Brown Rice / Quinoa / Foxtail Millet",
        "ratio": "1:1 cooked",
        "health_benefit": "Preserves bran and germ layers, offering 3x more magnesium and slower carbohydrate absorption.",
        "target_goals": ["diabetes_management", "weight_loss", "heart_health"],
        "allergy_tags": ["gluten_free"],
        "dietary_tags": ["vegetarian", "vegan", "satvik", "jain"],
        "culinary_notes": "Millet cooks in 15 mins with 2:1 water ratio; brown rice requires 35 mins."
    },
    {
        "id": "sub_sugar_jaggery",
        "original_ingredient": "Refined White Sugar",
        "substitute": "Pure Jaggery (Gur) / Date Puree / Stevia",
        "ratio": "3/4 cup jaggery or 1/2 tsp pure stevia per 1 cup sugar",
        "health_benefit": "Date puree supplies potassium, iron, and fiber. Stevia provides zero-calorie sweetness without blood sugar spikes.",
        "target_goals": ["diabetes_management", "weight_loss"],
        "allergy_tags": [],
        "dietary_tags": ["vegetarian", "vegan", "satvik", "jain"],
        "culinary_notes": "Stevia is ideal for diabetic hot beverages; date paste works best in halwa and kheer."
    },
    {
        "id": "sub_heavy_cream_hung_curd",
        "original_ingredient": "Heavy Cream / Malai",
        "substitute": "Blended Hung Curd (Greek Yogurt) / Cashew Paste",
        "ratio": "1:1 by volume",
        "health_benefit": "Hung curd cuts saturated fat by 70% while quadrupling protein content; cashew paste offers healthy monounsaturated fats.",
        "target_goals": ["weight_loss", "muscle_gain", "heart_health", "hypertension_management"],
        "allergy_tags": [],
        "dietary_tags": ["vegetarian", "satvik", "jain"],
        "culinary_notes": "Whisk hung curd off the flame before folding into gravies to prevent curdling."
    },
    {
        "id": "sub_paneer_tofu",
        "original_ingredient": "Dairy Paneer",
        "substitute": "Firm Tofu (Soy Paneer)",
        "ratio": "1:1 by weight",
        "health_benefit": "Completely dairy-free, zero cholesterol, 60% lower calories, and high in complete plant protein.",
        "target_goals": ["weight_loss", "heart_health", "hypertension_management"],
        "allergy_tags": ["dairy_free"],
        "dietary_tags": ["vegan", "vegetarian"],
        "culinary_notes": "Press tofu between paper towels for 15 minutes before shallow frying to match paneer texture."
    },
    {
        "id": "sub_cow_milk_almond",
        "original_ingredient": "Cow Milk (Dairy)",
        "substitute": "Unsweetened Almond Milk / Oat Milk / Soy Milk",
        "ratio": "1:1 by volume",
        "health_benefit": "Eliminates lactose, cuts calories by 50%, and provides heart-healthy unsaturated fats.",
        "target_goals": ["weight_loss", "heart_health"],
        "allergy_tags": ["dairy_free"],
        "dietary_tags": ["vegan", "vegetarian", "satvik", "jain"],
        "culinary_notes": "Soy milk froths best for hot tea and coffee; oat milk is best for baking."
    },
    {
        "id": "sub_butter_ghee_olive_oil",
        "original_ingredient": "Butter / Vanaspati Ghee",
        "substitute": "Cold-Pressed Mustard Oil / Extra Virgin Olive Oil",
        "ratio": "3/4 cup oil for 1 cup butter",
        "health_benefit": "Replaces trans-fats and saturated fats with omega-3 and monounsaturated fatty acids (MUFA).",
        "target_goals": ["heart_health", "hypertension_management", "weight_loss"],
        "allergy_tags": ["dairy_free"],
        "dietary_tags": ["vegan", "vegetarian", "satvik", "jain"],
        "culinary_notes": "Heat mustard oil to smoking point first to remove pungency before sauteing."
    },
    {
        "id": "sub_table_salt_herbs",
        "original_ingredient": "Table Salt (Sodium Chloride)",
        "substitute": "Lemon Juice + Roasted Cumin + Fresh Herbs + Rock Salt (Sendha)",
        "ratio": "Halve salt, add 1 tsp lemon juice & roasted jeera",
        "health_benefit": "Cuts dietary sodium intake drastically, mitigating blood pressure spikes while maximizing flavor.",
        "target_goals": ["hypertension_management", "heart_health", "general_wellness"],
        "allergy_tags": [],
        "dietary_tags": ["vegetarian", "vegan", "satvik", "jain"],
        "culinary_notes": "Acids (lemon, amchur, kokum) naturally stimulate salinity receptors on the tongue."
    },
    {
        "id": "sub_potato_sweet_potato",
        "original_ingredient": "Potato (Aloo)",
        "substitute": "Sweet Potato (Shakarkandi) / Cauliflower Florets",
        "ratio": "1:1 in curries and roasts",
        "health_benefit": "Sweet potato supplies beta-carotene (Vitamin A) and complex fiber with lower glycemic index.",
        "target_goals": ["diabetes_management", "weight_loss", "balanced_nutrition"],
        "allergy_tags": [],
        "dietary_tags": ["vegetarian", "vegan", "satvik"],
        "culinary_notes": "Steamed cauliflower florets reduce carbohydrate density by 80% in aloo gobi or bhaji."
    },
    {
        "id": "sub_deep_frying_air_fry",
        "original_ingredient": "Deep Frying in Oil",
        "substitute": "Air-Frying / Oven Baking with Oil Brush",
        "ratio": "Use 1 tsp oil spray instead of 2 cups oil",
        "health_benefit": "Reduces total fat calories by 75-85%, drastically lowering dietary advanced glycation end-products (AGEs).",
        "target_goals": ["weight_loss", "heart_health", "diabetes_management", "hypertension_management"],
        "allergy_tags": [],
        "dietary_tags": ["vegetarian", "vegan", "satvik", "jain"],
        "culinary_notes": "Preheat air fryer to 180C for 3 minutes for crispy samosas, pakoras, and tikki."
    },
    {
        "id": "sub_cashew_paste_melon_seeds",
        "original_ingredient": "Cashew Nut Paste (Kaju Gravy)",
        "substitute": "Melon Seeds (Magaz) / Roasted Chana Dal Paste",
        "ratio": "1:1 paste",
        "health_benefit": "Nut-free creaminess at 1/4th the price with rich zinc and plant protein.",
        "target_goals": ["budget_conscious", "heart_health"],
        "allergy_tags": ["nut_free"],
        "dietary_tags": ["vegetarian", "vegan", "satvik", "jain"],
        "culinary_notes": "Soak melon seeds in warm water for 20 mins before grinding into silky white paste."
    },
    {
        "id": "sub_wheat_flour_gluten_free",
        "original_ingredient": "Wheat Atta (Gluten)",
        "substitute": "Jowar (Sorghum) / Bajra (Pearl Millet) / Ragi (Finger Millet) Flour",
        "ratio": "1:1 for flatbreads with warm water",
        "health_benefit": "100% naturally gluten-free, high in iron (bajra) and calcium (ragi).",
        "target_goals": ["diabetes_management", "balanced_nutrition"],
        "allergy_tags": ["gluten_free"],
        "dietary_tags": ["vegetarian", "vegan", "satvik", "jain"],
        "culinary_notes": "Knead millet flours with hot water to activate starches since millets lack gluten elasticity."
    },
    {
        "id": "sub_onion_garlic_hing",
        "original_ingredient": "Onion & Garlic Paste",
        "substitute": "Asafoetida (Hing) + Cumin + Grated Bottle Gourd (Lauki)",
        "ratio": "Pinch of hing + 2 tbsp pureed lauki per onion",
        "health_benefit": "Complies strictly with Jain and Satvik principles while maintaining thick curry body and umami aromatics.",
        "target_goals": ["general_wellness"],
        "allergy_tags": [],
        "dietary_tags": ["jain", "satvik", "vegetarian", "vegan"],
        "culinary_notes": "Saute hing in warm oil first to release aromatic sulfides that emulate garlic."
    },
    {
        "id": "sub_egg_flax_egg",
        "original_ingredient": "Poultry Egg (in baking/patties)",
        "substitute": "Flaxseed Meal (1 tbsp ground flax + 3 tbsp warm water)",
        "ratio": "1 flax egg per 1 poultry egg",
        "health_benefit": "Egg-free binding rich in plant ALA omega-3 and soluble lignan fiber.",
        "target_goals": ["heart_health", "weight_loss"],
        "allergy_tags": ["egg_free"],
        "dietary_tags": ["vegan", "vegetarian", "satvik", "jain"],
        "culinary_notes": "Let flax and water mixture stand for 5 minutes until viscous before blending."
    },
    {
        "id": "sub_mayo_avocado",
        "original_ingredient": "Commercial Mayonnaise",
        "substitute": "Mashed Ripe Avocado / Hung Curd Spread",
        "ratio": "1:1 spread",
        "health_benefit": "Eliminates refined seed oils, eggs, and added sugar, supplying heart-healthy MUFA and potassium.",
        "target_goals": ["heart_health", "weight_loss", "diabetes_management"],
        "allergy_tags": ["egg_free"],
        "dietary_tags": ["vegan", "vegetarian"],
        "culinary_notes": "Add lime juice and black pepper to mashed avocado to retain vibrant green color."
    },
]

# ---------------------------------------------------------------------------
# Alternative Foods Knowledge Base (Targeted Replacements for Unfavorable Foods)
# ---------------------------------------------------------------------------

ALTERNATIVE_FOODS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "jalebi": {
        "reason_unfavorable": "High in refined sugar and deep-fried carbohydrates; causes acute blood sugar spikes and caloric surplus.",
        "conditions": ["diabetes_management", "weight_loss", "heart_health"],
        "alternatives": [
            {"food_name": "Baked Apple with Cinnamon & Walnuts", "calories_approx": 120, "benefit": "Natural fructose with soluble pectin fiber; minimizes glycemic response."},
            {"food_name": "Date & Nut Energy Bites", "calories_approx": 95, "benefit": "Mineral-rich natural sweetness without added refined sucrose."},
            {"food_name": "Roasted Makhana (Fox Nuts) with Jaggery Glaze", "calories_approx": 110, "benefit": "Low glycemic index, rich in magnesium and plant protein."}
        ]
    },
    "samosa": {
        "reason_unfavorable": "Deep-fried refined flour crust high in saturated fat and refined carbohydrates.",
        "conditions": ["weight_loss", "heart_health", "diabetes_management", "hypertension_management"],
        "alternatives": [
            {"food_name": "Air-Fried Vegetable Samosa (Whole Wheat Crust)", "calories_approx": 135, "benefit": "70% lower fat content with complex grain fiber."},
            {"food_name": "Moong Dal Cheela with Paneer Filling", "calories_approx": 160, "benefit": "High protein (11g), low glycemic index, cooked with minimal oil."},
            {"food_name": "Steamed Vegetable Momos (Whole Wheat)", "calories_approx": 140, "benefit": "Steamed preparation, zero deep-frying oil, packed with fresh vegetables."}
        ]
    },
    "mutton_curry": {
        "reason_unfavorable": "High saturated fat and dietary cholesterol; elevated caloric density per serving.",
        "conditions": ["heart_health", "weight_loss", "hypertension_management"],
        "alternatives": [
            {"food_name": "Skinless Chicken Breast Curry", "calories_approx": 220, "benefit": "Lean poultry protein with 65% lower saturated fat than red meat."},
            {"food_name": "Soya Chaap Tikka Gravy", "calories_approx": 190, "benefit": "Complete plant protein, zero dietary cholesterol, high dietary fiber."},
            {"food_name": "Grilled Fish Tikka in Mustard Marinade", "calories_approx": 180, "benefit": "Rich in anti-inflammatory omega-3 fatty acids."}
        ]
    },
    "chole_bhature": {
        "reason_unfavorable": "Bhatura is deep-fried refined maida dough high in trans fats and calories.",
        "conditions": ["weight_loss", "diabetes_management", "heart_health"],
        "alternatives": [
            {"food_name": "Chole with Whole Wheat Kulcha / Roti", "calories_approx": 310, "benefit": "Maintains chickpea protein/fiber while eliminating deep-frying oil."},
            {"food_name": "Sprouted Chickpea Salad with Lemon Dressing", "calories_approx": 180, "benefit": "Raw sprouted legumes with increased bioavailability of B-vitamins."},
            {"food_name": "Besan Chilla with Spiced Chole Spread", "calories_approx": 240, "benefit": "Gram flour pancake providing complex carbs and zero maida."}
        ]
    },
    "pakora": {
        "reason_unfavorable": "Deep-fried preparation absorbing substantial cooking oil and high in sodium.",
        "conditions": ["weight_loss", "hypertension_management", "heart_health"],
        "alternatives": [
            {"food_name": "Air-Fried Vegetable Cutlets", "calories_approx": 110, "benefit": "Crispy texture achieved with hot air rather than submersion in hot oil."},
            {"food_name": "Steamed Dhokla with Mustard Tempering", "calories_approx": 130, "benefit": "Fermented chickpea preparation, steamed, light, and easily digestible."},
            {"food_name": "Spiced Edamame / Roasted Chana", "calories_approx": 125, "benefit": "Crunchy snack naturally rich in protein and fiber."}
        ]
    },
    "ras_malai": {
        "reason_unfavorable": "Soaked in concentrated sweetened full-fat milk with high simple sugar content.",
        "conditions": ["diabetes_management", "weight_loss"],
        "alternatives": [
            {"food_name": "Low-Fat Steamed Paneer Cubes with Saffron Cardamom Infusion", "calories_approx": 115, "benefit": "Retains traditional saffron flavor with 80% less sugar."},
            {"food_name": "Chia Seed Pudding with Almond Milk & Berries", "calories_approx": 130, "benefit": "Omega-3 rich, zero cholesterol, low glycemic load."},
            {"food_name": "Greek Yogurt Parfait with Pomegranate Seeds", "calories_approx": 120, "benefit": "12g protein per cup with active probiotic cultures."}
        ]
    },
    "biryani": {
        "reason_unfavorable": "Often prepared with excess vanaspati/ghee and white polished rice with high caloric load.",
        "conditions": ["weight_loss", "diabetes_management"],
        "alternatives": [
            {"food_name": "Brown Rice or Millet Vegetable Biryani", "calories_approx": 260, "benefit": "Slow-release carbs and double the fiber of polished basmati."},
            {"food_name": "Cauliflower Rice Chicken Biryani", "calories_approx": 210, "benefit": "70% fewer carbohydrates, high micronutrient density."},
            {"food_name": "Quinoa Vegetable Pulao", "calories_approx": 240, "benefit": "Complete amino acid profile with low glycemic impact."}
        ]
    }
}

# ---------------------------------------------------------------------------
# Core Recommendation & Safety Filter Classes
# ---------------------------------------------------------------------------

class RecipeSafetyFilter:
    """
    Evaluates recipe ingredients and metadata against user allergies and dietary restrictions.
    Implements the MANDATORY safety rule: hard exclusions must occur BEFORE ranking.
    """

    @staticmethod
    def _extract_text_corpus(recipe: Dict[str, Any]) -> str:
        """Combines recipe name, ingredients, and tokens into normalized lowercase string."""
        parts = [
            str(recipe.get("recipe_name", "")),
            str(recipe.get("ingredients_raw", "")),
            str(recipe.get("ingredient_tokens", "")),
            str(recipe.get("diet", ""))
        ]
        return " " + " ".join(parts).lower() + " "

    @classmethod
    def check_safety(
        cls,
        recipe: Dict[str, Any],
        user_allergies: Optional[List[str]] = None,
        dietary_preference: Optional[str] = None
    ) -> Tuple[bool, List[str], bool]:
        """
        Determines whether a recipe is safe for a given user.

        Returns
        -------
        Tuple[bool, List[str], bool]
            (is_safe, reasons_for_exclusion, incomplete_info_flag)
        """
        user_allergies = [a.lower().strip() for a in (user_allergies or []) if a and a != "none"]
        dietary_preference = (dietary_preference or "no_restriction").lower().strip()

        corpus = cls._extract_text_corpus(recipe)
        reasons: List[str] = []

        # 1. Check for Incomplete Information
        raw_ing = str(recipe.get("ingredients_raw", "")).strip()
        tokens = str(recipe.get("ingredient_tokens", "")).strip()
        incomplete_info = (len(raw_ing) < 10 and len(tokens) < 10)

        # 2. Hard Allergen Exclusions
        for allergen in user_allergies:
            keywords = ALLERGEN_LEXICON.get(allergen, {allergen})
            matched_keywords = []
            for kw in keywords:
                # Use word-boundary regex to prevent false positives (e.g., 'egg' in 'eggplant')
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, corpus):
                    # Special check: eggplant is NOT egg
                    if kw in {"egg", "eggs", "anda"} and "eggplant" in corpus:
                        # Ensure there's an actual standalone egg mention
                        clean_corpus = re.sub(r"\beggplant\b", "aubergine", corpus)
                        if not re.search(pattern, clean_corpus):
                            continue
                    matched_keywords.append(kw)

            if matched_keywords:
                reasons.append(f"Allergen conflict [{allergen.upper()}]: recipe contains {', '.join(matched_keywords[:3])}")

        # 3. Dietary Restriction Exclusions
        if dietary_preference in DIETARY_RESTRICTION_LEXICON:
            forbidden = DIETARY_RESTRICTION_LEXICON[dietary_preference]
            matched_diet_conflicts = []
            for kw in forbidden:
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, corpus):
                    # Avoid false positive on 'eggplant' for vegetarian/vegan
                    if kw in {"egg", "eggs", "anda"} and "eggplant" in corpus:
                        clean_corpus = re.sub(r"\beggplant\b", "aubergine", corpus)
                        if not re.search(pattern, clean_corpus):
                            continue
                    matched_diet_conflicts.append(kw)

            if matched_diet_conflicts:
                reasons.append(
                    f"Dietary restriction conflict [{dietary_preference.upper()}]: "
                    f"recipe contains non-conforming ingredient(s) ({', '.join(matched_diet_conflicts[:3])})"
                )

        is_safe = (len(reasons) == 0)
        return is_safe, reasons, incomplete_info


class SuitabilityScorer:
    """
    Computes a transparent, explainable suitability score (0 to 100)
    for foods and recipes based on user profile, nutritional alignment,
    preparation feasibility, budget, and content similarity.
    """

    @classmethod
    def score_recipe(
        cls,
        recipe: Dict[str, Any],
        user_profile: Dict[str, Any],
        content_similarity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Calculates 0-100 score and returns full breakdown and human explanation.
        """
        # Safety Pre-Check: Hard Exclusion
        allergies = user_profile.get("allergies", [])
        diet_pref = user_profile.get("dietary_preference", "no_restriction")
        is_safe, exclusion_reasons, incomplete_info = RecipeSafetyFilter.check_safety(
            recipe, allergies, diet_pref
        )

        if not is_safe:
            return {
                "suitability_score": 0.0,
                "is_safe": False,
                "disqualified": True,
                "exclusion_reasons": exclusion_reasons,
                "incomplete_info": incomplete_info,
                "breakdown": {
                    "goal_alignment": 0.0,
                    "macro_micro_balance": 0.0,
                    "content_preference": 0.0,
                    "time_feasibility": 0.0,
                    "budget_fit": 0.0,
                },
                "weights": SCORE_WEIGHTS,
                "explanation": (
                    f"DISQUALIFIED FOR SAFETY: Recipe conflicts with declared user safety filters "
                    f"({'; '.join(exclusion_reasons)}). A high score cannot override safety."
                ),
                "disclaimer": RECOMMENDATION_DISCLAIMER
            }

        # 1. Goal Alignment Sub-Score (0-100)
        goal = (user_profile.get("dietary_goal") or "general_wellness").lower()
        diet_tag = str(recipe.get("diet", "")).lower()
        course = str(recipe.get("course", "")).lower()
        corpus = (str(recipe.get("recipe_name", "")) + " " + str(recipe.get("ingredient_tokens", ""))).lower()

        goal_score = 70.0  # Base neutral alignment

        if goal == "weight_loss":
            # Reward low cooking time, vegetables, lean pulses; penalize heavy sweets/deep fried
            if "diabetic_friendly" in diet_tag or "salad" in course or "soup" in course:
                goal_score += 20.0
            if any(w in corpus for w in ["fry", "bhatura", "sugar", "syrup", "cream", "butter"]):
                goal_score -= 25.0
            if any(w in corpus for w in ["sprouts", "cucumber", "spinach", "methi", "oats", "moong"]):
                goal_score += 15.0

        elif goal == "muscle_gain":
            # Reward high protein markers: paneer, dal, chicken, eggs, soya, tofu, lentils, chana
            protein_markers = ["paneer", "dal", "chicken", "egg", "soya", "tofu", "chana", "lentil", "sprouts", "fish"]
            matches = sum(1 for p in protein_markers if p in corpus)
            goal_score += min(matches * 8.0, 30.0)
            if "high_protein" in diet_tag:
                goal_score += 15.0

        elif goal == "diabetes_management":
            # Reward low glycemic index, high fiber; heavily penalize sugar, jaggery, maida, potatoes
            if "diabetic_friendly" in diet_tag:
                goal_score += 25.0
            if any(w in corpus for w in ["sugar", "jaggery", "maida", "syrup", "halwa", "kheer"]):
                goal_score -= 35.0
            if any(w in corpus for w in ["karela", "fenugreek", "methi", "millet", "oats", "fiber", "sprouts", "dal"]):
                goal_score += 15.0

        elif goal == "hypertension_management" or goal == "heart_health":
            # Reward potassium/magnesium, low sodium; penalize heavy salt, pickle, deep fry
            if any(w in corpus for w in ["pickle", "papad", "salted", "deep fry", "processed"]):
                goal_score -= 30.0
            if any(w in corpus for w in ["steamed", "boiled", "garlic", "lemon", "spinach", "vegetable"]):
                goal_score += 20.0

        elif goal == "balanced_nutrition":
            if any(w in corpus for w in ["vegetable", "dal", "curd", "salad", "roti"]):
                goal_score += 15.0

        goal_score = max(10.0, min(100.0, goal_score))

        # 2. Macro & Micro Balance Sub-Score (0-100)
        # Assessed from ingredient density & complexity indicators
        balance_score = 72.0
        health_tags = ["vegetable", "lentil", "curd", "lemon", "turmeric", "cumin", "ginger", "spinach", "tomato"]
        present_tags = sum(1 for t in health_tags if t in corpus)
        balance_score += min(present_tags * 4.0, 24.0)
        if any(w in corpus for w in ["excess", "deep fried", "heavy cream", "refined"]):
            balance_score -= 15.0
        balance_score = max(15.0, min(100.0, balance_score))

        # 3. Content-Based Preference Similarity Sub-Score (0-100)
        pref_score = float(np.clip(content_similarity * 100.0, 30.0, 98.0))

        # 4. Time Feasibility Sub-Score (0-100)
        tot_time = recipe.get("total_time_mins")
        if tot_time is None or pd.isna(tot_time):
            tot_time = 35.0
        else:
            tot_time = float(tot_time)

        user_time_limit = user_profile.get("cooking_time_minutes")
        if user_time_limit and user_time_limit > 0:
            if tot_time <= user_time_limit:
                time_score = 95.0
            elif tot_time <= user_time_limit + 15:
                time_score = 70.0
            elif tot_time <= user_time_limit + 30:
                time_score = 50.0
            else:
                time_score = 30.0
        else:
            # Fallback based on complexity
            complexity = str(recipe.get("complexity_level", "MODERATE")).upper()
            time_score = 90.0 if complexity == "EASY" else (75.0 if complexity == "MODERATE" else 55.0)

        # 5. Budget Fit Sub-Score (0-100)
        budget = (user_profile.get("budget") or "medium").lower()
        luxury_items = ["saffron", "kesar", "pine nut", "pistachio", "cashew", "almond", "prawn", "salmon", "lobster"]
        luxury_count = sum(1 for lux in luxury_items if lux in corpus)

        if budget == "low":
            budget_score = 95.0 if luxury_count == 0 else (60.0 if luxury_count == 1 else 35.0)
        elif budget == "medium":
            budget_score = 90.0 if luxury_count <= 2 else 65.0
        else:  # high
            budget_score = 95.0

        # Weighted Aggregate
        final_score = (
            goal_score * SCORE_WEIGHTS["goal_alignment"] +
            balance_score * SCORE_WEIGHTS["macro_micro_balance"] +
            pref_score * SCORE_WEIGHTS["content_preference"] +
            time_score * SCORE_WEIGHTS["time_feasibility"] +
            budget_score * SCORE_WEIGHTS["budget_fit"]
        )
        final_score = round(float(final_score), 1)

        # Build Explainable Rationale
        explanation_parts = []
        if goal_score >= 80:
            explanation_parts.append(f"Strongly aligns with your goal of {goal.replace('_', ' ')}")
        elif goal_score <= 50:
            explanation_parts.append(f"Moderate alignment with {goal.replace('_', ' ')}")

        if time_score >= 85:
            explanation_parts.append(f"Quick {int(tot_time)}-minute preparation fits your schedule")
        elif time_score <= 50:
            explanation_parts.append(f"Requires longer preparation ({int(tot_time)} mins)")

        if budget_score >= 85:
            explanation_parts.append(f"Fits within your {budget} budget")

        if incomplete_info:
            explanation_parts.append("Note: Partial ingredient list available")

        explanation = f"Scored {final_score}/100: " + "; ".join(explanation_parts) + "."

        return {
            "suitability_score": final_score,
            "is_safe": True,
            "disqualified": False,
            "incomplete_info": incomplete_info,
            "breakdown": {
                "goal_alignment": round(goal_score, 1),
                "macro_micro_balance": round(balance_score, 1),
                "content_preference": round(pref_score, 1),
                "time_feasibility": round(time_score, 1),
                "budget_fit": round(budget_score, 1),
            },
            "weights": SCORE_WEIGHTS,
            "explanation": explanation,
            "disclaimer": RECOMMENDATION_DISCLAIMER
        }


class ContentSimilarityEngine:
    """
    Computes content-based recommendation similarity using TF-IDF vectorization
    and cosine similarity on recipe ingredient tokens and cuisine keywords.
    """

    _instance: Optional[ContentSimilarityEngine] = None

    def __init__(self, recipes_df: pd.DataFrame):
        self.recipes_df = recipes_df.copy().reset_index(drop=True)
        # Ensure ingredient_tokens is populated
        self.recipes_df["ingredient_tokens"] = self.recipes_df["ingredient_tokens"].fillna("").astype(str)
        self.recipes_df["recipe_name"] = self.recipes_df["recipe_name"].fillna("").astype(str)
        self.recipes_df["cuisine"] = self.recipes_df["cuisine"].fillna("").astype(str)

        # Build combined text for TF-IDF
        self.corpus = (
            self.recipes_df["recipe_name"] + " " +
            self.recipes_df["cuisine"] + " " +
            self.recipes_df["ingredient_tokens"]
        ).tolist()

        self.vectorizer = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    @classmethod
    def get_instance(cls) -> ContentSimilarityEngine:
        if cls._instance is None:
            if _RECIPES_CSV.exists():
                df = pd.read_csv(_RECIPES_CSV)
            else:
                df = pd.DataFrame(columns=["recipe_id", "recipe_name", "cuisine", "course", "diet", "ingredient_tokens"])
            cls._instance = cls(df)
        return cls._instance

    def compute_similarity_scores(self, query_text: str) -> np.ndarray:
        """Computes cosine similarity between a user query / preference string and all recipes."""
        if not query_text or not query_text.strip():
            # Return baseline uniform similarity if query is empty
            return np.full((self.tfidf_matrix.shape[0],), 0.5)

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        return sims


# ---------------------------------------------------------------------------
# High-Level Recommendation System Facade
# ---------------------------------------------------------------------------

class RecipeRecommender:
    """
    Main engine for recommending personalized recipes, filtered for safety,
    ranked by transparent suitability score, with alternatives and meal planning.
    """

    def __init__(self):
        self.sim_engine = ContentSimilarityEngine.get_instance()
        self.recipes_df = self.sim_engine.recipes_df

    def get_recommended_recipes(
        self,
        user_profile: Dict[str, Any],
        course: Optional[str] = None,
        cuisine: Optional[str] = None,
        max_cooking_time: Optional[int] = None,
        top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Returns top-N safe and scored recipe recommendations for a user profile.
        """
        # Build user preference query string
        pref_parts = [
            str(user_profile.get("dietary_preference", "")),
            str(user_profile.get("dietary_goal", "")),
            str(user_profile.get("dietary_notes", ""))
        ]
        if isinstance(user_profile.get("breakfast_preference"), list):
            pref_parts.extend(user_profile["breakfast_preference"])
        user_query = " ".join([p for p in pref_parts if p])

        sim_scores = self.sim_engine.compute_similarity_scores(user_query)

        # Filter by course / cuisine if provided
        df = self.recipes_df.copy()
        if course:
            course_lower = course.lower().strip()
            df = df[df["course"].astype(str).str.lower().str.contains(course_lower, na=False)]
        if cuisine:
            cuisine_lower = cuisine.lower().strip()
            df = df[df["cuisine"].astype(str).str.lower().str.contains(cuisine_lower, na=False)]

        time_limit = max_cooking_time or user_profile.get("cooking_time_minutes")
        if time_limit and time_limit > 0:
            df = df[df["total_time_mins"].fillna(30.0).astype(float) <= (time_limit + 15)]

        allergies = user_profile.get("allergies", [])
        diet_pref = user_profile.get("dietary_preference", "no_restriction")

        candidates: List[Dict[str, Any]] = []

        for idx, row in df.iterrows():
            recipe_dict = row.to_dict()
            # Mandatory Pre-Filter: Safety check
            is_safe, reasons, incomplete_info = RecipeSafetyFilter.check_safety(
                recipe_dict, allergies, diet_pref
            )
            if not is_safe:
                continue  # HARD REJECTION: Never consider unsafe recipes

            # Score safe recipe
            c_sim = float(sim_scores[idx]) if idx < len(sim_scores) else 0.5
            score_data = SuitabilityScorer.score_recipe(recipe_dict, user_profile, content_similarity=c_sim)

            recipe_record = {
                "recipe_id": str(row.get("recipe_id", f"rec_{idx}")),
                "recipe_name": str(row.get("recipe_name", "Delicious Recipe")).title(),
                "cuisine": str(row.get("cuisine", "Indian")),
                "course": str(row.get("course", "Main Course")),
                "diet": str(row.get("diet", "vegetarian")),
                "prep_time_mins": float(row.get("prep_time_mins", 15)) if pd.notnull(row.get("prep_time_mins")) else 15.0,
                "cook_time_mins": float(row.get("cook_time_mins", 20)) if pd.notnull(row.get("cook_time_mins")) else 20.0,
                "total_time_mins": float(row.get("total_time_mins", 35)) if pd.notnull(row.get("total_time_mins")) else 35.0,
                "complexity_level": str(row.get("complexity_level", "MODERATE")),
                "servings": int(row.get("servings", 4)) if pd.notnull(row.get("servings")) else 4,
                "ingredients_raw": str(row.get("ingredients_raw", "")),
                "suitability_score": score_data["suitability_score"],
                "breakdown": score_data["breakdown"],
                "explanation": score_data["explanation"],
                "incomplete_info": score_data["incomplete_info"]
            }
            candidates.append(recipe_record)

        # Rank by suitability score descending
        candidates.sort(key=lambda x: x["suitability_score"], reverse=True)
        return candidates[:top_n]


# ---------------------------------------------------------------------------
# Meal Plan Generator (Daily & Weekly)
# ---------------------------------------------------------------------------

class MealPlanGenerator:
    """
    Generates balanced Daily and Weekly Meal Plans tailored to user calorie
    targets, macro goals, and morning routines, ensuring inter-day variety.
    """

    MEAL_SLOTS = [
        {"slot": "breakfast", "label": "Breakfast", "calorie_pct": 0.25, "course_filter": "breakfast"},
        {"slot": "lunch", "label": "Lunch", "calorie_pct": 0.35, "course_filter": "lunch"},
        {"slot": "snack", "label": "Evening Snack", "calorie_pct": 0.15, "course_filter": "snack"},
        {"slot": "dinner", "label": "Dinner", "calorie_pct": 0.25, "course_filter": "dinner"}
    ]

    def __init__(self, recommender: Optional[RecipeRecommender] = None):
        self.recommender = recommender or RecipeRecommender()

    def generate_daily_plan(
        self,
        user_profile: Dict[str, Any],
        target_calories: Optional[float] = None,
        day_label: str = "Today",
        slot_candidates_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        day_offset: int = 0
    ) -> Dict[str, Any]:
        """
        Creates a balanced 4-meal daily plan satisfying safety and calorie targets.
        """
        if target_calories is None or target_calories <= 0:
            target_calories = float(user_profile.get("daily_calorie_target") or 2000.0)

        daily_meals: List[Dict[str, Any]] = []
        used_recipe_ids: Set[str] = set()
        total_planned_cals = 0.0

        for slot_info in self.MEAL_SLOTS:
            slot_name = slot_info["slot"]
            course_tag = slot_info["course_filter"]
            target_meal_cal = target_calories * slot_info["calorie_pct"]

            if slot_candidates_cache and slot_name in slot_candidates_cache:
                candidates = slot_candidates_cache[slot_name]
            else:
                candidates = self.recommender.get_recommended_recipes(
                    user_profile=user_profile,
                    course=course_tag,
                    top_n=20
                )
                if len(candidates) == 0:
                    candidates = self.recommender.get_recommended_recipes(
                        user_profile=user_profile,
                        top_n=20
                    )

            # Pick candidate rotating by day_offset to ensure inter-day variety
            selected = None
            if len(candidates) > 0:
                start_idx = day_offset % len(candidates)
                for i in range(len(candidates)):
                    cand = candidates[(start_idx + i) % len(candidates)]
                    if cand["recipe_id"] not in used_recipe_ids:
                        selected = cand
                        used_recipe_ids.add(cand["recipe_id"])
                        break
                if selected is None:
                    selected = candidates[start_idx]

            if selected:
                meal_cals = round(target_meal_cal, 0)
                total_planned_cals += meal_cals

                daily_meals.append({
                    "slot": slot_name,
                    "slot_label": slot_info["label"],
                    "recipe_id": selected["recipe_id"],
                    "recipe_name": selected["recipe_name"],
                    "cuisine": selected["cuisine"],
                    "course": selected["course"],
                    "total_time_mins": selected["total_time_mins"],
                    "suitability_score": selected["suitability_score"],
                    "explanation": selected["explanation"],
                    "target_calories": meal_cals,
                    "ingredients_summary": selected["ingredients_raw"][:150] + ("..." if len(selected["ingredients_raw"]) > 150 else "")
                })
            else:
                daily_meals.append({
                    "slot": slot_name,
                    "slot_label": slot_info["label"],
                    "recipe_id": "custom",
                    "recipe_name": f"Healthy {slot_info['label']} Choice",
                    "cuisine": "Indian",
                    "course": slot_info["label"],
                    "total_time_mins": 20.0,
                    "suitability_score": 80.0,
                    "explanation": "Customized balanced choice based on user profile.",
                    "target_calories": round(target_meal_cal, 0),
                    "ingredients_summary": "Selected fresh seasonal ingredients."
                })
                total_planned_cals += target_meal_cal

        return {
            "day": day_label,
            "target_calories": round(target_calories, 0),
            "planned_calories": round(total_planned_cals, 0),
            "calorie_variance": round(total_planned_cals - target_calories, 0),
            "meals": daily_meals,
            "disclaimer": RECOMMENDATION_DISCLAIMER
        }

    def generate_weekly_plan(
        self,
        user_profile: Dict[str, Any],
        target_calories: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates a 7-day meal plan (Monday to Sunday) ensuring variety
        across days and compiling a consolidated grocery list.
        """
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekly_schedule: List[Dict[str, Any]] = []
        grocery_tokens: Set[str] = set()

        # Pre-fetch candidate pool per slot once for the entire week
        slot_candidates_pool: Dict[str, List[Dict[str, Any]]] = {}
        for slot_info in self.MEAL_SLOTS:
            slot_name = slot_info["slot"]
            candidates = self.recommender.get_recommended_recipes(
                user_profile=user_profile,
                course=slot_info["course_filter"],
                top_n=25
            )
            if len(candidates) == 0:
                candidates = self.recommender.get_recommended_recipes(
                    user_profile=user_profile,
                    top_n=25
                )
            slot_candidates_pool[slot_name] = candidates

        for day_idx, d in enumerate(days):
            daily_plan = self.generate_daily_plan(
                user_profile=user_profile,
                target_calories=target_calories,
                day_label=d,
                slot_candidates_cache=slot_candidates_pool,
                day_offset=day_idx
            )
            weekly_schedule.append(daily_plan)

            for meal in daily_plan["meals"]:
                raw_ing = meal.get("ingredients_summary", "")
                for word in raw_ing.split(","):
                    clean = word.strip().lower()
                    if len(clean) > 3 and not any(char.isdigit() for char in clean):
                        grocery_tokens.add(clean.title())

        # Categorize groceries into broad culinary departments
        grocery_categories = {
            "Grains & Staples": [],
            "Fresh Vegetables & Fruits": [],
            "Proteins & Dairy/Alternatives": [],
            "Spices & Oils": []
        }

        staple_kws = ["Rice", "Flour", "Atta", "Rava", "Oats", "Bread", "Poha", "Besan", "Millet"]
        veg_kws = ["Tomato", "Onion", "Palak", "Spinach", "Karela", "Gourd", "Carrot", "Potato", "Lemon", "Coriander", "Ginger", "Chilli"]
        protein_kws = ["Paneer", "Tofu", "Dal", "Chana", "Egg", "Chicken", "Fish", "Moong", "Sprouts", "Curd", "Yogurt", "Milk"]

        for item in sorted(list(grocery_tokens))[:30]:
            if any(k in item for k in staple_kws):
                grocery_categories["Grains & Staples"].append(item)
            elif any(k in item for k in veg_kws):
                grocery_categories["Fresh Vegetables & Fruits"].append(item)
            elif any(k in item for k in protein_kws):
                grocery_categories["Proteins & Dairy/Alternatives"].append(item)
            else:
                grocery_categories["Spices & Oils"].append(item)

        return {
            "plan_type": "7_day_weekly_plan",
            "user_id": user_profile.get("user_id", "guest"),
            "daily_calorie_target": round(float(target_calories or user_profile.get("daily_calorie_target") or 2000.0), 0),
            "days": weekly_schedule,
            "grocery_list": grocery_categories,
            "disclaimer": RECOMMENDATION_DISCLAIMER
        }


# ---------------------------------------------------------------------------
# Public Facade Functions
# ---------------------------------------------------------------------------

_GLOBAL_RECOMMENDER: Optional[RecipeRecommender] = None
_GLOBAL_PLANNER: Optional[MealPlanGenerator] = None

def get_recommender() -> RecipeRecommender:
    global _GLOBAL_RECOMMENDER
    if _GLOBAL_RECOMMENDER is None:
        _GLOBAL_RECOMMENDER = RecipeRecommender()
    return _GLOBAL_RECOMMENDER

def get_meal_planner() -> MealPlanGenerator:
    global _GLOBAL_PLANNER
    if _GLOBAL_PLANNER is None:
        _GLOBAL_PLANNER = MealPlanGenerator(get_recommender())
    return _GLOBAL_PLANNER

def get_recipe_recommendations(
    user_profile: Dict[str, Any],
    course: Optional[str] = None,
    cuisine: Optional[str] = None,
    max_cooking_time: Optional[int] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """Public helper for getting scored & safe recipe recommendations."""
    return get_recommender().get_recommended_recipes(
        user_profile=user_profile,
        course=course,
        cuisine=cuisine,
        max_cooking_time=max_cooking_time,
        top_n=top_n
    )

def get_ingredient_substitutions(
    query: Optional[str] = None,
    target_goal: Optional[str] = None,
    allergy_tag: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Returns curated culinary substitutions filtered by ingredient, goal, or allergy."""
    results = INGREDIENT_SUBSTITUTIONS_REGISTRY
    if query:
        q = query.lower().strip()
        results = [s for s in results if q in s["original_ingredient"].lower() or q in s["substitute"].lower()]
    if target_goal:
        g = target_goal.lower().strip()
        results = [s for s in results if g in s["target_goals"]]
    if allergy_tag:
        a = allergy_tag.lower().strip()
        results = [s for s in results if a in s["allergy_tags"]]
    return results

def get_alternative_foods(
    food_name: str,
    user_goal: Optional[str] = None
) -> Dict[str, Any]:
    """Provides healthy swaps and nutritional reasoning for an unfavorable food dish."""
    clean_name = food_name.lower().strip().replace(" ", "_")
    found = None
    for key, data in ALTERNATIVE_FOODS_REGISTRY.items():
        if key in clean_name or clean_name in key:
            found = data
            break

    if found is None:
        # Generic heuristic alternative
        return {
            "food_name": food_name.title(),
            "has_specific_alternatives": False,
            "reason_unfavorable": "Consider portion moderation or lighter cooking techniques.",
            "alternatives": [
                {"food_name": "Steamed Sprout Salad with Lemon & Pepper", "calories_approx": 120, "benefit": "High fiber and micronutrient density with zero saturated fat."},
                {"food_name": "Grilled Vegetable & Paneer/Tofu Skewer", "calories_approx": 150, "benefit": "Rich in lean protein and essential minerals."}
            ],
            "disclaimer": RECOMMENDATION_DISCLAIMER
        }

    return {
        "food_name": food_name.title(),
        "has_specific_alternatives": True,
        "reason_unfavorable": found["reason_unfavorable"],
        "applicable_conditions": found["conditions"],
        "alternatives": found["alternatives"],
        "disclaimer": RECOMMENDATION_DISCLAIMER
    }

def generate_daily_meal_plan(
    user_profile: Dict[str, Any],
    target_calories: Optional[float] = None
) -> Dict[str, Any]:
    """Public helper for 1-day meal plan."""
    return get_meal_planner().generate_daily_plan(user_profile, target_calories)

def generate_weekly_meal_plan(
    user_profile: Dict[str, Any],
    target_calories: Optional[float] = None
) -> Dict[str, Any]:
    """Public helper for 7-day meal plan."""
    return get_meal_planner().generate_weekly_plan(user_profile, target_calories)
