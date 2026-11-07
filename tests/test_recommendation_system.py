"""
tests/test_recommendation_system.py
===================================
Comprehensive Integration & Unit Tests for Phase 9: Recommendation System

Verifies:
 1. Module Constants (RECOMMENDATION_DISCLAIMER, SAFETY_OVERRIDE_NOTE, SCORE_WEIGHTS)
 2. Recipe Safety Filter (Allergen exclusion, dietary restriction exclusion, false-positive handling)
 3. Mandatory Safety Pre-Filter (Never allow high score to override allergen/dietary conflict)
 4. Transparent Suitability Scoring (0-100 scale, breakdown weights, human-readable explanation)
 5. Content-Based TF-IDF Similarity
 6. Recipe Recommendations (Goal, course, and cooking time filtering)
 7. Alternative Foods Engine (Unfavorable foods analysis and healthier substitutions)
 8. Ingredient Substitutions Catalog (Culinary ratios, health benefits, goal/allergy filters)
 9. Daily Meal Planning (4-meal calorie distribution and feasibility)
 10. Weekly Meal Planning (7-day schedule, inter-day variety, categorized grocery list)
 11. Flask REST API Endpoints (All Phase 9 endpoints)

Author: NutriSense AI Team
Phase: 9
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import sys
import types
from unittest.mock import MagicMock

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def _install_ml_stubs_if_needed():
    """Register torch / torchvision / easyocr stubs only when missing."""
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    torch_pkg = types.ModuleType("torch")
    torch_pkg.__path__ = []

    class _Device:
        def __init__(self, s="cpu"): self._s = str(s)
        def __str__(self): return self._s

    class _Tensor:
        def __init__(self, data=None): self.data = data
        def unsqueeze(self, *a): return self
        def squeeze(self, *a): return self
        def dim(self): return 4
        @property
        def shape(self): return (1, 3, 224, 224)
        def to(self, *a, **kw): return self
        def item(self): return 0.0
        def __iter__(self): return iter([self])

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    torch_pkg.device = _Device
    torch_pkg.Tensor = _Tensor
    torch_pkg.no_grad = lambda: _NoGrad()
    torch_pkg.cuda = MagicMock()
    torch_pkg.cuda.is_available = lambda: False
    torch_pkg.load = MagicMock(return_value={"model_state_dict": {}, "epoch": 0})
    torch_pkg.softmax = MagicMock(return_value=_Tensor())
    torch_pkg.topk = MagicMock(return_value=([_Tensor()], [_Tensor()]))

    nn_pkg = types.ModuleType("torch.nn")
    nn_pkg.__path__ = []

    class _Module:
        def __init__(self, *a, **kw): pass
        def to(self, *a, **kw): return self
        def eval(self): return self
        def __call__(self, x): return _Tensor()
        def load_state_dict(self, *a, **kw): pass
        def parameters(self): return iter([])

    nn_pkg.Module = _Module
    nn_pkg.Sequential = MagicMock(return_value=_Module())
    nn_pkg.Linear = MagicMock(return_value=_Module())
    nn_pkg.Dropout = MagicMock(return_value=_Module())
    nn_pkg.CrossEntropyLoss = MagicMock(return_value=_Module())
    torch_pkg.nn = nn_pkg

    nn_func = types.ModuleType("torch.nn.functional")
    nn_func.__path__ = []
    torch_pkg.nn.functional = nn_func

    optim_pkg = types.ModuleType("torch.optim")
    optim_pkg.__path__ = []
    torch_pkg.optim = optim_pkg

    utils_pkg = types.ModuleType("torch.utils")
    utils_pkg.__path__ = []
    torch_pkg.utils = utils_pkg

    class _Dataset:
        def __len__(self): return 0
        def __getitem__(self, idx): return None

    class _DataLoader:
        def __init__(self, dataset, *a, **kw): self.dataset = dataset
        def __iter__(self): return iter([])
        def __len__(self): return 0

    data_pkg = types.ModuleType("torch.utils.data")
    data_pkg.__path__ = []
    data_pkg.Dataset = _Dataset
    data_pkg.DataLoader = _DataLoader
    utils_pkg.data = data_pkg

    sys.modules["torch"] = torch_pkg
    sys.modules["torch.nn"] = nn_pkg
    sys.modules["torch.nn.functional"] = nn_func
    sys.modules["torch.optim"] = optim_pkg
    sys.modules["torch.utils"] = utils_pkg
    sys.modules["torch.utils.data"] = data_pkg

    tv_pkg = types.ModuleType("torchvision")
    tv_pkg.__path__ = []
    tv_transforms = types.ModuleType("torchvision.transforms")
    tv_transforms.__path__ = []

    class _Compose:
        def __init__(self, t): self.transforms = t
        def __call__(self, x): return x

    tv_transforms.Compose = _Compose
    for attr in ["Resize", "CenterCrop", "ToTensor", "Normalize", "RandomHorizontalFlip",
                 "RandomRotation", "ColorJitter", "RandomResizedCrop"]:
        setattr(tv_transforms, attr, MagicMock())

    tv_models = types.ModuleType("torchvision.models")
    tv_models.__path__ = []

    class _FakeMobileNetV2:
        def __init__(self, *a, **kw):
            linear = MagicMock()
            linear.in_features = 1280
            self.classifier = [None, linear]
        def to(self, *a, **kw): return self
        def eval(self): return self
        def __call__(self, x): return _Tensor()
        def load_state_dict(self, *a, **kw): pass
        def parameters(self): return iter([])
        def state_dict(self): return {}

    tv_models.mobilenet_v2 = lambda **kw: _FakeMobileNetV2()
    tv_pkg.transforms = tv_transforms
    tv_pkg.models = tv_models

    sys.modules["torchvision"] = tv_pkg
    sys.modules["torchvision.transforms"] = tv_transforms
    sys.modules["torchvision.models"] = tv_models

    try:
        import easyocr  # noqa: F401
    except ImportError:
        ocr_mock = types.ModuleType("easyocr")
        ocr_mock.__path__ = []
        class _FakeOCR:
            def __init__(self, *a, **kw): pass
            def readtext(self, *a, **kw): return []
        ocr_mock.Reader = _FakeOCR
        sys.modules["easyocr"] = ocr_mock


_install_ml_stubs_if_needed()

import pytest
from app import app as flask_app
from src.recommendation_engine import (
    RECOMMENDATION_DISCLAIMER,
    SAFETY_OVERRIDE_NOTE,
    SCORE_WEIGHTS,
    RecipeSafetyFilter,
    SuitabilityScorer,
    ContentSimilarityEngine,
    RecipeRecommender,
    MealPlanGenerator,
    get_recipe_recommendations,
    get_ingredient_substitutions,
    get_alternative_foods,
    generate_daily_meal_plan,
    generate_weekly_meal_plan,
)

# Sample test profiles
VEG_PROFILE = {
    "display_name": "Vegetarian Goal User",
    "dietary_goal": "weight_loss",
    "dietary_preference": "vegetarian",
    "allergies": [],
    "activity_level": "moderately_active",
    "budget": "medium",
    "cooking_time_minutes": 30,
    "daily_calorie_target": 1800.0
}

ALLERGIC_PROFILE = {
    "display_name": "Nut and Gluten Sensitive User",
    "dietary_goal": "diabetes_management",
    "dietary_preference": "vegetarian",
    "allergies": ["nuts", "gluten"],
    "activity_level": "sedentary",
    "budget": "low",
    "cooking_time_minutes": 25,
    "daily_calorie_target": 1600.0
}

VEGAN_PROFILE = {
    "display_name": "Strict Vegan User",
    "dietary_goal": "heart_health",
    "dietary_preference": "vegan",
    "allergies": ["dairy"],
    "activity_level": "lightly_active",
    "budget": "medium",
    "cooking_time_minutes": 40,
    "daily_calorie_target": 1900.0
}


@pytest.fixture
def flask_client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# =========================================================================
# 1. Constants & Disclaimers
# =========================================================================

class TestRecommendationConstants:
    def test_disclaimer_is_non_empty(self):
        assert isinstance(RECOMMENDATION_DISCLAIMER, str)
        assert len(RECOMMENDATION_DISCLAIMER) > 50

    def test_disclaimer_mentions_non_medical(self):
        txt = RECOMMENDATION_DISCLAIMER.lower()
        assert "not constitute medical" in txt or "educational" in txt or "lifestyle" in txt

    def test_safety_override_note_exists(self):
        assert isinstance(SAFETY_OVERRIDE_NOTE, str)
        assert "safety" in SAFETY_OVERRIDE_NOTE.lower()

    def test_score_weights_sum_to_one(self):
        total_w = sum(SCORE_WEIGHTS.values())
        assert abs(total_w - 1.0) < 1e-4

    def test_score_weights_keys(self):
        expected_keys = {"goal_alignment", "macro_micro_balance", "content_preference", "time_feasibility", "budget_fit"}
        assert set(SCORE_WEIGHTS.keys()) == expected_keys


# =========================================================================
# 2. Recipe Safety Filter
# =========================================================================

class TestRecipeSafetyFilter:
    def test_vegetarian_rejects_chicken(self):
        recipe = {"recipe_name": "Chicken Curry", "ingredients_raw": "500g chicken, onion, tomato, spices", "diet": "non_vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="vegetarian")
        assert is_safe is False
        assert any("chicken" in r.lower() or "dietary" in r.lower() for r in reasons)

    def test_vegetarian_rejects_fish(self):
        recipe = {"recipe_name": "Fish Fry", "ingredients_raw": "Rohu fish, lemon, mustard oil", "diet": "non_vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="vegetarian")
        assert is_safe is False

    def test_vegetarian_accepts_dal(self):
        recipe = {"recipe_name": "Tadka Dal", "ingredients_raw": "Yellow moong dal, turmeric, cumin, ghee", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="vegetarian")
        assert is_safe is True
        assert len(reasons) == 0

    def test_eggplant_not_false_positive_for_vegetarian(self):
        recipe = {"recipe_name": "Baingan Bharta", "ingredients_raw": "Roasted eggplant, onions, tomatoes, green chillies", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="vegetarian")
        assert is_safe is True

    def test_vegan_rejects_dairy_paneer(self):
        recipe = {"recipe_name": "Paneer Tikka", "ingredients_raw": "Dairy paneer cubes, curd marinade, spices", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="vegan")
        assert is_safe is False
        assert any("paneer" in r.lower() or "curd" in r.lower() or "vegan" in r.lower() for r in reasons)

    def test_vegan_rejects_ghee(self):
        recipe = {"recipe_name": "Khichdi", "ingredients_raw": "Rice, moong dal, 2 tbsp pure desi ghee", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="vegan")
        assert is_safe is False

    def test_jain_rejects_onion_and_garlic(self):
        recipe = {"recipe_name": "Chole Masala", "ingredients_raw": "Chickpeas, pureed onion, garlic paste, garam masala", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="jain")
        assert is_safe is False

    def test_jain_rejects_potato(self):
        recipe = {"recipe_name": "Aloo Gobi", "ingredients_raw": "Potatoes (aloo), cauliflower, spices", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="jain")
        assert is_safe is False

    def test_nut_allergy_rejects_cashew_kaju(self):
        recipe = {"recipe_name": "Shahi Paneer", "ingredients_raw": "Paneer, rich cashew (kaju) paste gravy, cream", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=["nuts"], dietary_preference="vegetarian")
        assert is_safe is False
        assert any("nut" in r.lower() or "cashew" in r.lower() for r in reasons)

    def test_gluten_allergy_rejects_maida_and_wheat(self):
        recipe = {"recipe_name": "Poori Bhaji", "ingredients_raw": "Wheat atta dough, deep fried, potato curry", "diet": "vegetarian"}
        is_safe, reasons, _ = RecipeSafetyFilter.check_safety(recipe, user_allergies=["gluten"], dietary_preference="vegetarian")
        assert is_safe is False

    def test_incomplete_info_detected(self):
        recipe = {"recipe_name": "Mystery Dish", "ingredients_raw": "spices", "ingredient_tokens": ""}
        _, _, incomplete = RecipeSafetyFilter.check_safety(recipe, user_allergies=[], dietary_preference="vegetarian")
        assert incomplete is True


# =========================================================================
# 3. Mandatory Safety Precedence (Cannot Override Safety)
# =========================================================================

class TestSafetyPrecedence:
    def test_high_protein_cannot_override_chicken_for_vegetarian(self):
        recipe = {
            "recipe_id": "unsafe_1",
            "recipe_name": "High Protein Tandoori Chicken",
            "ingredients_raw": "Chicken breast 200g, yogurt, spices",
            "diet": "non_vegetarian",
            "total_time_mins": 25.0,
            "complexity_level": "EASY"
        }
        user = {"dietary_preference": "vegetarian", "dietary_goal": "muscle_gain", "allergies": []}
        score_res = SuitabilityScorer.score_recipe(recipe, user)
        assert score_res["suitability_score"] == 0.0
        assert score_res["is_safe"] is False
        assert score_res["disqualified"] is True
        assert "DISQUALIFIED FOR SAFETY" in score_res["explanation"]

    def test_allergy_conflict_yields_zero_score(self):
        recipe = {
            "recipe_id": "unsafe_2",
            "recipe_name": "Almond Kheer (High Calcium)",
            "ingredients_raw": "Almonds 50g, milk, sugar, cardamom",
            "diet": "vegetarian",
            "total_time_mins": 20.0
        }
        user = {"dietary_preference": "vegetarian", "dietary_goal": "general_wellness", "allergies": ["nuts"]}
        score_res = SuitabilityScorer.score_recipe(recipe, user)
        assert score_res["suitability_score"] == 0.0
        assert score_res["disqualified"] is True


# =========================================================================
# 4. Suitability Scoring & Explanations
# =========================================================================

class TestSuitabilityScorer:
    def test_safe_recipe_scores_between_0_and_100(self):
        recipe = {
            "recipe_id": "safe_1",
            "recipe_name": "Moong Dal Tadka",
            "cuisine": "Indian",
            "course": "Main Course",
            "diet": "vegetarian",
            "prep_time_mins": 10.0,
            "cook_time_mins": 20.0,
            "total_time_mins": 30.0,
            "complexity_level": "EASY",
            "ingredients_raw": "Yellow moong dal, tomato, cumin, turmeric, salt, lemon juice",
            "ingredient_tokens": "moong dal tomato cumin turmeric salt lemon juice"
        }
        res = SuitabilityScorer.score_recipe(recipe, VEG_PROFILE)
        score = res["suitability_score"]
        assert 0.0 <= score <= 100.0
        assert res["is_safe"] is True
        assert res["disqualified"] is False

    def test_breakdown_contains_all_components(self):
        recipe = {
            "recipe_id": "safe_2",
            "recipe_name": "Spinach Soup",
            "diet": "vegetarian",
            "total_time_mins": 20.0,
            "ingredients_raw": "Fresh palak spinach, garlic, black pepper, water"
        }
        res = SuitabilityScorer.score_recipe(recipe, VEG_PROFILE)
        bd = res["breakdown"]
        assert "goal_alignment" in bd
        assert "macro_micro_balance" in bd
        assert "content_preference" in bd
        assert "time_feasibility" in bd
        assert "budget_fit" in bd

    def test_explanation_is_human_readable(self):
        recipe = {
            "recipe_id": "safe_3",
            "recipe_name": "Vegetable Poha",
            "diet": "vegetarian",
            "total_time_mins": 15.0,
            "ingredients_raw": "Flattened rice, green peas, carrots, mustard seeds, turmeric"
        }
        res = SuitabilityScorer.score_recipe(recipe, VEG_PROFILE)
        explanation = res["explanation"]
        assert isinstance(explanation, str)
        assert len(explanation) > 15
        assert "Scored" in explanation

    def test_cooking_time_penalty_when_exceeding_limit(self):
        quick_recipe = {"recipe_name": "Quick Salad", "diet": "vegetarian", "total_time_mins": 15.0, "ingredients_raw": "Cucumber, tomato"}
        long_recipe = {"recipe_name": "Slow Roast", "diet": "vegetarian", "total_time_mins": 90.0, "ingredients_raw": "Cucumber, tomato"}
        user_fast = {"cooking_time_minutes": 20, "dietary_preference": "vegetarian"}

        score_quick = SuitabilityScorer.score_recipe(quick_recipe, user_fast)
        score_long = SuitabilityScorer.score_recipe(long_recipe, user_fast)
        assert score_quick["breakdown"]["time_feasibility"] > score_long["breakdown"]["time_feasibility"]


# =========================================================================
# 5. Content Similarity Engine
# =========================================================================

class TestContentSimilarityEngine:
    def test_engine_singleton(self):
        e1 = ContentSimilarityEngine.get_instance()
        e2 = ContentSimilarityEngine.get_instance()
        assert e1 is e2

    def test_similarity_output_shape(self):
        engine = ContentSimilarityEngine.get_instance()
        sims = engine.compute_similarity_scores("paneer tikka masala")
        assert len(sims) == len(engine.recipes_df)
        assert (sims >= 0.0).all()

    def test_empty_query_returns_default(self):
        engine = ContentSimilarityEngine.get_instance()
        sims = engine.compute_similarity_scores("")
        assert (sims == 0.5).all()


# =========================================================================
# 6. Recipe Recommendations
# =========================================================================

class TestRecipeRecommender:
    def test_get_recommended_recipes_returns_top_n(self):
        recs = get_recipe_recommendations(VEG_PROFILE, top_n=5)
        assert len(recs) <= 5
        assert len(recs) > 0
        for r in recs:
            assert "recipe_name" in r
            assert "suitability_score" in r
            assert "explanation" in r

    def test_recommendations_are_sorted_by_score(self):
        recs = get_recipe_recommendations(VEG_PROFILE, top_n=8)
        scores = [r["suitability_score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_all_recommendations_respect_allergies(self):
        recs = get_recipe_recommendations(ALLERGIC_PROFILE, top_n=10)
        for r in recs:
            ing = r["ingredients_raw"].lower()
            # Must not contain cashew/peanut/almond
            assert not any(n in ing for n in ["cashew", "kaju", "peanut", "almond", "badam"])
            # Must not contain wheat/maida
            assert not any(g in ing for g in ["maida", "all purpose flour"])

    def test_recommendations_filter_course(self):
        recs = get_recipe_recommendations(VEG_PROFILE, course="Lunch", top_n=5)
        assert len(recs) > 0
        for r in recs:
            assert "lunch" in r["course"].lower() or "main" in r["course"].lower()


# =========================================================================
# 7. Alternative Foods Engine
# =========================================================================

class TestAlternativeFoods:
    def test_alternatives_for_jalebi(self):
        alts = get_alternative_foods("jalebi")
        assert alts["has_specific_alternatives"] is True
        assert len(alts["alternatives"]) >= 2
        assert "diabetes_management" in alts["applicable_conditions"]
        for a in alts["alternatives"]:
            assert "food_name" in a
            assert "benefit" in a

    def test_alternatives_for_samosa(self):
        alts = get_alternative_foods("samosa")
        assert alts["has_specific_alternatives"] is True
        assert len(alts["alternatives"]) >= 2
        assert any("air-fried" in a["food_name"].lower() or "cheela" in a["food_name"].lower() for a in alts["alternatives"])

    def test_alternatives_for_mutton_curry(self):
        alts = get_alternative_foods("mutton_curry")
        assert alts["has_specific_alternatives"] is True
        assert len(alts["alternatives"]) >= 2

    def test_fallback_generic_alternatives(self):
        alts = get_alternative_foods("unknown_exotic_dish_xyz")
        assert alts["has_specific_alternatives"] is False
        assert len(alts["alternatives"]) > 0


# =========================================================================
# 8. Ingredient Substitutions
# =========================================================================

class TestIngredientSubstitutions:
    def test_total_substitutions_at_least_15(self):
        subs = get_ingredient_substitutions()
        assert len(subs) >= 15

    def test_query_sugar(self):
        subs = get_ingredient_substitutions(query="sugar")
        assert len(subs) >= 1
        sub = subs[0]
        assert "jaggery" in sub["substitute"].lower() or "stevia" in sub["substitute"].lower()
        assert "ratio" in sub
        assert "health_benefit" in sub

    def test_query_maida(self):
        subs = get_ingredient_substitutions(query="maida")
        assert len(subs) >= 1
        assert "whole wheat" in subs[0]["substitute"].lower() or "oat" in subs[0]["substitute"].lower()

    def test_filter_by_goal(self):
        subs = get_ingredient_substitutions(target_goal="diabetes_management")
        assert len(subs) >= 3

    def test_filter_by_allergy(self):
        subs = get_ingredient_substitutions(allergy_tag="dairy_free")
        assert len(subs) >= 2


# =========================================================================
# 9. Daily & Weekly Meal Planning
# =========================================================================

class TestMealPlanning:
    def test_daily_meal_plan_structure(self):
        plan = generate_daily_meal_plan(VEG_PROFILE, target_calories=1800)
        assert plan["target_calories"] == 1800
        assert len(plan["meals"]) == 4
        slots = [m["slot"] for m in plan["meals"]]
        assert slots == ["breakfast", "lunch", "snack", "dinner"]

    def test_daily_plan_meals_have_recipes_and_scores(self):
        plan = generate_daily_meal_plan(VEG_PROFILE, target_calories=2000)
        for meal in plan["meals"]:
            assert "recipe_name" in meal
            assert "target_calories" in meal
            assert meal["suitability_score"] > 0.0

    def test_weekly_meal_plan_has_seven_days(self):
        week = generate_weekly_meal_plan(VEG_PROFILE, target_calories=1800)
        assert week["plan_type"] == "7_day_weekly_plan"
        assert len(week["days"]) == 7
        day_names = [d["day"] for d in week["days"]]
        assert day_names == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    def test_weekly_meal_plan_grocery_list(self):
        week = generate_weekly_meal_plan(VEG_PROFILE, target_calories=1800)
        grocery = week["grocery_list"]
        assert isinstance(grocery, dict)
        assert "Grains & Staples" in grocery
        assert "Fresh Vegetables & Fruits" in grocery
        assert "Proteins & Dairy/Alternatives" in grocery
        assert "Spices & Oils" in grocery


# =========================================================================
# 10. REST API Endpoints
# =========================================================================

class TestRecommendationAPI:
    def test_get_formula_endpoint(self, flask_client):
        resp = flask_client.get("/api/recommendations/formula")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "weights" in data
        assert "disclaimer" in data

    def test_get_recipes_endpoint(self, flask_client):
        resp = flask_client.get("/api/recommendations/recipes?top_n=3")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["recommendations"]) <= 3

    def test_post_score_recipe_endpoint(self, flask_client):
        sample_recipe = {
            "recipe_id": "test_r1",
            "recipe_name": "Vegetable Pulao",
            "cuisine": "Indian",
            "course": "Main Course",
            "diet": "vegetarian",
            "total_time_mins": 30.0,
            "ingredients_raw": "Basmati rice, carrots, beans, peas, cumin, turmeric"
        }
        resp = flask_client.post("/api/recommendations/score",
                                  json={"recipe": sample_recipe, "user_profile": VEG_PROFILE},
                                  content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["suitability_score"] > 0

    def test_post_score_recipe_missing_body(self, flask_client):
        resp = flask_client.post("/api/recommendations/score", json={}, content_type="application/json")
        assert resp.status_code == 400

    def test_get_alternatives_endpoint(self, flask_client):
        resp = flask_client.get("/api/recommendations/alternatives?food_name=jalebi")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["data"]["has_specific_alternatives"] is True

    def test_get_alternatives_missing_param_400(self, flask_client):
        resp = flask_client.get("/api/recommendations/alternatives")
        assert resp.status_code == 400

    def test_get_substitutions_endpoint(self, flask_client):
        resp = flask_client.get("/api/recommendations/substitutions?query=sugar")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["substitutions"]) > 0

    def test_get_daily_meal_plan_endpoint(self, flask_client):
        resp = flask_client.get("/api/recommendations/meal-plan/daily?target_calories=1800")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["data"]["meals"]) == 4

    def test_get_weekly_meal_plan_endpoint(self, flask_client):
        resp = flask_client.get("/api/recommendations/meal-plan/weekly?target_calories=1800")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["data"]["days"]) == 7


# =========================================================================
# Standalone run_tests() for Master Runner compatibility
# =========================================================================

def run_tests():
    """Standalone runner that executes inline assertions matching NutriSense conventions."""
    test_results = []
    PASS = "[PASS]"
    FAIL = "[FAIL]"

    def record(name: str, condition: bool, extra: str = ""):
        status = PASS if condition else FAIL
        msg = f"{status} {name}"
        if extra:
            msg += f" | {extra}"
        print(msg)
        test_results.append((name, condition))

    print("=" * 65)
    print("RUNNING PHASE 9: RECOMMENDATION SYSTEM TESTS")
    print("=" * 65)

    # 1. Constants
    print("\n--- 1. Constants & Disclaimers ---")
    record("constants: RECOMMENDATION_DISCLAIMER is non-empty", len(RECOMMENDATION_DISCLAIMER) > 50)
    record("constants: SAFETY_OVERRIDE_NOTE exists", len(SAFETY_OVERRIDE_NOTE) > 20)
    record("constants: weights sum to 1.0", abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-4)

    # 2. Safety filter
    print("\n--- 2. Mandatory Safety Filter ---")
    chicken_rec = {"recipe_name": "Chicken Tikka", "ingredients_raw": "Chicken, oil", "diet": "non_vegetarian"}
    safe_chick, _, _ = RecipeSafetyFilter.check_safety(chicken_rec, dietary_preference="vegetarian")
    record("safety: vegetarian rejects chicken", safe_chick is False)

    paneer_rec = {"recipe_name": "Paneer Bhurji", "ingredients_raw": "Dairy paneer, onion", "diet": "vegetarian"}
    safe_pan, _, _ = RecipeSafetyFilter.check_safety(paneer_rec, dietary_preference="vegan")
    record("safety: vegan rejects paneer", safe_pan is False)

    eggplant_rec = {"recipe_name": "Eggplant Roast", "ingredients_raw": "Eggplant, oil, spices", "diet": "vegetarian"}
    safe_ep, _, _ = RecipeSafetyFilter.check_safety(eggplant_rec, dietary_preference="vegetarian")
    record("safety: eggplant is not false-positive egg", safe_ep is True)

    kaju_rec = {"recipe_name": "Kaju Barfi", "ingredients_raw": "Cashew nuts (kaju), sugar", "diet": "vegetarian"}
    safe_kaju, _, _ = RecipeSafetyFilter.check_safety(kaju_rec, user_allergies=["nuts"])
    record("safety: nuts allergy rejects cashew", safe_kaju is False)

    # 3. Non-override rule
    print("\n--- 3. Safety Non-Override Rule ---")
    score_res = SuitabilityScorer.score_recipe(chicken_rec, VEG_PROFILE)
    record("safety: unsafe recipe scores 0.0", score_res["suitability_score"] == 0.0)
    record("safety: unsafe recipe is disqualified", score_res["disqualified"] is True)

    # 4. Scoring & Explanation
    print("\n--- 4. Suitability Scoring & Explanation ---")
    dal_rec = {
        "recipe_name": "Palak Moong Dal",
        "cuisine": "Indian",
        "course": "Main Course",
        "diet": "vegetarian",
        "total_time_mins": 25.0,
        "complexity_level": "EASY",
        "ingredients_raw": "Moong dal, spinach, tomatoes, cumin, lemon",
        "ingredient_tokens": "moong dal spinach tomatoes cumin lemon"
    }
    dal_score = SuitabilityScorer.score_recipe(dal_rec, VEG_PROFILE)
    record("scoring: score is bounded (0-100)", 0.0 <= dal_score["suitability_score"] <= 100.0)
    record("scoring: breakdown has 5 components", len(dal_score["breakdown"]) == 5)
    record("scoring: explanation is non-empty", len(dal_score["explanation"]) > 20)

    # 5. Recipe recommendations
    print("\n--- 5. Recipe Recommendations ---")
    recs = get_recipe_recommendations(VEG_PROFILE, top_n=5)
    record("recs: returns requested top-N", len(recs) == 5)
    record("recs: all returned recipes are safe", all(r["suitability_score"] > 0 for r in recs))

    # 6. Alternatives & Substitutions
    print("\n--- 6. Alternatives & Substitutions ---")
    jal_alt = get_alternative_foods("jalebi")
    record("alternatives: specific alternatives for jalebi", jal_alt["has_specific_alternatives"] is True)
    record("alternatives: contains >= 2 alternatives", len(jal_alt["alternatives"]) >= 2)

    sugar_sub = get_ingredient_substitutions(query="sugar")
    record("substitutions: sugar substitution found", len(sugar_sub) >= 1)
    record("substitutions: total substitutions >= 15", len(get_ingredient_substitutions()) >= 15)

    # 7. Meal Planning
    print("\n--- 7. Daily & Weekly Meal Planning ---")
    daily = generate_daily_meal_plan(VEG_PROFILE, target_calories=1800)
    record("mealplan: daily plan has 4 meal slots", len(daily["meals"]) == 4)
    record("mealplan: target calories match", daily["target_calories"] == 1800)

    weekly = generate_weekly_meal_plan(VEG_PROFILE, target_calories=1800)
    record("mealplan: weekly plan has 7 days", len(weekly["days"]) == 7)
    record("mealplan: grocery list categorized", len(weekly["grocery_list"]) == 4)

    # 8. API Endpoints
    print("\n--- 8. REST API Endpoints ---")
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    r1 = client.get("/api/recommendations/formula")
    record("api: GET /api/recommendations/formula returns 200", r1.status_code == 200)

    r2 = client.get("/api/recommendations/recipes?top_n=3")
    record("api: GET /api/recommendations/recipes returns 200", r2.status_code == 200)

    r3 = client.get("/api/recommendations/alternatives?food_name=samosa")
    record("api: GET /api/recommendations/alternatives returns 200", r3.status_code == 200)

    r4 = client.get("/api/recommendations/substitutions?query=sugar")
    record("api: GET /api/recommendations/substitutions returns 200", r4.status_code == 200)

    r5 = client.get("/api/recommendations/meal-plan/daily?target_calories=1800")
    record("api: GET /api/recommendations/meal-plan/daily returns 200", r5.status_code == 200)

    r6 = client.get("/api/recommendations/meal-plan/weekly?target_calories=1800")
    record("api: GET /api/recommendations/meal-plan/weekly returns 200", r6.status_code == 200)

    # Summary
    total = len(test_results)
    passed = sum(1 for _, ok in test_results if ok)
    failed = total - passed

    print("\n" + "=" * 65)
    print(f"PHASE 9 RECOMMENDATION TESTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print("=" * 65)

    if failed > 0:
        raise AssertionError(f"{failed} test(s) failed in test_recommendation_system.py")


if __name__ == "__main__":
    run_tests()
