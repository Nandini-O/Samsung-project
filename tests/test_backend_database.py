"""
tests/test_backend_database.py
==============================
Comprehensive Unit & Integration Tests for Phase 10: Backend and Database Integration

Verifies:
 1. SQLite Schema Initialization & Table Creation (All 10 relational tables)
 2. Automatic Seeding & Data Migration (Foods KB, Recipes, Allergens, Profiles)
 3. Parameterized SQL & SQL-Injection Immunity
 4. User Profile Repository (CRUD, active toggle, versioning, JSON list deserialization)
 5. Meal Logging & Validation (Portion bounds, nutrient checks, slot validation)
 6. Meal Diary History & Multi-Criteria Filtering (Date range, meal slot)
 7. Daily & Period Nutrition Totals Calculation (Calorie targets, macro progress, averages)
 8. Meal Plan & Recommendation Feedback Persistence
 9. End-to-End Module Data Flow (Vision/Search -> Analysis -> Profile -> Score -> DB Log -> Totals)
 10. Flask REST API Endpoints (Meals log/history/totals/delete, Feedback, DB stats)

Author: NutriSense AI Team
Phase: 10
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import MagicMock
import uuid

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
import sqlite3
from app import app as flask_app
import src.database as db
from src.nutrition_analyzer import calculate_nutrients
from src.recommendation_engine import SuitabilityScorer


@pytest.fixture
def temp_db(tmp_path):
    """Provides an isolated SQLite database path for testing."""
    test_db_file = tmp_path / "test_nutrisense.db"
    db.init_db(test_db_file)
    return test_db_file


@pytest.fixture
def flask_client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


# =========================================================================
# 1. Database Schema & Initialization
# =========================================================================

class TestDatabaseSchema:
    def test_init_db_creates_all_tables(self, temp_db):
        conn = db.get_connection(temp_db)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        conn.close()

        expected_tables = {
            "user_profiles", "foods", "recipes", "recipe_ingredients",
            "allergens", "food_allergens", "meal_logs", "meal_plans",
            "user_feedback", "scores_history"
        }
        for t in expected_tables:
            assert t in tables, f"Table {t} missing from SQLite schema"

    def test_init_db_idempotent(self, temp_db):
        assert db.init_db(temp_db) is True
        assert db.init_db(temp_db) is True

    def test_foreign_keys_and_wal_mode(self, temp_db):
        conn = db.get_connection(temp_db)
        fk_on = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert fk_on == 1
        assert journal.lower() == "wal"


# =========================================================================
# 2. Seeding & Data Migration
# =========================================================================

class TestDataSeeding:
    def test_seed_initial_data_populates_tables(self, temp_db):
        counts = db.seed_initial_data(temp_db)
        assert counts["foods"] >= 43
        assert counts["recipes"] >= 1000
        assert counts["allergens"] >= 8

    def test_seed_initial_data_preserves_existing_data(self, temp_db):
        counts1 = db.seed_initial_data(temp_db)
        counts2 = db.seed_initial_data(temp_db)
        # Second run should insert 0 because tables already populated
        assert counts2["foods"] == 0
        assert counts2["recipes"] == 0


# =========================================================================
# 3. Parameterized SQL & Injection Defense
# =========================================================================

class TestSQLInjectionDefense:
    def test_malicious_sql_in_profile_name(self, temp_db):
        malicious_name = "Robert'); DROP TABLE user_profiles;--"
        p = db.db_save_profile({"display_name": malicious_name}, db_path=temp_db)
        assert p["display_name"] == malicious_name

        # Verify table still exists and record is preserved
        conn = db.get_connection(temp_db)
        cur = conn.execute("SELECT COUNT(*) FROM user_profiles")
        count = cur.fetchone()[0]
        conn.close()
        assert count >= 1

    def test_quotes_in_food_name(self, temp_db):
        p = db.db_save_profile({"display_name": "Test User"}, db_path=temp_db)
        meal = db.log_meal({
            "user_id": p["user_id"],
            "food_name": "Chef's \"Special\" Curry -- ;",
            "portion_g": 200,
            "calories": 250,
            "protein_g": 10,
            "carbs_g": 20,
            "fat_g": 12,
            "meal_slot": "dinner"
        }, db_path=temp_db)
        assert meal["food_name"] == "Chef's \"Special\" Curry -- ;"


# =========================================================================
# 4. User Profile Repository (CRUD & Active Management)
# =========================================================================

class TestProfileRepository:
    def test_save_and_get_profile(self, temp_db):
        prof = {
            "display_name": "Nandini Researcher",
            "age": 28,
            "gender": "female",
            "dietary_goal": "diabetes_management",
            "allergies": ["gluten", "dairy"],
            "health_conditions": ["hypertension"],
            "daily_calorie_target": 1750.0
        }
        saved = db.db_save_profile(prof, db_path=temp_db)
        assert "user_id" in saved
        assert saved["display_name"] == "Nandini Researcher"
        assert saved["allergies"] == ["gluten", "dairy"]
        assert saved["health_conditions"] == ["hypertension"]
        assert saved["profile_version"] == 1

        fetched = db.db_get_profile(saved["user_id"], db_path=temp_db)
        assert fetched is not None
        assert fetched["user_id"] == saved["user_id"]
        assert fetched["daily_calorie_target"] == 1750.0

    def test_update_increments_version(self, temp_db):
        p1 = db.db_save_profile({"display_name": "Original Name"}, db_path=temp_db)
        assert p1["profile_version"] == 1

        p2 = db.db_save_profile({"user_id": p1["user_id"], "display_name": "Updated Name", "weight_kg": 62.5}, db_path=temp_db)
        assert p2["display_name"] == "Updated Name"
        assert p2["weight_kg"] == 62.5
        assert p2["profile_version"] == 2

    def test_empty_display_name_raises(self, temp_db):
        with pytest.raises(ValueError):
            db.db_save_profile({"display_name": ""}, db_path=temp_db)

    def test_list_and_delete_profile(self, temp_db):
        p1 = db.db_save_profile({"display_name": "User One"}, db_path=temp_db)
        p2 = db.db_save_profile({"display_name": "User Two"}, db_path=temp_db)
        profiles = db.db_list_profiles(db_path=temp_db)
        assert len(profiles) >= 2

        deleted = db.db_delete_profile(p1["user_id"], db_path=temp_db)
        assert deleted is True
        assert db.db_get_profile(p1["user_id"], db_path=temp_db) is None

    def test_active_profile_toggle(self, temp_db):
        p1 = db.db_save_profile({"display_name": "User 1"}, db_path=temp_db)
        p2 = db.db_save_profile({"display_name": "User 2"}, db_path=temp_db)

        db.db_set_active_profile(p2["user_id"], db_path=temp_db)
        active = db.db_get_active_profile(db_path=temp_db)
        assert active is not None
        assert active["user_id"] == p2["user_id"]

        db.db_set_active_profile(p1["user_id"], db_path=temp_db)
        active2 = db.db_get_active_profile(db_path=temp_db)
        assert active2["user_id"] == p1["user_id"]


# =========================================================================
# 5. Meal Logging & Validation
# =========================================================================

class TestMealLogging:
    def test_log_valid_meal(self, temp_db):
        p = db.db_save_profile({"display_name": "Meal Logger"}, db_path=temp_db)
        meal_entry = {
            "user_id": p["user_id"],
            "food_name": "Dal Tadka with Rice",
            "portion_g": 300,
            "calories": 420.0,
            "protein_g": 14.5,
            "carbs_g": 65.0,
            "fat_g": 11.0,
            "fibre_g": 6.5,
            "sodium_mg": 450.0,
            "meal_slot": "lunch",
            "suitability_score": 88.0,
            "notes": "Home cooked with less oil"
        }
        res = db.log_meal(meal_entry, db_path=temp_db)
        assert res["log_id"] is not None
        assert res["food_name"] == "Dal Tadka with Rice"
        assert res["calories"] == 420.0
        assert res["meal_slot"] == "lunch"

    def test_missing_user_id_raises(self, temp_db):
        with pytest.raises(ValueError, match="user_id"):
            db.log_meal({"food_name": "Apple", "portion_g": 100, "calories": 50, "protein_g": 0, "carbs_g": 10, "fat_g": 0}, db_path=temp_db)

    def test_missing_food_name_raises(self, temp_db):
        with pytest.raises(ValueError, match="food_name"):
            db.log_meal({"user_id": "u1", "food_name": "", "portion_g": 100, "calories": 50, "protein_g": 0, "carbs_g": 10, "fat_g": 0}, db_path=temp_db)

    def test_negative_portion_raises(self, temp_db):
        with pytest.raises(ValueError, match="portion_g"):
            db.log_meal({"user_id": "u1", "food_name": "Dal", "portion_g": -50, "calories": 50, "protein_g": 0, "carbs_g": 10, "fat_g": 0}, db_path=temp_db)

    def test_negative_calories_raises(self, temp_db):
        with pytest.raises(ValueError, match="cannot be negative"):
            db.log_meal({"user_id": "u1", "food_name": "Dal", "portion_g": 100, "calories": -50, "protein_g": 0, "carbs_g": 10, "fat_g": 0}, db_path=temp_db)


# =========================================================================
# 6. Meal Diary History & Filtering
# =========================================================================

class TestMealHistoryAndFiltering:
    def test_filter_by_slot_and_date(self, temp_db):
        p = db.db_save_profile({"display_name": "History User"}, db_path=temp_db)
        uid = p["user_id"]

        db.log_meal({"user_id": uid, "food_name": "Poha", "portion_g": 150, "calories": 200, "protein_g": 4, "carbs_g": 35, "fat_g": 5, "meal_slot": "breakfast", "meal_date": "2026-11-01"}, db_path=temp_db)
        db.log_meal({"user_id": uid, "food_name": "Rajma Chawal", "portion_g": 350, "calories": 480, "protein_g": 18, "carbs_g": 75, "fat_g": 8, "meal_slot": "lunch", "meal_date": "2026-11-01"}, db_path=temp_db)
        db.log_meal({"user_id": uid, "food_name": "Khichdi", "portion_g": 250, "calories": 300, "protein_g": 10, "carbs_g": 45, "fat_g": 6, "meal_slot": "dinner", "meal_date": "2026-11-02"}, db_path=temp_db)

        # Slot filter
        b_meals = db.get_meal_history(uid, meal_slot="breakfast", db_path=temp_db)
        assert len(b_meals) == 1
        assert b_meals[0]["food_name"] == "Poha"

        # Date filter
        d1_meals = db.get_meal_history(uid, start_date="2026-11-01", end_date="2026-11-01", db_path=temp_db)
        assert len(d1_meals) == 2

        # Delete meal log
        deleted = db.delete_meal_log(b_meals[0]["log_id"], user_id=uid, db_path=temp_db)
        assert deleted is True
        assert len(db.get_meal_history(uid, meal_slot="breakfast", db_path=temp_db)) == 0


# =========================================================================
# 7. Daily & Period Nutrition Totals Calculation
# =========================================================================

class TestNutritionTotals:
    def test_daily_totals_aggregation(self, temp_db):
        p = db.db_save_profile({"display_name": "Totals User", "daily_calorie_target": 1800.0}, db_path=temp_db)
        uid = p["user_id"]
        today = "2026-11-05"

        db.log_meal({"user_id": uid, "food_name": "Breakfast Oats", "portion_g": 200, "calories": 250, "protein_g": 10, "carbs_g": 40, "fat_g": 5, "fibre_g": 6, "sodium_mg": 100, "meal_slot": "breakfast", "meal_date": today, "suitability_score": 90.0}, db_path=temp_db)
        db.log_meal({"user_id": uid, "food_name": "Lunch Dal Roti", "portion_g": 350, "calories": 450, "protein_g": 16, "carbs_g": 65, "fat_g": 12, "fibre_g": 8, "sodium_mg": 400, "meal_slot": "lunch", "meal_date": today, "suitability_score": 85.0}, db_path=temp_db)

        totals = db.get_meal_totals(uid, date_str=today, db_path=temp_db)
        assert totals["meal_count"] == 2
        assert totals["totals"]["calories"] == 700.0
        assert totals["totals"]["protein_g"] == 26.0
        assert totals["totals"]["carbs_g"] == 105.0
        assert totals["totals"]["fat_g"] == 17.0
        assert totals["totals"]["fibre_g"] == 14.0
        assert totals["totals"]["sodium_mg"] == 500.0
        assert totals["totals"]["avg_suitability_score"] == 87.5
        assert totals["target"]["daily_calorie_target"] == 1800.0
        assert totals["target"]["calorie_progress_pct"] == round((700.0 / 1800.0) * 100.0, 1)
        assert totals["target"]["calories_remaining"] == 1100.0


# =========================================================================
# 8. Meal Plans & Feedback Persistence
# =========================================================================

class TestMealPlansAndFeedback:
    def test_save_and_retrieve_meal_plan(self, temp_db):
        p = db.db_save_profile({"display_name": "Plan User"}, db_path=temp_db)
        payload = {"day": "Monday", "target_calories": 1800, "meals": [{"slot": "lunch", "recipe": "Khichdi"}]}
        plan_id = db.save_meal_plan_db(p["user_id"], "daily", payload, db_path=temp_db)
        assert plan_id is not None

        retrieved = db.get_latest_meal_plan_db(p["user_id"], "daily", db_path=temp_db)
        assert retrieved is not None
        assert retrieved["plan_payload"]["day"] == "Monday"

    def test_record_and_list_feedback(self, temp_db):
        p = db.db_save_profile({"display_name": "Feedback User"}, db_path=temp_db)
        res = db.record_feedback_db({
            "user_id": p["user_id"],
            "feedback_type": "prediction_accuracy",
            "item_reference": "idli",
            "rating": 5,
            "comment": "Accurately recognized idli portion and nutrients!"
        }, db_path=temp_db)
        assert res["status"] == "recorded"

        f_list = db.get_feedback_list_db(limit=10, db_path=temp_db)
        assert len(f_list) >= 1
        assert f_list[0]["item_reference"] == "idli"


# =========================================================================
# 9. End-to-End Data Flow Integration
# =========================================================================

class TestEndToEndDataFlow:
    def test_full_pipeline_to_database_log(self, temp_db):
        """
        Tests the complete data flow:
        User Profile -> Nutrition Calculation -> Suitability Scoring -> DB Meal Logging -> Totals
        """
        # 1. Create Profile
        user = db.db_save_profile({
            "display_name": "End To End User",
            "dietary_goal": "muscle_gain",
            "dietary_preference": "vegetarian",
            "daily_calorie_target": 2200.0
        }, db_path=temp_db)

        # 2. Nutrition Analysis calculation
        food_nutrients = calculate_nutrients("samosa", 130.0)
        cals = food_nutrients["nutrients"]["calories_kcal"]["value"]
        assert cals > 0

        # 3. Suitability Scoring
        score_eval = SuitabilityScorer.score_recipe({
            "recipe_name": "Samosa",
            "diet": "vegetarian",
            "total_time_mins": 30.0,
            "ingredients_raw": "Potato, spices, flour"
        }, user)

        # 4. Log Meal to Database
        log_res = db.log_meal({
            "user_id": user["user_id"],
            "food_name": "Samosa",
            "portion_g": 130.0,
            "calories": cals,
            "protein_g": food_nutrients["nutrients"]["protein_g"]["value"],
            "carbs_g": food_nutrients["nutrients"]["carbs_g"]["value"],
            "fat_g": food_nutrients["nutrients"]["fat_g"]["value"],
            "fibre_g": food_nutrients["nutrients"]["fibre_g"]["value"],
            "sodium_mg": food_nutrients["nutrients"]["sodium_mg"]["value"],
            "suitability_score": score_eval["suitability_score"],
            "source_type": "nutrition_analysis_pipeline",
            "meal_slot": "snack"
        }, db_path=temp_db)
        assert log_res["calories"] == cals

        # 5. Check daily totals
        totals = db.get_meal_totals(user["user_id"], db_path=temp_db)
        assert totals["meal_count"] == 1
        assert totals["totals"]["calories"] == cals


# =========================================================================
# 10. Flask REST API Endpoints
# =========================================================================

class TestBackendAPI:
    def test_post_log_meal_201(self, flask_client):
        payload = {
            "food_name": "Masala Dosa",
            "portion_g": 200,
            "calories": 300,
            "protein_g": 6,
            "carbs_g": 45,
            "fat_g": 8,
            "meal_slot": "breakfast"
        }
        resp = flask_client.post("/api/meals/log", json=payload, content_type="application/json")
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert "log_id" in data["log"]

    def test_post_log_meal_missing_fields_400(self, flask_client):
        resp = flask_client.post("/api/meals/log", json={"portion_g": 200}, content_type="application/json")
        assert resp.status_code == 400

    def test_get_meals_history_200(self, flask_client):
        resp = flask_client.get("/api/meals/history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "meals" in data

    def test_get_meals_totals_200(self, flask_client):
        resp = flask_client.get("/api/meals/totals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "totals" in data["data"]

    def test_delete_meal_log_endpoint(self, flask_client):
        # Create a log first
        create_resp = flask_client.post("/api/meals/log", json={
            "food_name": "Idli", "portion_g": 100, "calories": 130, "protein_g": 4, "carbs_g": 26, "fat_g": 1, "meal_slot": "breakfast"
        }, content_type="application/json")
        log_id = create_resp.get_json()["log"]["log_id"]

        del_resp = flask_client.delete(f"/api/meals/{log_id}")
        assert del_resp.status_code == 200

        del_resp_again = flask_client.delete(f"/api/meals/{log_id}")
        assert del_resp_again.status_code == 404

    def test_post_feedback_201(self, flask_client):
        resp = flask_client.post("/api/feedback", json={
            "feedback_type": "recommendation_quality",
            "item_reference": "Khichdi",
            "rating": 5,
            "comment": "Super healthy suggestion!"
        }, content_type="application/json")
        assert resp.status_code == 201

    def test_get_feedback_200(self, flask_client):
        resp = flask_client.get("/api/feedback")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "feedback" in data

    def test_get_db_stats_200(self, flask_client):
        resp = flask_client.get("/api/db/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["database"]["exists"] is True
        assert data["database"]["row_counts"]["foods"] >= 43


# =========================================================================
# Standalone run_tests() for Master Runner compatibility
# =========================================================================

def run_tests():
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
    print("RUNNING PHASE 10: BACKEND & DATABASE INTEGRATION TESTS")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_db = Path(tmp_dir) / "nutrisense_test.db"

        # 1. Schema
        print("\n--- 1. Schema & Initial Seeding ---")
        init_ok = db.init_db(test_db)
        record("db: init_db creates tables", init_ok is True)

        counts = db.seed_initial_data(test_db)
        record("db: foods seeded >= 43", counts["foods"] >= 43, f"count={counts['foods']}")
        record("db: recipes seeded >= 1000", counts["recipes"] >= 1000, f"count={counts['recipes']}")
        record("db: allergens seeded >= 8", counts["allergens"] >= 8)

        # 2. Profiles
        print("\n--- 2. Profile Repository & Versioning ---")
        p = db.db_save_profile({"display_name": "Standalone User", "daily_calorie_target": 1900.0}, db_path=test_db)
        record("db: save profile returns user_id", bool(p.get("user_id")))
        record("db: initial profile_version is 1", p.get("profile_version") == 1)

        p_up = db.db_save_profile({"user_id": p["user_id"], "display_name": "Updated Standalone User"}, db_path=test_db)
        record("db: update increments profile_version to 2", p_up.get("profile_version") == 2)

        # 3. SQL Injection Defense
        print("\n--- 3. SQL Injection Defense ---")
        mal = db.db_save_profile({"display_name": "Attacker'); DROP TABLE meal_logs;--"}, db_path=test_db)
        record("db: parameterized SQL immune to injection", "DROP TABLE" in mal["display_name"])

        # 4. Meal Diary & Totals
        print("\n--- 4. Meal Diary & Totals Calculation ---")
        m1 = db.log_meal({
            "user_id": p["user_id"],
            "food_name": "Masala Dosa",
            "portion_g": 200,
            "calories": 300,
            "protein_g": 6,
            "carbs_g": 45,
            "fat_g": 8,
            "meal_slot": "breakfast",
            "meal_date": "2026-11-05"
        }, db_path=test_db)
        record("db: log meal returns log_id", bool(m1.get("log_id")))

        m2 = db.log_meal({
            "user_id": p["user_id"],
            "food_name": "Curd Rice",
            "portion_g": 250,
            "calories": 280,
            "protein_g": 7,
            "carbs_g": 42,
            "fat_g": 7,
            "meal_slot": "lunch",
            "meal_date": "2026-11-05"
        }, db_path=test_db)
        record("db: log second meal", bool(m2.get("log_id")))

        totals = db.get_meal_totals(p["user_id"], date_str="2026-11-05", db_path=test_db)
        record("db: meal count is 2", totals["meal_count"] == 2)
        record("db: total calories sum is 580.0", totals["totals"]["calories"] == 580.0)
        record("db: calorie progress pct calculated", totals["target"]["calorie_progress_pct"] > 0)

        # 5. API Endpoints
        print("\n--- 5. REST API Endpoints ---")
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()

        r_log = client.post("/api/meals/log", json={
            "food_name": "Vegetable Biryani",
            "portion_g": 300,
            "calories": 400,
            "protein_g": 8,
            "carbs_g": 60,
            "fat_g": 10,
            "meal_slot": "dinner"
        }, content_type="application/json")
        record("api: POST /api/meals/log returns 201", r_log.status_code == 201)
        api_log_id = r_log.get_json()["log"]["log_id"]

        r_hist = client.get("/api/meals/history")
        record("api: GET /api/meals/history returns 200", r_hist.status_code == 200)

        r_tot = client.get("/api/meals/totals")
        record("api: GET /api/meals/totals returns 200", r_tot.status_code == 200)

        r_del = client.delete(f"/api/meals/{api_log_id}")
        record("api: DELETE /api/meals/<id> returns 200", r_del.status_code == 200)

        r_stats = client.get("/api/db/stats")
        record("api: GET /api/db/stats returns 200", r_stats.status_code == 200)

        # Summary
        total = len(test_results)
        passed = sum(1 for _, ok in test_results if ok)
        failed = total - passed

        print("\n" + "=" * 65)
        print(f"PHASE 10 DATABASE TESTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
        print("=" * 65)

        if failed > 0:
            raise AssertionError(f"{failed} test(s) failed in test_backend_database.py")


if __name__ == "__main__":
    run_tests()
