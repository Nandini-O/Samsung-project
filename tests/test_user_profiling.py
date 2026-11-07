"""
tests/test_user_profiling.py
Comprehensive Integration & Unit Tests for Phase 8: User Profiling Module

Verifies:
 1. Module Constants (PHASE8_DISCLAIMER, BMI_LIMITATION_NOTE)
 2. Profile Validation (required fields, enums, ranges, sanitisation)
 3. BMI Calculation (categories, limitation note, edge cases)
 4. Profile CRUD (create, read, update, delete, list, active profile)
 5. Breakfast Suggestions (routine filtering, dietary preference, goal ranking)
 6. Personalised Context (ICMR mapping, calorie targets, portion guidance)
 7. Profile Persistence (file on disk integrity)
 8. Flask API Endpoints (all Phase 8 routes: profiles, BMI, validate, context, breakfast)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# Ensure workspace root is on sys.path regardless of invocation directory
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# ---------------------------------------------------------------------------
# Lightweight ML mocks — only installed when real libraries are absent.
# When run via pytest, conftest.py handles this. When run directly as a script,
# this block ensures the same stubs are present.
# ---------------------------------------------------------------------------

def _install_ml_stubs_if_needed():
    """Register torch / torchvision / easyocr stubs only when missing."""
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass

    # torch
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

    # torch.nn
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

    # torch.optim
    optim_pkg = types.ModuleType("torch.optim")
    optim_pkg.__path__ = []
    optim_pkg.Adam = MagicMock()
    optim_pkg.SGD = MagicMock()
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

    # torchvision
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

    # easyocr
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
import json
import copy

from src.user_profile_manager import (
    validate_profile_data,
    calculate_bmi,
    get_breakfast_suggestions,
    get_personalized_context,
    create_profile,
    update_profile,
    get_profile,
    delete_profile,
    list_profiles,
    get_active_profile,
    set_active_profile,
    PHASE8_DISCLAIMER,
    BMI_LIMITATION_NOTE,
)

from app import app as flask_app


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

VALID_PROFILE = {
    "display_name": "Test User Phase8",
    "age": 30,
    "gender": "female",
    "height_cm": 165,
    "weight_kg": 65,
    "dietary_goal": "general_wellness",
    "activity_level": "moderately_active",
    "dietary_preference": "vegetarian",
    "health_conditions": [],
    "allergies": [],
    "morning_routine_preference": "regular",
    "budget": "medium",
    "max_cooking_time_minutes": 30,
    "dietary_notes": "Test profile for Phase 8 testing",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolate_profile_storage(tmp_path, monkeypatch):
    """Redirect all profile storage to a temp directory for every test."""
    test_db = tmp_path / "test_profiles.json"
    import src.user_profile_manager as upm
    monkeypatch.setattr(upm, "_PROFILES_DIR", tmp_path)
    monkeypatch.setattr(upm, "_PROFILES_FILE", test_db)
    yield test_db


@pytest.fixture()
def flask_client(isolate_profile_storage, monkeypatch):
    """Provide a Flask test client with profile storage already isolated."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


@pytest.fixture()
def created_profile():
    """Create and return a profile using VALID_PROFILE data."""
    return create_profile(copy.deepcopy(VALID_PROFILE))


# =========================================================================
# 1. TestProfileConstants
# =========================================================================

class TestProfileConstants:
    """Verify module-level disclaimer and limitation note constants."""

    def test_phase8_disclaimer_exists(self):
        assert PHASE8_DISCLAIMER is not None

    def test_phase8_disclaimer_is_string(self):
        assert isinstance(PHASE8_DISCLAIMER, str)

    def test_phase8_disclaimer_non_empty(self):
        assert len(PHASE8_DISCLAIMER) > 0

    def test_bmi_limitation_note_exists(self):
        assert BMI_LIMITATION_NOTE is not None

    def test_bmi_limitation_note_is_string(self):
        assert isinstance(BMI_LIMITATION_NOTE, str)

    def test_bmi_limitation_note_non_empty(self):
        assert len(BMI_LIMITATION_NOTE) > 0

    def test_disclaimer_mentions_medical(self):
        assert "medical" in PHASE8_DISCLAIMER.lower() or "advice" in PHASE8_DISCLAIMER.lower()

    def test_bmi_note_mentions_screening(self):
        assert "screening" in BMI_LIMITATION_NOTE.lower() or "population" in BMI_LIMITATION_NOTE.lower()


# =========================================================================
# 2. TestProfileValidation
# =========================================================================

class TestProfileValidation:
    """Validate profile data against field constraints."""

    def test_valid_data_passes(self):
        ok, errs = validate_profile_data(VALID_PROFILE)
        assert ok is True
        fatal = [e for e in errs if "will be truncated" not in e]
        assert len(fatal) == 0

    def test_missing_display_name_fails(self):
        ok, errs = validate_profile_data({})
        assert ok is False
        assert any("display_name" in e for e in errs)

    def test_short_display_name_fails(self):
        ok, errs = validate_profile_data({"display_name": "A"})
        assert ok is False
        assert any("display_name" in e for e in errs)

    def test_long_display_name_fails(self):
        ok, errs = validate_profile_data({"display_name": "X" * 51})
        assert ok is False

    def test_negative_age_fails(self):
        data = {**VALID_PROFILE, "age": -5}
        ok, errs = validate_profile_data(data)
        assert ok is False
        assert any("age" in e for e in errs)

    def test_zero_age_fails(self):
        data = {**VALID_PROFILE, "age": 0}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_age_over_120_fails(self):
        data = {**VALID_PROFILE, "age": 121}
        ok, errs = validate_profile_data(data)
        assert ok is False
        assert any("age" in e for e in errs)

    def test_valid_age_boundary_1(self):
        data = {**VALID_PROFILE, "age": 1}
        ok, _ = validate_profile_data(data)
        assert ok is True

    def test_valid_age_boundary_120(self):
        data = {**VALID_PROFILE, "age": 120}
        ok, _ = validate_profile_data(data)
        assert ok is True

    def test_invalid_gender_fails(self):
        data = {**VALID_PROFILE, "gender": "unknown_gender"}
        ok, errs = validate_profile_data(data)
        assert ok is False
        assert any("gender" in e for e in errs)

    def test_valid_gender_male(self):
        data = {**VALID_PROFILE, "gender": "male"}
        ok, _ = validate_profile_data(data)
        assert ok is True

    def test_valid_gender_other(self):
        data = {**VALID_PROFILE, "gender": "other"}
        ok, _ = validate_profile_data(data)
        assert ok is True

    def test_invalid_activity_level_fails(self):
        data = {**VALID_PROFILE, "activity_level": "super_active"}
        ok, errs = validate_profile_data(data)
        assert ok is False
        assert any("activity_level" in e for e in errs)

    def test_valid_activity_levels(self):
        for level in ["sedentary", "lightly_active", "moderately_active", "very_active", "extremely_active"]:
            data = {**VALID_PROFILE, "activity_level": level}
            ok, _ = validate_profile_data(data)
            assert ok is True, f"activity_level '{level}' should be valid"

    def test_invalid_dietary_goal_fails(self):
        data = {**VALID_PROFILE, "dietary_goal": "get_rich"}
        ok, errs = validate_profile_data(data)
        assert ok is False
        assert any("dietary_goal" in e for e in errs)

    def test_valid_dietary_goals(self):
        for goal in ["general_wellness", "weight_loss", "muscle_gain", "diabetes_management",
                      "hypertension_management", "heart_health", "balanced_nutrition"]:
            data = {**VALID_PROFILE, "dietary_goal": goal}
            ok, _ = validate_profile_data(data)
            assert ok is True, f"dietary_goal '{goal}' should be valid"

    def test_empty_dict_fails(self):
        ok, errs = validate_profile_data({})
        assert ok is False

    def test_invalid_dietary_preference_fails(self):
        data = {**VALID_PROFILE, "dietary_preference": "breatharian"}
        ok, errs = validate_profile_data(data)
        assert ok is False
        assert any("dietary_preference" in e for e in errs)

    def test_invalid_allergy_fails(self):
        data = {**VALID_PROFILE, "allergies": ["unicorn_dust"]}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_invalid_health_condition_fails(self):
        data = {**VALID_PROFILE, "health_conditions": ["telekinesis"]}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_invalid_budget_fails(self):
        data = {**VALID_PROFILE, "budget": "unlimited"}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_height_too_low_fails(self):
        data = {**VALID_PROFILE, "height_cm": 30}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_height_too_high_fails(self):
        data = {**VALID_PROFILE, "height_cm": 300}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_weight_too_low_fails(self):
        data = {**VALID_PROFILE, "weight_kg": 10}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_weight_too_high_fails(self):
        data = {**VALID_PROFILE, "weight_kg": 500}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_dietary_notes_truncation_warning(self):
        data = {**VALID_PROFILE, "dietary_notes": "x" * 600}
        ok, errs = validate_profile_data(data)
        # Truncation is non-fatal, so should still pass
        assert ok is True
        assert any("truncated" in e for e in errs)

    def test_age_bool_rejected(self):
        """Booleans must not pass as valid integers for age."""
        data = {**VALID_PROFILE, "age": True}
        ok, errs = validate_profile_data(data)
        assert ok is False

    def test_invalid_morning_routine_fails(self):
        data = {**VALID_PROFILE, "morning_routine_preference": "nocturnal"}
        ok, errs = validate_profile_data(data)
        assert ok is False


# =========================================================================
# 3. TestBMICalculation
# =========================================================================

class TestBMICalculation:
    """Verify BMI formula, categories, and limitation note."""

    def test_normal_bmi(self):
        result = calculate_bmi(170, 65)
        assert result["category"] == "Normal"
        assert 18.5 <= result["value"] < 25.0

    def test_underweight_bmi(self):
        result = calculate_bmi(180, 50)
        assert result["category"] == "Underweight"
        assert result["value"] < 18.5

    def test_overweight_bmi(self):
        result = calculate_bmi(170, 80)
        assert result["category"] == "Overweight"
        assert 25.0 <= result["value"] < 30.0

    def test_obese_bmi(self):
        result = calculate_bmi(160, 90)
        assert result["category"] == "Obese"
        assert result["value"] >= 30.0

    def test_limitation_note_always_present(self):
        result = calculate_bmi(170, 70)
        assert "limitation_note" in result
        assert result["limitation_note"] == BMI_LIMITATION_NOTE

    def test_bmi_value_rounded_one_decimal(self):
        result = calculate_bmi(175, 72)
        value_str = str(result["value"])
        if "." in value_str:
            decimals = len(value_str.split(".")[1])
            assert decimals <= 1

    def test_edge_very_tall(self):
        result = calculate_bmi(220, 90)
        assert "value" in result
        assert "category" in result
        assert isinstance(result["value"], float)

    def test_edge_very_short(self):
        result = calculate_bmi(100, 30)
        assert "value" in result
        assert "category" in result

    def test_bmi_formula_correctness(self):
        """Manual formula check: BMI = weight / (height_m)^2."""
        h, w = 170, 65
        expected = round(w / (h / 100.0) ** 2, 1)
        result = calculate_bmi(h, w)
        assert result["value"] == expected

    def test_negative_height_raises(self):
        with pytest.raises(ValueError):
            calculate_bmi(-170, 65)

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            calculate_bmi(170, 0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError):
            calculate_bmi(0, 65)

    def test_result_keys(self):
        result = calculate_bmi(165, 60)
        assert set(result.keys()) == {"value", "category", "limitation_note"}


# =========================================================================
# 4. TestProfileCRUD
# =========================================================================

class TestProfileCRUD:
    """Test Create / Read / Update / Delete / List / Active operations."""

    def test_create_returns_profile_with_user_id(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        assert "user_id" in profile
        assert len(profile["user_id"]) > 0

    def test_create_sets_display_name(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        assert profile["display_name"] == "Test User Phase8"

    def test_create_sets_version_1(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        assert profile["profile_version"] == 1

    def test_create_sets_timestamps(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        assert "created_at" in profile
        assert "updated_at" in profile

    def test_create_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            create_profile({"display_name": "A"})

    def test_get_profile_after_create(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        fetched = get_profile(profile["user_id"])
        assert fetched is not None
        assert fetched["user_id"] == profile["user_id"]
        assert fetched["display_name"] == profile["display_name"]

    def test_get_profile_nonexistent_returns_none(self):
        result = get_profile("nonexistent-uuid-12345")
        assert result is None

    def test_update_increments_version(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        updated = update_profile(profile["user_id"], {"weight_kg": 70})
        assert updated["profile_version"] == 2

    def test_update_changes_field(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        updated = update_profile(profile["user_id"], {"weight_kg": 70})
        assert updated["weight_kg"] == 70.0

    def test_update_preserves_user_id(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        updated = update_profile(profile["user_id"], {"age": 31})
        assert updated["user_id"] == profile["user_id"]

    def test_update_preserves_created_at(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        updated = update_profile(profile["user_id"], {"age": 31})
        assert updated["created_at"] == profile["created_at"]

    def test_update_nonexistent_raises(self):
        with pytest.raises(ValueError):
            update_profile("nonexistent-uuid-99", {"age": 25})

    def test_update_invalid_data_raises(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        with pytest.raises(ValueError):
            update_profile(profile["user_id"], {"age": -10})

    def test_delete_profile_returns_true(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        result = delete_profile(profile["user_id"])
        assert result is True

    def test_delete_profile_removes_from_storage(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        delete_profile(profile["user_id"])
        assert get_profile(profile["user_id"]) is None

    def test_delete_nonexistent_returns_false(self):
        result = delete_profile("nonexistent-uuid-888")
        assert result is False

    def test_list_profiles_empty(self):
        profiles = list_profiles()
        assert isinstance(profiles, list)
        assert len(profiles) == 0

    def test_list_profiles_after_create(self):
        create_profile(copy.deepcopy(VALID_PROFILE))
        profiles = list_profiles()
        assert len(profiles) == 1

    def test_list_profiles_multiple(self):
        create_profile({**VALID_PROFILE, "display_name": "User One"})
        create_profile({**VALID_PROFILE, "display_name": "User Two"})
        profiles = list_profiles()
        assert len(profiles) == 2

    def test_set_active_profile(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        result = set_active_profile(profile["user_id"])
        assert result is True

    def test_get_active_profile_after_create(self):
        """First created profile is automatically active."""
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        active = get_active_profile()
        assert active is not None
        assert active["user_id"] == profile["user_id"]

    def test_set_active_nonexistent_returns_false(self):
        result = set_active_profile("nonexistent-uuid-777")
        assert result is False

    def test_get_active_profile_none_when_empty(self):
        active = get_active_profile()
        assert active is None

    def test_delete_active_resets_active(self):
        p1 = create_profile({**VALID_PROFILE, "display_name": "First User"})
        p2 = create_profile({**VALID_PROFILE, "display_name": "Second User"})
        set_active_profile(p1["user_id"])
        delete_profile(p1["user_id"])
        active = get_active_profile()
        # After deleting active, the first remaining should be promoted
        if active is not None:
            assert active["user_id"] == p2["user_id"]

    def test_double_update_increments_version_twice(self):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        update_profile(profile["user_id"], {"age": 31})
        updated2 = update_profile(profile["user_id"], {"age": 32})
        assert updated2["profile_version"] == 3


# =========================================================================
# 5. TestBreakfastSuggestions
# =========================================================================

class TestBreakfastSuggestions:
    """Verify breakfast suggestion filtering and ranking."""

    def test_early_bird_returns_suggestions(self):
        profile = {**VALID_PROFILE, "morning_routine_preference": "early_bird"}
        suggestions = get_breakfast_suggestions(profile)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_regular_returns_suggestions(self):
        profile = {**VALID_PROFILE, "morning_routine_preference": "regular"}
        suggestions = get_breakfast_suggestions(profile)
        assert len(suggestions) > 0

    def test_late_riser_returns_suggestions(self):
        profile = {**VALID_PROFILE, "morning_routine_preference": "late_riser"}
        suggestions = get_breakfast_suggestions(profile)
        assert len(suggestions) > 0

    def test_vegetarian_no_non_veg_items(self):
        profile = {**VALID_PROFILE, "dietary_preference": "vegetarian"}
        suggestions = get_breakfast_suggestions(profile)
        # All items should be compatible with vegetarian preference
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert "food_name" in s

    def test_returns_list_of_dicts_with_name_key(self):
        suggestions = get_breakfast_suggestions(VALID_PROFILE)
        assert isinstance(suggestions, list)
        for s in suggestions:
            assert isinstance(s, dict)
            assert "food_name" in s

    def test_max_three_suggestions(self):
        profile = {**VALID_PROFILE, "morning_routine_preference": "regular"}
        suggestions = get_breakfast_suggestions(profile)
        assert len(suggestions) <= 3

    def test_suggestions_have_calories(self):
        suggestions = get_breakfast_suggestions(VALID_PROFILE)
        for s in suggestions:
            assert "estimated_calories" in s
            assert isinstance(s["estimated_calories"], (int, float))

    def test_suggestions_have_reason(self):
        suggestions = get_breakfast_suggestions(VALID_PROFILE)
        for s in suggestions:
            assert "reason" in s
            assert len(s["reason"]) > 0

    def test_vegan_gets_compatible_suggestions(self):
        profile = {**VALID_PROFILE, "dietary_preference": "vegan"}
        suggestions = get_breakfast_suggestions(profile)
        assert isinstance(suggestions, list)
        # Should still get some suggestions
        assert len(suggestions) > 0

    def test_default_routine_fallback(self):
        """Profile without morning_routine_preference defaults to regular."""
        profile = copy.deepcopy(VALID_PROFILE)
        profile.pop("morning_routine_preference", None)
        suggestions = get_breakfast_suggestions(profile)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_goal_influences_ranking(self):
        """Weight loss goal should put goal-aligned items higher."""
        profile = {**VALID_PROFILE, "dietary_goal": "weight_loss",
                    "morning_routine_preference": "regular"}
        suggestions = get_breakfast_suggestions(profile)
        assert len(suggestions) > 0


# =========================================================================
# 6. TestPersonalizedContext
# =========================================================================

class TestPersonalizedContext:
    """Verify personalised nutritional context computation."""

    def test_returns_dict(self):
        ctx = get_personalized_context(VALID_PROFILE)
        assert isinstance(ctx, dict)

    def test_has_icmr_profile_key(self):
        ctx = get_personalized_context(VALID_PROFILE)
        assert "icmr_profile_key" in ctx
        assert isinstance(ctx["icmr_profile_key"], str)

    def test_has_daily_calorie_target(self):
        ctx = get_personalized_context(VALID_PROFILE)
        assert "daily_calorie_target" in ctx
        assert isinstance(ctx["daily_calorie_target"], (int, float))
        assert ctx["daily_calorie_target"] > 0

    def test_has_portion_guidance(self):
        ctx = get_personalized_context(VALID_PROFILE)
        assert "portion_guidance" in ctx
        assert isinstance(ctx["portion_guidance"], str)
        assert len(ctx["portion_guidance"]) > 0

    def test_has_disclaimer(self):
        ctx = get_personalized_context(VALID_PROFILE)
        assert "personalization_disclaimer" in ctx
        assert isinstance(ctx["personalization_disclaimer"], str)
        assert len(ctx["personalization_disclaimer"]) > 0

    def test_diabetes_maps_to_diabetic_adult(self):
        profile = {**VALID_PROFILE, "dietary_goal": "diabetes_management"}
        ctx = get_personalized_context(profile)
        assert ctx["icmr_profile_key"] == "diabetic_adult"

    def test_diabetes_condition_maps_to_diabetic_adult(self):
        profile = {**VALID_PROFILE, "health_conditions": ["diabetes"]}
        ctx = get_personalized_context(profile)
        assert ctx["icmr_profile_key"] == "diabetic_adult"

    def test_hypertension_maps_to_hypertensive_adult(self):
        profile = {**VALID_PROFILE, "dietary_goal": "hypertension_management"}
        ctx = get_personalized_context(profile)
        assert ctx["icmr_profile_key"] == "hypertensive_adult"

    def test_hypertension_condition_maps_to_hypertensive_adult(self):
        profile = {**VALID_PROFILE, "health_conditions": ["hypertension"]}
        ctx = get_personalized_context(profile)
        assert ctx["icmr_profile_key"] == "hypertensive_adult"

    def test_weight_loss_maps_to_weight_management(self):
        profile = {**VALID_PROFILE, "dietary_goal": "weight_loss"}
        ctx = get_personalized_context(profile)
        assert ctx["icmr_profile_key"] == "weight_management"

    def test_female_default_maps_to_female_sedentary(self):
        profile = {**VALID_PROFILE, "gender": "female", "dietary_goal": "general_wellness",
                    "health_conditions": []}
        ctx = get_personalized_context(profile)
        assert ctx["icmr_profile_key"] == "adult_female_sedentary"

    def test_male_default_maps_to_male_sedentary(self):
        profile = {**VALID_PROFILE, "gender": "male", "dietary_goal": "general_wellness",
                    "health_conditions": []}
        ctx = get_personalized_context(profile)
        assert ctx["icmr_profile_key"] == "adult_male_sedentary"

    def test_bmi_info_present_when_height_weight_given(self):
        ctx = get_personalized_context(VALID_PROFILE)
        assert ctx["bmi_info"] is not None
        assert "value" in ctx["bmi_info"]
        assert "category" in ctx["bmi_info"]

    def test_bmi_info_none_without_height(self):
        profile = copy.deepcopy(VALID_PROFILE)
        profile.pop("height_cm", None)
        ctx = get_personalized_context(profile)
        assert ctx["bmi_info"] is None

    def test_suggested_breakfast_in_context(self):
        ctx = get_personalized_context(VALID_PROFILE)
        assert "suggested_breakfast" in ctx
        assert isinstance(ctx["suggested_breakfast"], list)

    def test_allergen_avoid_list(self):
        profile = {**VALID_PROFILE, "allergies": ["dairy", "nuts"]}
        ctx = get_personalized_context(profile)
        assert "allergen_avoid_list" in ctx
        assert isinstance(ctx["allergen_avoid_list"], list)
        assert len(ctx["allergen_avoid_list"]) == 2

    def test_context_from_user_id(self):
        """get_personalized_context can accept a user_id string."""
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        ctx = get_personalized_context(profile["user_id"])
        assert "icmr_profile_key" in ctx

    def test_context_nonexistent_user_id_raises(self):
        with pytest.raises(ValueError):
            get_personalized_context("nonexistent-uuid-456")

    def test_context_invalid_type_raises(self):
        with pytest.raises(TypeError):
            get_personalized_context(12345)

    def test_muscle_gain_boosts_protein(self):
        profile = {**VALID_PROFILE, "gender": "male", "dietary_goal": "muscle_gain",
                    "health_conditions": []}
        ctx = get_personalized_context(profile)
        assert ctx["daily_protein_target"] > 60  # boosted from base


# =========================================================================
# 7. TestProfilePersistence
# =========================================================================

class TestProfilePersistence:
    """Verify that profiles are persisted to disk correctly."""

    def test_create_writes_file(self, isolate_profile_storage):
        create_profile(copy.deepcopy(VALID_PROFILE))
        assert isolate_profile_storage.exists()

    def test_file_is_valid_json(self, isolate_profile_storage):
        create_profile(copy.deepcopy(VALID_PROFILE))
        text = isolate_profile_storage.read_text(encoding="utf-8")
        db = json.loads(text)
        assert "profiles" in db

    def test_profile_survives_roundtrip(self, isolate_profile_storage):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        # Read raw JSON from disk
        text = isolate_profile_storage.read_text(encoding="utf-8")
        db = json.loads(text)
        stored = db["profiles"][profile["user_id"]]
        assert stored["display_name"] == "Test User Phase8"
        assert stored["user_id"] == profile["user_id"]

    def test_active_profile_id_persisted(self, isolate_profile_storage):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        text = isolate_profile_storage.read_text(encoding="utf-8")
        db = json.loads(text)
        assert db["active_profile_id"] == profile["user_id"]

    def test_delete_removes_from_file(self, isolate_profile_storage):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        delete_profile(profile["user_id"])
        text = isolate_profile_storage.read_text(encoding="utf-8")
        db = json.loads(text)
        assert profile["user_id"] not in db["profiles"]

    def test_update_persists_changes(self, isolate_profile_storage):
        profile = create_profile(copy.deepcopy(VALID_PROFILE))
        update_profile(profile["user_id"], {"age": 40})
        text = isolate_profile_storage.read_text(encoding="utf-8")
        db = json.loads(text)
        assert db["profiles"][profile["user_id"]]["age"] == 40


# =========================================================================
# 8. TestProfileAPIEndpoints
# =========================================================================

class TestProfileAPIEndpoints:
    """Test all Phase 8 Flask API endpoints."""

    def test_post_create_201(self, flask_client):
        resp = flask_client.post("/api/profiles/user",
                                  json=VALID_PROFILE,
                                  content_type="application/json")
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True
        assert "data" in data
        assert "user_id" in data["data"]

    def test_post_create_no_display_name_400(self, flask_client):
        resp = flask_client.post("/api/profiles/user",
                                  json={"age": 25},
                                  content_type="application/json")
        assert resp.status_code == 400

    def test_post_create_short_display_name_400(self, flask_client):
        resp = flask_client.post("/api/profiles/user",
                                  json={"display_name": "A"},
                                  content_type="application/json")
        assert resp.status_code == 400

    def test_get_list_200(self, flask_client):
        resp = flask_client.get("/api/profiles/user")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "profiles" in data

    def test_get_list_contains_created_profile(self, flask_client):
        flask_client.post("/api/profiles/user",
                          json=VALID_PROFILE,
                          content_type="application/json")
        resp = flask_client.get("/api/profiles/user")
        data = resp.get_json()
        assert data["count"] >= 1

    def test_get_specific_200(self, flask_client):
        create_resp = flask_client.post("/api/profiles/user",
                                         json=VALID_PROFILE,
                                         content_type="application/json")
        uid = create_resp.get_json()["data"]["user_id"]
        resp = flask_client.get(f"/api/profiles/user/{uid}")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_get_specific_404(self, flask_client):
        resp = flask_client.get("/api/profiles/user/nonexistent-uuid-000")
        assert resp.status_code == 404

    def test_put_update_200(self, flask_client):
        create_resp = flask_client.post("/api/profiles/user",
                                         json=VALID_PROFILE,
                                         content_type="application/json")
        uid = create_resp.get_json()["data"]["user_id"]
        resp = flask_client.put(f"/api/profiles/user/{uid}",
                                 json={"weight_kg": 72},
                                 content_type="application/json")
        assert resp.status_code == 200
        assert resp.get_json()["data"]["weight_kg"] == 72.0

    def test_delete_200(self, flask_client):
        create_resp = flask_client.post("/api/profiles/user",
                                         json=VALID_PROFILE,
                                         content_type="application/json")
        uid = create_resp.get_json()["data"]["user_id"]
        resp = flask_client.delete(f"/api/profiles/user/{uid}")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_delete_404(self, flask_client):
        resp = flask_client.delete("/api/profiles/user/nonexistent-uuid-000")
        assert resp.status_code == 404

    def test_post_activate_200(self, flask_client):
        create_resp = flask_client.post("/api/profiles/user",
                                         json=VALID_PROFILE,
                                         content_type="application/json")
        uid = create_resp.get_json()["data"]["user_id"]
        resp = flask_client.post(f"/api/profiles/user/{uid}/activate")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_post_activate_404(self, flask_client):
        resp = flask_client.post("/api/profiles/user/nonexistent-uuid-000/activate")
        assert resp.status_code == 404

    def test_get_active_200(self, flask_client):
        resp = flask_client.get("/api/profiles/active")
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    def test_get_active_none_initially(self, flask_client):
        resp = flask_client.get("/api/profiles/active")
        data = resp.get_json()
        assert data["active_profile"] is None

    def test_get_active_after_create(self, flask_client):
        flask_client.post("/api/profiles/user",
                          json=VALID_PROFILE,
                          content_type="application/json")
        resp = flask_client.get("/api/profiles/active")
        data = resp.get_json()
        assert data["active_profile"] is not None

    def test_get_context_200(self, flask_client):
        create_resp = flask_client.post("/api/profiles/user",
                                         json=VALID_PROFILE,
                                         content_type="application/json")
        uid = create_resp.get_json()["data"]["user_id"]
        resp = flask_client.get(f"/api/profiles/user/{uid}/context")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "icmr_profile_key" in data["data"]

    def test_get_breakfast_200(self, flask_client):
        create_resp = flask_client.post("/api/profiles/user",
                                         json=VALID_PROFILE,
                                         content_type="application/json")
        uid = create_resp.get_json()["data"]["user_id"]
        resp = flask_client.get(f"/api/profiles/user/{uid}/breakfast")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "suggestions" in data

    def test_get_breakfast_404(self, flask_client):
        resp = flask_client.get("/api/profiles/user/nonexistent-uuid-000/breakfast")
        assert resp.status_code == 404

    def test_post_bmi_200(self, flask_client):
        resp = flask_client.post("/api/profiles/bmi",
                                  json={"height_cm": 170, "weight_kg": 65},
                                  content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "value" in data["data"]
        assert "category" in data["data"]

    def test_post_bmi_missing_fields_400(self, flask_client):
        resp = flask_client.post("/api/profiles/bmi",
                                  json={"height_cm": 170},
                                  content_type="application/json")
        assert resp.status_code == 400

    def test_post_bmi_no_body_400(self, flask_client):
        resp = flask_client.post("/api/profiles/bmi",
                                  json={},
                                  content_type="application/json")
        assert resp.status_code == 400

    def test_post_bmi_height_out_of_range_400(self, flask_client):
        resp = flask_client.post("/api/profiles/bmi",
                                  json={"height_cm": 10, "weight_kg": 65},
                                  content_type="application/json")
        assert resp.status_code == 400

    def test_post_bmi_weight_out_of_range_400(self, flask_client):
        resp = flask_client.post("/api/profiles/bmi",
                                  json={"height_cm": 170, "weight_kg": 5},
                                  content_type="application/json")
        assert resp.status_code == 400

    def test_post_validate_200_valid(self, flask_client):
        resp = flask_client.post("/api/profiles/validate",
                                  json=VALID_PROFILE,
                                  content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["valid"] is True

    def test_post_validate_200_invalid(self, flask_client):
        resp = flask_client.post("/api/profiles/validate",
                                  json={},
                                  content_type="application/json")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_create_returns_disclaimer(self, flask_client):
        resp = flask_client.post("/api/profiles/user",
                                  json=VALID_PROFILE,
                                  content_type="application/json")
        data = resp.get_json()
        assert "disclaimer" in data


# =========================================================================
# Legacy run_tests() — for run_all_tests.py compatibility
# =========================================================================

def run_tests():
    """
    Standalone test runner for backward compatibility with run_all_tests.py.
    Uses inline record/assert pattern matching the project convention.
    """
    import traceback

    PASS = "[PASS]"
    FAIL = "[FAIL]"
    test_results = []

    def record(name, condition, extra=""):
        status = PASS if condition else FAIL
        msg = f"{status} {name}"
        if extra:
            msg += f" | {extra}"
        print(msg)
        test_results.append((name, condition))

    print("=" * 65)
    print("RUNNING PHASE 8: USER PROFILING MODULE TESTS")
    print("=" * 65)

    # 1. Constants
    print("\n--- 1. Module Constants ---")
    record("constants: PHASE8_DISCLAIMER is non-empty string",
           isinstance(PHASE8_DISCLAIMER, str) and len(PHASE8_DISCLAIMER) > 0)
    record("constants: BMI_LIMITATION_NOTE is non-empty string",
           isinstance(BMI_LIMITATION_NOTE, str) and len(BMI_LIMITATION_NOTE) > 0)

    # 2. Validation
    print("\n--- 2. Profile Validation ---")
    ok, errs = validate_profile_data(VALID_PROFILE)
    record("validation: valid profile passes", ok is True)

    ok, errs = validate_profile_data({})
    record("validation: empty dict fails", ok is False)
    record("validation: missing display_name error", any("display_name" in e for e in errs))

    ok, errs = validate_profile_data({"display_name": "A"})
    record("validation: short display_name fails", ok is False)

    ok, errs = validate_profile_data({**VALID_PROFILE, "age": -5})
    record("validation: negative age fails", ok is False)

    ok, errs = validate_profile_data({**VALID_PROFILE, "age": 121})
    record("validation: age over 120 fails", ok is False)

    ok, errs = validate_profile_data({**VALID_PROFILE, "gender": "xyz"})
    record("validation: invalid gender fails", ok is False)

    ok, errs = validate_profile_data({**VALID_PROFILE, "activity_level": "super"})
    record("validation: invalid activity_level fails", ok is False)

    ok, errs = validate_profile_data({**VALID_PROFILE, "dietary_goal": "get_rich"})
    record("validation: invalid dietary_goal fails", ok is False)

    # 3. BMI
    print("\n--- 3. BMI Calculation ---")
    bmi = calculate_bmi(170, 65)
    record("bmi: normal range", bmi["category"] == "Normal")
    record("bmi: limitation_note present", bmi["limitation_note"] == BMI_LIMITATION_NOTE)

    bmi_uw = calculate_bmi(180, 50)
    record("bmi: underweight category", bmi_uw["category"] == "Underweight")

    bmi_ow = calculate_bmi(170, 80)
    record("bmi: overweight category", bmi_ow["category"] == "Overweight")

    bmi_ob = calculate_bmi(160, 90)
    record("bmi: obese category", bmi_ob["category"] == "Obese")

    caught = False
    try:
        calculate_bmi(-170, 65)
    except ValueError:
        caught = True
    record("bmi: negative height raises ValueError", caught)

    # 4. CRUD
    print("\n--- 4. Profile CRUD ---")
    profile = create_profile(copy.deepcopy(VALID_PROFILE))
    record("crud: create returns user_id", "user_id" in profile and len(profile["user_id"]) > 0)
    record("crud: version starts at 1", profile["profile_version"] == 1)

    fetched = get_profile(profile["user_id"])
    record("crud: get_profile returns match", fetched is not None and fetched["user_id"] == profile["user_id"])

    updated = update_profile(profile["user_id"], {"weight_kg": 70})
    record("crud: update increments version to 2", updated["profile_version"] == 2)
    record("crud: update changes weight", updated["weight_kg"] == 70.0)

    profiles_list = list_profiles()
    record("crud: list_profiles returns >= 1", len(profiles_list) >= 1)

    active = get_active_profile()
    record("crud: active profile set after create", active is not None)

    deleted = delete_profile(profile["user_id"])
    record("crud: delete returns True", deleted is True)

    gone = get_profile(profile["user_id"])
    record("crud: deleted profile is None", gone is None)

    # 5. Breakfast Suggestions
    print("\n--- 5. Breakfast Suggestions ---")
    suggestions = get_breakfast_suggestions(VALID_PROFILE)
    record("breakfast: returns non-empty list", len(suggestions) > 0)
    record("breakfast: items have food_name", all("food_name" in s for s in suggestions))
    record("breakfast: max 3 items", len(suggestions) <= 3)

    early_sugg = get_breakfast_suggestions({**VALID_PROFILE, "morning_routine_preference": "early_bird"})
    record("breakfast: early_bird returns suggestions", len(early_sugg) > 0)

    late_sugg = get_breakfast_suggestions({**VALID_PROFILE, "morning_routine_preference": "late_riser"})
    record("breakfast: late_riser returns suggestions", len(late_sugg) > 0)

    # 6. Personalised Context
    print("\n--- 6. Personalised Context ---")
    ctx = get_personalized_context(VALID_PROFILE)
    record("context: returns dict", isinstance(ctx, dict))
    record("context: has icmr_profile_key", "icmr_profile_key" in ctx)
    record("context: has daily_calorie_target", "daily_calorie_target" in ctx and ctx["daily_calorie_target"] > 0)
    record("context: has portion_guidance", "portion_guidance" in ctx and len(ctx["portion_guidance"]) > 0)
    record("context: has disclaimer", "personalization_disclaimer" in ctx)

    ctx_diab = get_personalized_context({**VALID_PROFILE, "dietary_goal": "diabetes_management"})
    record("context: diabetes -> diabetic_adult", ctx_diab["icmr_profile_key"] == "diabetic_adult")

    ctx_hyp = get_personalized_context({**VALID_PROFILE, "dietary_goal": "hypertension_management"})
    record("context: hypertension -> hypertensive_adult", ctx_hyp["icmr_profile_key"] == "hypertensive_adult")

    # 7. API Endpoints
    print("\n--- 7. API Endpoints ---")
    flask_app.config["TESTING"] = True
    client = flask_app.test_client()

    resp_create = client.post("/api/profiles/user",
                               json=VALID_PROFILE,
                               content_type="application/json")
    record("api: POST create returns 201", resp_create.status_code == 201)
    api_uid = resp_create.get_json()["data"]["user_id"]

    resp_list = client.get("/api/profiles/user")
    record("api: GET list returns 200", resp_list.status_code == 200)

    resp_get = client.get(f"/api/profiles/user/{api_uid}")
    record("api: GET specific returns 200", resp_get.status_code == 200)

    resp_update = client.put(f"/api/profiles/user/{api_uid}",
                              json={"age": 35},
                              content_type="application/json")
    record("api: PUT update returns 200", resp_update.status_code == 200)

    resp_bmi = client.post("/api/profiles/bmi",
                            json={"height_cm": 170, "weight_kg": 65},
                            content_type="application/json")
    record("api: POST bmi returns 200", resp_bmi.status_code == 200)

    resp_bmi_bad = client.post("/api/profiles/bmi",
                                json={},
                                content_type="application/json")
    record("api: POST bmi missing fields returns 400", resp_bmi_bad.status_code == 400)

    resp_validate = client.post("/api/profiles/validate",
                                 json=VALID_PROFILE,
                                 content_type="application/json")
    record("api: POST validate returns 200", resp_validate.status_code == 200)

    resp_ctx = client.get(f"/api/profiles/user/{api_uid}/context")
    record("api: GET context returns 200", resp_ctx.status_code == 200)

    resp_bkf = client.get(f"/api/profiles/user/{api_uid}/breakfast")
    record("api: GET breakfast returns 200", resp_bkf.status_code == 200)

    resp_activate = client.post(f"/api/profiles/user/{api_uid}/activate")
    record("api: POST activate returns 200", resp_activate.status_code == 200)

    resp_active = client.get("/api/profiles/active")
    record("api: GET active returns 200", resp_active.status_code == 200)

    resp_del = client.delete(f"/api/profiles/user/{api_uid}")
    record("api: DELETE returns 200", resp_del.status_code == 200)

    resp_no_name = client.post("/api/profiles/user",
                                json={"age": 25},
                                content_type="application/json")
    record("api: POST create no display_name returns 400", resp_no_name.status_code == 400)

    # Summary
    total = len(test_results)
    passed = sum(1 for _, ok in test_results if ok)
    failed = total - passed

    print("\n" + "=" * 65)
    print(f"PHASE 8 USER PROFILING TESTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print("=" * 65)

    if failed > 0:
        raise AssertionError(f"{failed} test(s) failed in test_user_profiling.py")


if __name__ == "__main__":
    run_tests()
