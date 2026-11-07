"""
tests/test_nutrition_analysis.py
Comprehensive Integration & Unit Tests for Phase 7: Nutrition Analysis Module

Verifies:
 1. Food Database Lookup (Master KB, Subtypes, Unified 1,556 items)
 2. Portion Conversion & Defined Servings (grams, piece, katori, plate, bowl, cup)
 3. Standard Formula Scaling:
      nutrient_amount = (portion_grams / 100.0) * nutrient_value_per_100g
 4. Complete Nutrient Spectrum (Calories, Macros, Micros, Vitamins & Minerals)
 5. Missing Data Integrity (Missing fields remain None; ZERO is NEVER substituted)
 6. Data Quality Flags & Source Transparency
 7. Manual Food Search Fallback
 8. Error Handling for Invalid Inputs (negative, zero, NaN, non-numeric, unknown units)
 9. Web API Endpoints: /api/nutrition/search, /api/nutrition/servings, /api/nutrition/analyze
10. Explicit Non-Claim Disclaimer (no exact portion/calories from vision alone)
"""

import math
from pathlib import Path
import sys
import types
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

from app import app as flask_app
from src.nutrition_analyzer import (
    NutritionAnalyzer,
    get_analyzer,
    lookup_food,
    convert_portion_to_grams,
    calculate_nutrients,
    search_foods,
    get_available_servings,
    NUTRIENT_UNITS,
    PHASE7_DISCLAIMER,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyzer():
    """Module-scoped NutritionAnalyzer singleton."""
    return get_analyzer()


@pytest.fixture(scope="module")
def client():
    """Flask test client."""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def samosa_record(analyzer):
    """Pre-fetched standardized record for 'samosa'."""
    return analyzer.lookup_food("samosa")


# ===========================================================================
# 1. Food Database Lookup & Subtype Connection
# ===========================================================================

class TestFoodLookup:
    """Verifies connection of confirmed food name to cleaned nutrition database."""

    def test_lookup_primary_vision_class(self, samosa_record):
        """lookup_food finds primary vision class 'samosa' by name."""
        assert samosa_record is not None
        assert samosa_record["class_name"] == "samosa"

    def test_lookup_has_per_100g_nutrients(self, samosa_record):
        """Found record contains per_100g_nutrients with positive calorie baseline."""
        assert "per_100g_nutrients" in samosa_record
        cal = samosa_record["per_100g_nutrients"].get("calories_kcal")
        assert cal is not None and cal > 0

    def test_lookup_samosa_reference_serving_65g(self, samosa_record):
        """Samosa reference serving is 65g per ICMR-NIN standard."""
        assert samosa_record["reference_serving_g"] == 65.0

    def test_lookup_subtype_keema_samosa(self):
        """Resolves specific culinary subtype 'samosa_keema'."""
        result = lookup_food("samosa", subtype_id="samosa_keema")
        assert result is not None
        assert result.get("selected_subtype") == "samosa_keema"

    def test_lookup_subtype_keema_calories(self):
        """Keema samosa subtype has correct per-serving calories of 318 kcal."""
        result = lookup_food("samosa", subtype_id="samosa_keema")
        assert result is not None
        assert result["reference_nutrients"]["calories_kcal"] == 318.0

    def test_lookup_composite_recipe_aloo_gobi(self):
        """Finds composite recipe entry 'aloo_gobi' with COMPOSITE_RECIPE match_type."""
        result = lookup_food("aloo_gobi")
        assert result is not None
        assert result["match_type"] == "COMPOSITE_RECIPE"

    def test_lookup_unified_dataset_fallback(self):
        """Finds item from unified 1,556-item dataset (chai / hot tea)."""
        result = lookup_food("hot tea (garam chai)")
        assert result is not None
        assert "tea" in result["display_name"].lower() or "chai" in result["display_name"].lower()

    def test_lookup_nonexistent_returns_none(self):
        """Non-existent dish returns None — no fabricated data."""
        result = lookup_food("non_existent_dish_xyz_999")
        assert result is None

    def test_lookup_empty_string_returns_none(self):
        """Empty string query returns None gracefully."""
        result = lookup_food("")
        assert result is None

    def test_lookup_none_query_returns_none(self):
        """None query returns None gracefully (no crash)."""
        result = lookup_food(None)  # type: ignore[arg-type]
        assert result is None

    def test_lookup_has_source_field(self, samosa_record):
        """Found record reports data source for transparency."""
        assert bool(samosa_record.get("source"))

    def test_lookup_has_reference_basis(self, samosa_record):
        """Found record reports reference basis (what the values represent)."""
        assert bool(samosa_record.get("reference_basis"))


# ===========================================================================
# 2. Portion Conversion & Defined Serving Units
# ===========================================================================

class TestPortionConversion:
    """Verifies convert_portion_to_grams handles all supported units correctly."""

    def test_direct_grams_integer(self):
        assert convert_portion_to_grams(150, "g") == 150.0

    def test_direct_grams_float_string(self):
        assert convert_portion_to_grams("175.5", "g") == 175.5

    def test_100g_baseline_unit(self):
        assert convert_portion_to_grams(1.0, "100g") == 100.0

    def test_one_piece_samosa(self):
        """1 piece of samosa = 65.0g (from standard serving weight table)."""
        result = convert_portion_to_grams(1, "piece", food_name="samosa")
        assert result == 65.0, f"Expected 65.0g, got {result}"

    def test_two_pieces_samosa(self):
        """piece_2 unit yields 2 × reference serving (130g for samosa)."""
        result = convert_portion_to_grams(1, "piece_2", food_name="samosa")
        assert result == 130.0, f"Expected 130.0g, got {result}"

    def test_half_piece_samosa(self):
        """piece_half unit yields 0.5 × reference serving (32.5g for samosa)."""
        result = convert_portion_to_grams(1, "piece_half", food_name="samosa")
        assert result == 32.5, f"Expected 32.5g, got {result}"

    def test_one_plate_biryani(self):
        """1 plate of biryani = 250g."""
        result = convert_portion_to_grams(1, "plate", food_name="biryani")
        assert result == 250.0, f"Expected 250.0g, got {result}"

    def test_one_katori_dal(self):
        """1 katori of dal = 150g (medium katori)."""
        result = convert_portion_to_grams(1, "katori", food_name="dal")
        assert result == 150.0, f"Expected 150.0g, got {result}"

    def test_one_piece_idli(self):
        """1 piece of idli = 80g (from standard serving table)."""
        result = convert_portion_to_grams(1, "piece", food_name="idli")
        assert result == 80.0, f"Expected 80.0g, got {result}"

    def test_one_piece_chapati(self):
        """1 chapati = 35g (from standard serving table)."""
        result = convert_portion_to_grams(1, "piece", food_name="chapati")
        assert result == 35.0, f"Expected 35.0g, got {result}"

    def test_one_cup(self):
        """1 cup defaults to 150g."""
        result = convert_portion_to_grams(1, "cup")
        assert result == 150.0

    def test_decimal_grams(self):
        """Decimal gram value is preserved correctly."""
        result = convert_portion_to_grams(87.5, "g")
        assert result == 87.5

    def test_available_servings_returns_multiple_options(self):
        """get_available_servings returns >= 3 options for samosa."""
        servings = get_available_servings("samosa")
        assert len(servings) >= 3

    def test_available_servings_contains_piece_option(self):
        """get_available_servings for samosa includes a piece-based option."""
        servings = get_available_servings("samosa")
        unit_names = [s["unit"] for s in servings]
        assert any("piece" in u for u in unit_names)

    def test_available_servings_contains_100g_option(self):
        """get_available_servings always includes 100g baseline."""
        servings = get_available_servings("samosa")
        assert any(s["unit"] == "100g" for s in servings)


# ===========================================================================
# 3. Invalid Inputs & Graceful Error Handling
# ===========================================================================

class TestInvalidInputs:
    """Ensures all invalid inputs are caught with descriptive errors."""

    def test_negative_portion_raises_valueerror(self):
        with pytest.raises(ValueError, match="greater than zero"):
            convert_portion_to_grams(-50, "g")

    def test_zero_portion_raises_valueerror(self):
        with pytest.raises(ValueError):
            convert_portion_to_grams(0, "g")

    def test_nan_portion_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            convert_portion_to_grams(float("nan"), "g")

    def test_inf_portion_raises_valueerror(self):
        with pytest.raises(ValueError, match="finite"):
            convert_portion_to_grams(float("inf"), "g")

    def test_nonnumeric_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="valid number"):
            convert_portion_to_grams("not_a_number", "g")

    def test_unsupported_unit_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unsupported portion unit"):
            convert_portion_to_grams(100, "unsupported_unit_xyz")

    def test_none_portion_raises_valueerror(self):
        with pytest.raises(ValueError, match="cannot be None"):
            convert_portion_to_grams(None, "g")  # type: ignore[arg-type]

    def test_calculate_nutrients_unknown_food_raises_keyerror(self):
        with pytest.raises(KeyError):
            calculate_nutrients("imaginary_food_unknown_xyz", 100.0)

    def test_calculate_nutrients_negative_portion_raises_valueerror(self):
        with pytest.raises(ValueError):
            calculate_nutrients("samosa", -50.0)

    def test_calculate_nutrients_zero_portion_raises_valueerror(self):
        with pytest.raises(ValueError):
            calculate_nutrients("samosa", 0.0)

    def test_calculate_nutrients_none_portion_raises_valueerror(self):
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_nutrients("samosa", None)  # type: ignore[arg-type]


# ===========================================================================
# 4. Standard Formula Nutrient Scaling
# ===========================================================================

class TestNutrientScalingFormula:
    """
    Verifies the core per-100g scaling formula:
        nutrient_amount = (portion_grams / 100.0) * nutrient_value_per_100g
    """

    @pytest.fixture(scope="class")
    def samosa_per_100g_cal(self, analyzer):
        food = analyzer.lookup_food("samosa")
        return food["per_100g_nutrients"]["calories_kcal"]

    def test_100g_portion_matches_per_100g_baseline(self, analyzer, samosa_per_100g_cal):
        calc = calculate_nutrients("samosa", 100.0)
        val = calc["nutrients"]["calories_kcal"]["value"]
        assert abs(val - samosa_per_100g_cal) < 0.1, (
            f"100g calc {val} ≠ per_100g baseline {samosa_per_100g_cal}"
        )

    def test_200g_portion_is_exactly_2x(self, analyzer, samosa_per_100g_cal):
        calc = calculate_nutrients("samosa", 200.0)
        val = calc["nutrients"]["calories_kcal"]["value"]
        expected = 2.0 * samosa_per_100g_cal
        assert abs(val - expected) < 0.2, f"200g val={val}, expected≈{expected}"

    def test_65g_portion_follows_formula(self, analyzer, samosa_per_100g_cal):
        calc = calculate_nutrients("samosa", 65.0)
        val = calc["nutrients"]["calories_kcal"]["value"]
        expected = round(0.65 * samosa_per_100g_cal, 2)
        assert abs(val - expected) < 0.1, f"65g val={val}, expected≈{expected}"

    def test_fractional_portion_follows_formula(self, analyzer, samosa_per_100g_cal):
        calc = calculate_nutrients("samosa", 32.5)
        val = calc["nutrients"]["calories_kcal"]["value"]
        expected = round(0.325 * samosa_per_100g_cal, 2)
        assert abs(val - expected) < 0.1

    def test_protein_has_gram_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["protein_g"]["unit"] == "g"
        assert calc["nutrients"]["protein_g"]["value"] is not None
        assert calc["nutrients"]["protein_g"]["value"] > 0

    def test_carbs_has_gram_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["carbs_g"]["unit"] == "g"
        assert calc["nutrients"]["carbs_g"]["value"] is not None

    def test_fat_has_gram_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["fat_g"]["unit"] == "g"
        assert calc["nutrients"]["fat_g"]["value"] is not None

    def test_fibre_has_gram_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["fibre_g"]["unit"] == "g"

    def test_sodium_has_mg_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["sodium_mg"]["unit"] == "mg"

    def test_calcium_has_mg_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["calcium_mg"]["unit"] == "mg"

    def test_iron_has_mg_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["iron_mg"]["unit"] == "mg"

    def test_folate_has_mcg_unit(self):
        calc = calculate_nutrients("samosa", 100.0)
        assert calc["nutrients"]["folate_ug"]["unit"] == "mcg"

    def test_portion_grams_in_result(self):
        calc = calculate_nutrients("samosa", 65.0)
        assert calc["portion"]["grams"] == 65.0

    def test_scale_ratio_in_result(self):
        calc = calculate_nutrients("samosa", 200.0)
        assert abs(calc["portion"]["scale_ratio_from_100g"] - 2.0) < 0.001

    def test_all_nutrient_keys_present(self):
        calc = calculate_nutrients("samosa", 100.0)
        for key in NUTRIENT_UNITS:
            assert key in calc["nutrients"], f"Missing nutrient key: {key}"


# ===========================================================================
# 5. Missing Nutrition Data Integrity (Never Zero-Filled)
# ===========================================================================

class TestMissingDataIntegrity:
    """
    Ensures that missing nutrient fields are reported as None / DATA_NOT_AVAILABLE
    and are NEVER substituted with 0.
    """

    @pytest.fixture(scope="class")
    def partial_food(self):
        return {
            "food_id": "test_partial_ph7",
            "class_name": "test_partial_ph7",
            "display_name": "Test Partial Item (Phase 7)",
            "serving_size_g": 100.0,
            "serving_unit": "100g",
            "calories_kcal": 200.0,
            "protein_g": 5.0,
            "carbs_g": 30.0,
            "fat_g": 4.0,
            "fibre_g": 2.0,
            "sodium_mg": 150.0,
            "calcium_mg": None,     # deliberately missing
            "iron_mg": None,        # deliberately missing
            "vitamin_c_mg": None,   # deliberately missing
            "folate_ug": None,      # deliberately missing
            "data_source": "PHASE7_TEST_SOURCE",
        }

    @pytest.fixture(scope="class")
    def partial_calc(self, partial_food):
        return calculate_nutrients(partial_food, 150.0)

    def test_missing_calcium_is_none_not_zero(self, partial_calc):
        assert partial_calc["nutrients"]["calcium_mg"]["value"] is None, (
            "Missing calcium MUST remain None, not zero"
        )

    def test_missing_iron_is_none_not_zero(self, partial_calc):
        assert partial_calc["nutrients"]["iron_mg"]["value"] is None, (
            "Missing iron MUST remain None, not zero"
        )

    def test_missing_vitamin_c_is_none_not_zero(self, partial_calc):
        assert partial_calc["nutrients"]["vitamin_c_mg"]["value"] is None

    def test_missing_folate_is_none_not_zero(self, partial_calc):
        assert partial_calc["nutrients"]["folate_ug"]["value"] is None

    def test_missing_nutrient_status_is_data_not_available(self, partial_calc):
        status = partial_calc["nutrients"]["calcium_mg"]["status"]
        assert status == "DATA_NOT_AVAILABLE", f"Got status: {status}"

    def test_missing_nutrients_reported_in_list(self, partial_calc):
        missing_keys = [m["nutrient"] for m in partial_calc["missing_nutrients"]]
        assert "calcium_mg" in missing_keys
        assert "iron_mg" in missing_keys
        assert "vitamin_c_mg" in missing_keys
        assert "folate_ug" in missing_keys

    def test_missing_nutrients_has_descriptive_reason(self, partial_calc, partial_food):
        """Reason must reference the actual source string."""
        source = partial_food["data_source"]
        for entry in partial_calc["missing_nutrients"]:
            assert source in entry["reason"], (
                f"Source '{source}' not found in reason: {entry['reason']}"
            )

    def test_available_macros_are_calculated_correctly(self, partial_calc):
        """Available macros (calories, protein…) are properly scaled for 150g."""
        cal = partial_calc["nutrients"]["calories_kcal"]["value"]
        expected_cal = round(1.5 * 200.0, 2)   # 150g / 100 × 200
        assert abs(cal - expected_cal) < 0.1, f"cal={cal}, expected≈{expected_cal}"

        prot = partial_calc["nutrients"]["protein_g"]["value"]
        expected_prot = round(1.5 * 5.0, 2)
        assert abs(prot - expected_prot) < 0.05


# ===========================================================================
# 6. Source Transparency & Non-Claim Disclaimer
# ===========================================================================

class TestSourceTransparencyAndDisclaimer:
    """Verifies Phase 7 transparency requirements are met in every result."""

    @pytest.fixture(scope="class")
    def samosa_calc(self):
        return calculate_nutrients("samosa", 100.0)

    def test_source_field_reported(self, samosa_calc):
        assert bool(samosa_calc.get("source"))

    def test_reference_basis_reported(self, samosa_calc):
        assert bool(samosa_calc.get("reference_basis"))

    def test_reference_serving_g_reported(self, samosa_calc):
        assert samosa_calc.get("reference_serving_g") is not None

    def test_quality_flags_present(self, samosa_calc):
        flags = samosa_calc.get("data_quality_flags", {})
        assert "sodium_flag" in flags
        assert "calorie_flag" in flags

    def test_disclaimer_is_non_empty(self, samosa_calc):
        disc = samosa_calc.get("disclaimer", "")
        assert len(disc) > 20

    def test_disclaimer_states_portion_weight_must_be_confirmed(self, samosa_calc):
        disc = samosa_calc.get("disclaimer", "").lower()
        assert "portion weight must be confirmed" in disc, (
            "Disclaimer must state portion weight must be confirmed by user"
        )

    def test_disclaimer_states_category_only_from_vision(self, samosa_calc):
        disc = samosa_calc.get("disclaimer", "").lower()
        assert "category only" in disc, (
            "Disclaimer must state vision identifies category only"
        )

    def test_phase7_disclaimer_constant_is_complete(self):
        """PHASE7_DISCLAIMER module constant satisfies both transparency requirements."""
        lower = PHASE7_DISCLAIMER.lower()
        assert "category only" in lower
        assert "portion weight must be confirmed" in lower

    def test_ambiguous_subtype_info_in_result(self, samosa_calc):
        """Result contains ambiguous_subtype section."""
        assert "ambiguous_subtype" in samosa_calc
        assert "has_subtypes" in samosa_calc["ambiguous_subtype"]


# ===========================================================================
# 7. Manual Food Search Fallback
# ===========================================================================

class TestManualFoodSearch:
    """Verifies search_foods() works as fallback when image recognition fails."""

    def test_search_samosa_returns_matches(self):
        results = search_foods("samosa")
        assert len(results) > 0

    def test_search_samosa_top_match_is_samosa(self):
        results = search_foods("samosa")
        assert "samosa" in results[0]["food_name"].lower()

    def test_search_chai_returns_matches_from_unified(self):
        results = search_foods("chai")
        assert len(results) > 0

    def test_search_roti_returns_matches(self):
        results = search_foods("roti")
        assert len(results) > 0

    def test_search_empty_query_returns_empty_list(self):
        assert search_foods("") == []

    def test_search_none_query_returns_empty_list(self):
        assert search_foods(None) == []  # type: ignore[arg-type]

    def test_search_respects_limit(self):
        results = search_foods("rice", limit=3)
        assert len(results) <= 3

    def test_search_result_has_required_fields(self):
        results = search_foods("idli", limit=1)
        if results:
            r = results[0]
            for field in ["food_id", "food_name", "display_name", "match_type", "source"]:
                assert field in r, f"Missing field: {field}"

    def test_search_result_has_calorie_info(self):
        results = search_foods("idli", limit=1)
        if results:
            r = results[0]
            # calories_per_100g may be None for some items but key must exist
            assert "calories_per_100g" in r

    def test_search_benchmark_classes_ranked_first(self):
        """Master KB benchmark foods are ranked ahead of unified lookup items."""
        results = search_foods("biryani")
        assert results, "Expected at least one result for 'biryani'"
        # Check that is_benchmark_class=True appears high in ranking
        benchmark_indices = [i for i, r in enumerate(results) if r.get("is_benchmark_class")]
        if benchmark_indices:
            assert benchmark_indices[0] < 5, "Benchmark class should be in top 5 results"


# ===========================================================================
# 8. Web API Endpoints: /api/nutrition/*
# ===========================================================================

class TestNutritionAPIEndpoints:
    """Integration tests for all Phase 7 REST API endpoints."""

    # --- GET /api/nutrition/search ---

    def test_search_api_returns_200(self, client):
        resp = client.get("/api/nutrition/search?q=biryani")
        assert resp.status_code == 200

    def test_search_api_success_true(self, client):
        resp = client.get("/api/nutrition/search?q=biryani")
        data = resp.get_json()
        assert data.get("success") is True

    def test_search_api_returns_matches(self, client):
        resp = client.get("/api/nutrition/search?q=biryani")
        data = resp.get_json()
        assert data.get("count", 0) > 0

    def test_search_api_empty_query_returns_zero_count(self, client):
        resp = client.get("/api/nutrition/search?q=")
        data = resp.get_json()
        assert data.get("success") is True
        assert data.get("count", 0) == 0

    def test_search_api_respects_limit_param(self, client):
        resp = client.get("/api/nutrition/search?q=rice&limit=3")
        data = resp.get_json()
        assert len(data.get("data", [])) <= 3

    # --- GET /api/nutrition/servings ---

    def test_servings_api_returns_200(self, client):
        resp = client.get("/api/nutrition/servings?food_name=samosa")
        assert resp.status_code == 200

    def test_servings_api_contains_multiple_options(self, client):
        resp = client.get("/api/nutrition/servings?food_name=samosa")
        data = resp.get_json()
        assert len(data.get("servings", [])) >= 2

    def test_servings_api_missing_food_name_returns_400(self, client):
        resp = client.get("/api/nutrition/servings")
        assert resp.status_code == 400

    # --- POST /api/nutrition/analyze ---

    def test_analyze_api_defined_serving_2_pieces_returns_200(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 2, "unit": "piece"}
        )
        assert resp.status_code == 200

    def test_analyze_api_success_true(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 2, "unit": "piece"}
        )
        data = resp.get_json()
        assert data.get("success") is True

    def test_analyze_api_portion_is_130g(self, client):
        """2 pieces of samosa = 2 × 65g = 130g."""
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 2, "unit": "piece"}
        )
        data = resp.get_json()
        portion = data.get("data", {}).get("portion", {})
        assert portion.get("grams") == 130.0

    def test_analyze_api_calories_calculated(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 2, "unit": "piece"}
        )
        data = resp.get_json()
        cal = data.get("data", {}).get("nutrients", {}).get("calories_kcal", {}).get("value")
        assert cal is not None and cal > 0

    def test_analyze_api_health_evaluation_present(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 2, "unit": "piece"}
        )
        data = resp.get_json()
        health_eval = data.get("data", {}).get("health_evaluation", {})
        assert "suitability_score" in health_eval

    def test_analyze_api_missing_food_name_returns_400(self, client):
        resp = client.post("/api/nutrition/analyze", json={})
        assert resp.status_code == 400

    def test_analyze_api_negative_portion_returns_400(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": -10}
        )
        assert resp.status_code == 400

    def test_analyze_api_unknown_food_returns_404(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "non_existent_food_xyz_phase7"}
        )
        assert resp.status_code == 404

    def test_analyze_api_direct_grams_input(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "idli", "portion_value": 160, "unit": "g"}
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("success") is True
        assert data["data"]["portion"]["grams"] == 160.0

    def test_analyze_api_contains_disclaimer(self, client):
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 100, "unit": "g"}
        )
        data = resp.get_json()
        disc = data.get("data", {}).get("disclaimer", "")
        assert len(disc) > 20

    def test_analyze_api_contains_missing_nutrients_list(self, client):
        """API result includes missing_nutrients list (may be empty for complete records)."""
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 100, "unit": "g"}
        )
        data = resp.get_json()
        assert "missing_nutrients" in data.get("data", {})

    def test_analyze_api_available_servings_in_response(self, client):
        """API response includes available serving options for the food."""
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 100, "unit": "g"}
        )
        data = resp.get_json()
        assert "available_servings" in data.get("data", {})
        assert len(data["data"]["available_servings"]) >= 2

    def test_analyze_api_user_profile_hypertensive_no_crash(self, client):
        """Hypertensive profile runs without error."""
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "samosa", "portion_value": 100, "unit": "g",
                  "user_profile": "hypertensive_adult"}
        )
        assert resp.status_code == 200

    def test_analyze_api_user_profile_diabetic_no_crash(self, client):
        """Diabetic profile runs without error."""
        resp = client.post(
            "/api/nutrition/analyze",
            json={"food_name": "biryani", "portion_value": 250, "unit": "g",
                  "user_profile": "diabetic_adult"}
        )
        assert resp.status_code == 200


# ===========================================================================
# 9. Additional Phase 7 Nutrient Calculation Edge-Case Tests
# ===========================================================================

class TestNutrientEdgeCases:
    """Extra tests for reusable calculation correctness and stability."""

    def test_calculate_nutrients_accepts_dict_record_directly(self):
        """calculate_nutrients accepts a raw dict (not just a name string)."""
        food = lookup_food("idli")
        assert food is not None
        result = calculate_nutrients(food, 100.0)
        assert result["nutrients"]["calories_kcal"]["value"] is not None

    def test_calculate_nutrients_composite_recipe(self):
        """Composite recipe (aloo_gobi) is calculated correctly."""
        result = calculate_nutrients("aloo_gobi", 150.0)
        assert result is not None
        cal = result["nutrients"]["calories_kcal"]["value"]
        assert cal is not None and cal > 0

    def test_all_benchmark_foods_calculable(self, analyzer):
        """All 59 Master KB foods can be calculated at their reference serving."""
        failures = []
        for item in analyzer.master_kb_items:
            name = item.get("class_name", "")
            serving_g = float(item.get("serving_size_g") or 100.0)
            try:
                r = calculate_nutrients(name, serving_g)
                cal = r["nutrients"]["calories_kcal"]["value"]
                if cal is None:
                    failures.append(f"{name}: calories returned None")
            except Exception as e:
                failures.append(f"{name}: {e}")
        assert not failures, "Failed benchmark foods:\n" + "\n".join(failures)

    def test_nutrient_values_are_rounded_to_2dp(self):
        """Scaled nutrient values are rounded to max 2 decimal places."""
        result = calculate_nutrients("samosa", 65.0)
        for key, nut in result["nutrients"].items():
            val = nut.get("value")
            if val is not None:
                # Check at most 2 decimal places
                assert round(val, 2) == val, f"Key {key} value {val} has >2dp"

    def test_portion_scale_ratio_correct(self):
        result = calculate_nutrients("samosa", 50.0)
        assert abs(result["portion"]["scale_ratio_from_100g"] - 0.5) < 0.001

    def test_food_id_present_in_result(self):
        result = calculate_nutrients("samosa", 100.0)
        assert bool(result.get("food_id"))

    def test_display_name_present_in_result(self):
        result = calculate_nutrients("samosa", 100.0)
        assert bool(result.get("display_name"))

    def test_match_type_present_in_result(self):
        result = calculate_nutrients("samosa", 100.0)
        assert bool(result.get("match_type"))


# ---------------------------------------------------------------------------
# Legacy-compatible runner (for run_all_tests.py compatibility)
# ---------------------------------------------------------------------------

def run_tests():
    """
    Maintained for backward compatibility with run_all_tests.py which calls
    this module via subprocess with run_tests().
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
    print("RUNNING PHASE 7: NUTRITION ANALYSIS MODULE TESTS")
    print("=" * 65)

    _analyzer = get_analyzer()

    # 1. Lookup
    print("\n--- 1. Food Lookup & Culinary Subtype Resolution ---")
    samosa = lookup_food("samosa")
    record("lookup: finds primary vision class 'samosa'", samosa is not None and samosa["class_name"] == "samosa")
    record("lookup: samosa has per_100g_nutrients baseline", "per_100g_nutrients" in samosa and samosa["per_100g_nutrients"]["calories_kcal"] > 0)
    record("lookup: samosa reference serving is 65g", samosa["reference_serving_g"] == 65.0)
    samosa_keema = lookup_food("samosa", subtype_id="samosa_keema")
    record("lookup: resolves specific subtype 'samosa_keema'", samosa_keema is not None and samosa_keema.get("selected_subtype") == "samosa_keema")
    record("lookup: keema samosa calories differ from default potato", samosa_keema["reference_nutrients"]["calories_kcal"] == 318.0)
    aloo_gobi = lookup_food("aloo_gobi")
    record("lookup: finds composite recipe 'aloo_gobi'", aloo_gobi is not None and aloo_gobi["match_type"] == "COMPOSITE_RECIPE")
    chai = lookup_food("hot tea (garam chai)")
    record("lookup: finds item from unified 1,556 dataset", chai is not None and "tea" in chai["display_name"].lower())
    unknown = lookup_food("non_existent_dish_xyz_999")
    record("lookup: non-existent dish returns None", unknown is None)

    # 2. Portion Conversion
    print("\n--- 2. Portion Conversion & Defined Serving Units ---")
    record("convert: direct grams 150g -> 150.0g", convert_portion_to_grams(150, "g") == 150.0)
    record("convert: string grams '175.5' -> 175.5g", convert_portion_to_grams("175.5", "g") == 175.5)
    record("convert: baseline '100g' unit -> 100.0g", convert_portion_to_grams(1.0, "100g") == 100.0)
    g_samosa_1pc = convert_portion_to_grams(1, "piece", food_name="samosa")
    record("convert: 1 piece of samosa = 65.0g", g_samosa_1pc == 65.0, f"g={g_samosa_1pc}")
    g_samosa_2pc = convert_portion_to_grams(1, "piece_2", food_name="samosa")
    record("convert: 2 pieces of samosa = 130.0g", g_samosa_2pc == 130.0, f"g={g_samosa_2pc}")
    g_samosa_half = convert_portion_to_grams(1, "piece_half", food_name="samosa")
    record("convert: 0.5 piece of samosa = 32.5g", g_samosa_half == 32.5, f"g={g_samosa_half}")
    g_biryani_plate = convert_portion_to_grams(1, "plate", food_name="biryani")
    record("convert: 1 plate of biryani = 250.0g", g_biryani_plate == 250.0, f"g={g_biryani_plate}")
    g_dal_katori = convert_portion_to_grams(1, "katori", food_name="dal")
    record("convert: 1 katori of dal = 150.0g", g_dal_katori == 150.0, f"g={g_dal_katori}")
    g_idli_pc = convert_portion_to_grams(1, "piece", food_name="idli")
    record("convert: 1 piece of idli = 80.0g", g_idli_pc == 80.0, f"g={g_idli_pc}")
    g_chapati_pc = convert_portion_to_grams(1, "piece", food_name="chapati")
    record("convert: 1 chapati = 35.0g", g_chapati_pc == 35.0, f"g={g_chapati_pc}")
    servings = get_available_servings("samosa")
    record("servings: get_available_servings returns >= 3 options", len(servings) >= 3)
    record("servings: contains piece option for samosa", any("piece" in s["unit"] for s in servings))

    # 3. Invalid Inputs
    print("\n--- 3. Validation & Invalid Input Error Handling ---")
    for label, fn in [
        ("rejects negative portion value (-50g)", lambda: convert_portion_to_grams(-50, "g")),
        ("rejects zero portion value", lambda: convert_portion_to_grams(0, "g")),
        ("rejects NaN portion value", lambda: convert_portion_to_grams(float("nan"), "g")),
        ("rejects non-numeric string portion", lambda: convert_portion_to_grams("not_a_number", "g")),
        ("rejects unsupported portion unit", lambda: convert_portion_to_grams(100, "unsupported_unit_xyz")),
    ]:
        caught = False
        try:
            fn()
        except (ValueError, TypeError):
            caught = True
        record(f"error_handling: {label}", caught)

    calc_unknown_caught = False
    try:
        calculate_nutrients("imaginary_food_unknown", 100.0)
    except KeyError:
        calc_unknown_caught = True
    record("error_handling: calculate_nutrients raises KeyError for unknown food", calc_unknown_caught)

    # 4. Standard Formula
    print("\n--- 4. Standard Nutrient Scaling Formula Execution ---")
    calc_100 = calculate_nutrients("samosa", 100.0)
    calc_200 = calculate_nutrients("samosa", 200.0)
    calc_65 = calculate_nutrients("samosa", 65.0)
    per_100_cal = samosa["per_100g_nutrients"]["calories_kcal"]
    val_100_cal = calc_100["nutrients"]["calories_kcal"]["value"]
    record("formula: 100g portion matches per_100g baseline", abs(val_100_cal - per_100_cal) < 0.1, f"calc={val_100_cal}, base={per_100_cal}")
    val_200_cal = calc_200["nutrients"]["calories_kcal"]["value"]
    record("formula: 200g portion is exactly 2.0x 100g baseline", abs(val_200_cal - 2.0 * per_100_cal) < 0.2, f"val={val_200_cal}")
    val_65_cal = calc_65["nutrients"]["calories_kcal"]["value"]
    expected_65 = round(0.65 * per_100_cal, 2)
    record("formula: 65g portion follows portion/100 * per_100g", abs(val_65_cal - expected_65) < 0.1, f"calc={val_65_cal}, expected={expected_65}")
    record("nutrients: protein calculated with unit 'g'", calc_100["nutrients"]["protein_g"]["unit"] == "g" and calc_100["nutrients"]["protein_g"]["value"] > 0)
    record("nutrients: carbs calculated with unit 'g'", calc_100["nutrients"]["carbs_g"]["unit"] == "g")
    record("nutrients: fat calculated with unit 'g'", calc_100["nutrients"]["fat_g"]["unit"] == "g")
    record("nutrients: fibre calculated with unit 'g'", calc_100["nutrients"]["fibre_g"]["unit"] == "g")
    record("nutrients: sodium calculated with unit 'mg'", calc_100["nutrients"]["sodium_mg"]["unit"] == "mg")

    # 5. Missing Data
    print("\n--- 5. Missing Nutrition Data Integrity ---")
    partial_food = {
        "food_id": "test_partial",
        "class_name": "test_partial",
        "display_name": "Test Partial Item",
        "serving_size_g": 100.0,
        "serving_unit": "100g",
        "calories_kcal": 200.0,
        "protein_g": 5.0,
        "carbs_g": 30.0,
        "fat_g": 4.0,
        "fibre_g": 2.0,
        "sodium_mg": 150.0,
        "calcium_mg": None,
        "iron_mg": None,
        "vitamin_c_mg": None,
        "folate_ug": None,
        "data_source": "TEST_SOURCE",
    }
    calc_partial = calculate_nutrients(partial_food, 150.0)
    record("missing_data: missing calcium remains None, NOT 0", calc_partial["nutrients"]["calcium_mg"]["value"] is None)
    record("missing_data: missing iron remains None, NOT 0", calc_partial["nutrients"]["iron_mg"]["value"] is None)
    record("missing_data: status is DATA_NOT_AVAILABLE", calc_partial["nutrients"]["calcium_mg"]["status"] == "DATA_NOT_AVAILABLE")
    record("missing_data: reported in missing_nutrients list", any(m["nutrient"] == "calcium_mg" for m in calc_partial["missing_nutrients"]))
    record("missing_data: missing nutrients has descriptive reason", "TEST_SOURCE" in calc_partial["missing_nutrients"][0]["reason"])

    # 6. Source Transparency
    print("\n--- 6. Source Transparency & Non-Claim Disclaimer ---")
    record("metadata: source is reported", bool(calc_100.get("source")))
    record("metadata: reference basis is reported", bool(calc_100.get("reference_basis")))
    record("metadata: quality flags present (sodium_flag, calorie_flag)", "sodium_flag" in calc_100.get("data_quality_flags", {}))
    record("disclaimer: disclaimer is non-empty", len(calc_100.get("disclaimer", "")) > 20)
    record("disclaimer: explicitly states portion weight must be confirmed by user", "portion weight must be confirmed" in calc_100["disclaimer"].lower())
    record("disclaimer: explicitly states category only from vision", "category only" in calc_100["disclaimer"].lower())

    # 7. Manual Search Fallback
    print("\n--- 7. Manual Food Search Fallback ---")
    search_samosa = search_foods("samosa")
    record("search: 'samosa' returns matches", len(search_samosa) > 0)
    record("search: top match is samosa", "samosa" in search_samosa[0]["food_name"].lower())
    search_chai = search_foods("chai")
    record("search: 'chai' returns matches across unified lookup", len(search_chai) > 0)
    search_roti = search_foods("roti")
    record("search: 'roti' returns matches", len(search_roti) > 0)
    search_empty = search_foods("")
    record("search: empty query returns empty list", len(search_empty) == 0)

    # 8. Web API Endpoints
    print("\n--- 8. Web API Endpoints ---")
    flask_app.config["TESTING"] = True
    flask_client = flask_app.test_client()

    resp_s = flask_client.get("/api/nutrition/search?q=biryani")
    record("api_search: returns 200 OK", resp_s.status_code == 200)
    s_json = resp_s.get_json()
    record("api_search: success is True", s_json.get("success") is True)
    record("api_search: returns matches", s_json.get("count", 0) > 0)

    resp_serv = flask_client.get("/api/nutrition/servings?food_name=samosa")
    record("api_servings: returns 200 OK", resp_serv.status_code == 200)
    serv_json = resp_serv.get_json()
    record("api_servings: contains servings list", len(serv_json.get("servings", [])) >= 2)

    resp_an = flask_client.post(
        "/api/nutrition/analyze",
        json={"food_name": "samosa", "portion_value": 2, "unit": "piece", "user_profile": "adult_male_sedentary"}
    )
    record("api_analyze: defined serving 2 pieces returns 200 OK", resp_an.status_code == 200)
    an_json = resp_an.get_json()
    record("api_analyze: success is True", an_json.get("success") is True)
    an_data = an_json.get("data", {})
    record("api_analyze: calculated portion is 130g", an_data.get("portion", {}).get("grams") == 130.0)
    record("api_analyze: calories calculated for 130g", an_data.get("nutrients", {}).get("calories_kcal", {}).get("value") > 0)
    record("api_analyze: health_evaluation present with suitability_score", "suitability_score" in an_data.get("health_evaluation", {}))

    resp_err1 = flask_client.post("/api/nutrition/analyze", json={})
    record("api_analyze: missing food_name returns 400", resp_err1.status_code == 400)
    resp_err2 = flask_client.post("/api/nutrition/analyze", json={"food_name": "samosa", "portion_value": -10})
    record("api_analyze: negative portion returns 400", resp_err2.status_code == 400)
    resp_err3 = flask_client.post("/api/nutrition/analyze", json={"food_name": "non_existent_food_xyz"})
    record("api_analyze: unknown food returns 404", resp_err3.status_code == 404)

    # Summary
    total = len(test_results)
    passed = sum(1 for _, ok in test_results if ok)
    failed = total - passed

    print("\n" + "=" * 65)
    print(f"PHASE 7 NUTRITION ANALYSIS TESTS: {passed} PASSED / {failed} FAILED / {total} TOTAL")
    print("=" * 65)

    if failed > 0:
        raise AssertionError(f"{failed} test(s) failed in test_nutrition_analysis.py")


if __name__ == "__main__":
    run_tests()
