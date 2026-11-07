"""
tests/test_label_scanner.py
Comprehensive Integration & Unit Tests for the Packaged Food Label Scanner

Covers:
1. Regex Parsing & Unit Extraction (Calories, Macros, Micros, Serving Basis)
2. Sodium vs. Salt Separation (No automatic conversion, units preserved)
3. Missing Field Detection (Reports None/unavailable, never invents zero)
4. Allergen & Ingredient Extraction (Declared vs. May Contain)
5. Readability & Confidence Scoring (Good, Partial, Low, Unreadable)
6. Blank / Degraded Image Handling (Graceful degradation, no fabrication)
7. Flask API: /api/scan-label (Multipart & Base64 JSON)
8. Flask API: /api/confirm-label (User edits, ICMR RDA scoring, alerts)
9. Medical / Educational Disclaimers & Non-Clinical Framing
"""

import base64
import io
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from app import app
from src.label_scanner import (
    _normalize_text,
    _find_nutrient,
    _extract_nutrition,
    _extract_serving,
    _extract_ingredients,
    _extract_allergens,
    _summarise_missing,
    _assess_readability,
    scan_label,
)

PASS = "[PASS]"
FAIL = "[FAIL]"
test_results = []


def record(name: str, condition: bool, extra: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if extra:
        msg += f" | {extra}"
    print(msg)
    test_results.append((name, condition))


def create_mock_label_image(text_lines):
    """Generates a synthetic crisp image containing printed label text for OCR testing."""
    img = Image.new("RGB", (600, 500), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = 20
    for line in text_lines:
        draw.text((20, y), line, fill=(0, 0, 0))
        y += 24
    return img


def run_tests():
    print("=" * 65)
    print("RUNNING PACKAGED FOOD LABEL SCANNER TESTS")
    print("=" * 65)

    # =========================================================================
    # 1. Text Normalization & Regex Pattern Extraction
    # =========================================================================
    print("\n--- 1. Text Normalization & Robust Numeric Pattern Matching ---")
    norm1 = _normalize_text("Energy: 250 kcal | Protein: 8.5g • Salt: 1.2g")
    record("norm: replaces unicode bullet and pipe", "•" not in norm1 and "|" not in norm1)

    # =========================================================================
    # 2. Nutrition Fact Regex Extraction
    # =========================================================================
    print("\n--- 2. Nutrition Fact Regex Extraction ---")
    mock_nutrition_text = """
    NUTRITION FACTS
    Per 100g
    Energy: 450 kcal
    Protein: 7.2 g
    Total Carbohydrate: 65.0 g
    Dietary Fibre: 4.5 g
    Total Sugars: 22.0 g
    Added Sugars: 18.0 g
    Total Fat: 18.5 g
    Saturated Fat: 6.2 g
    Trans Fat: 0.0 g
    Cholesterol: 12 mg
    Sodium: 380 mg
    Salt: 0.95 g
    Calcium: 120 mg
    Iron: 2.8 mg
    Potassium: 210 mg
    Vitamin D: 3.5 mcg
    """

    nut = _extract_nutrition(mock_nutrition_text)
    record("extract_nut: calories extracted correctly", nut["calories"] is not None and nut["calories"]["value"] == "450" and nut["calories"]["unit"] == "kcal")
    record("extract_nut: protein extracted correctly", nut["protein"] is not None and nut["protein"]["value"] == "7.2" and nut["protein"]["unit"] == "g")
    record("extract_nut: carbs extracted correctly", nut["total_carbohydrates"] is not None and nut["total_carbohydrates"]["value"] == "65.0")
    record("extract_nut: fiber extracted correctly", nut["dietary_fibre"] is not None and nut["dietary_fibre"]["value"] == "4.5")
    record("extract_nut: added sugars extracted correctly", nut["added_sugars"] is not None and nut["added_sugars"]["value"] == "18.0")
    record("extract_nut: trans fat extracted as 0.0", nut["trans_fat"] is not None and nut["trans_fat"]["value"] == "0.0")
    record("extract_nut: sodium extracted as 380 mg", nut["sodium"] is not None and nut["sodium"]["value"] == "380" and nut["sodium"]["unit"] == "mg")
    record("extract_nut: salt extracted as 0.95 g", nut["salt"] is not None and nut["salt"]["value"] == "0.95" and nut["salt"]["unit"] == "g")

    # =========================================================================
    # 3. Sodium vs Salt Separation & Unit Integrity
    # =========================================================================
    print("\n--- 3. Sodium vs Salt Separation ---")
    salt_only_text = """
    Nutrition Information
    Energy: 300 kcal
    Salt: 1.5 g
    """
    nut_salt_only = _extract_nutrition(salt_only_text)
    record("salt_separation: salt is extracted when present", nut_salt_only["salt"] is not None and nut_salt_only["salt"]["value"] == "1.5")
    record(
        "salt_separation: sodium remains None when only salt declared (NO auto-conversion)",
        nut_salt_only["sodium"] is None,
        f"sodium={nut_salt_only['sodium']}"
    )

    # =========================================================================
    # 4. Serving Size & Nutrition Basis
    # =========================================================================
    print("\n--- 4. Serving Size & Nutrition Basis Extraction ---")
    serv_text = "Serving Size: 30 g. Servings per container: 10. Nutrition Information per 100 g."
    serv = _extract_serving(serv_text)
    record("serving: serving size 30 extracted", serv["serving_size"] == "30" and serv["serving_unit"] == "g")
    record("serving: servings per container is 10", serv["servings_per_container"] == "10")
    record("serving: nutrition basis is per 100 g", "100" in (serv["nutrition_basis"] or ""))

    # =========================================================================
    # 5. Ingredient & Allergen Parsing
    # =========================================================================
    print("\n--- 5. Ingredients & Allergen Extraction ---")
    ing_text = "Ingredients: Wheat Flour (45%), Palm Oil, Sugar, Cocoa Solids, Emulsifier (INS 322), Salt."
    ing = _extract_ingredients(ing_text)
    record("ingredients: extracts ingredient string", ing is not None and "Wheat Flour" in ing and "Cocoa Solids" in ing)

    allergen_text = "Allergy Advice: Contains Wheat, Soy. May contain trace of Peanuts."
    alg = _extract_allergens(allergen_text, ing_text)
    record("allergens: declared contains wheat and soy", "wheat" in alg["declared"] and "soy" in alg["declared"])
    record("allergens: may contain includes peanuts", alg["may_contain"] is not None and any("peanut" in m.lower() for m in alg["may_contain"]))

    # =========================================================================
    # 6. Missing Fields Detection (Never Invent Zeroes)
    # =========================================================================
    print("\n--- 6. Missing Fields Detection ---")
    partial_nut = {"calories": {"value": "200", "unit": "kcal"}, "protein": None, "sodium": None}
    missing = _summarise_missing(
        partial_nut,
        {"serving_size": None, "nutrition_basis": None},
        {"product_name": None, "brand": None},
        None,
        {"declared": None, "may_contain": None}
    )
    record("missing: detects protein missing", "nutrition.protein" in missing)
    record("missing: detects sodium missing", "nutrition.sodium" in missing)
    record("missing: detects serving size missing", "serving_size" in missing)
    record("missing: detects ingredients missing", "ingredients" in missing)
    record("missing: does NOT include calories (since present)", "nutrition.calories" not in missing)

    # =========================================================================
    # 7. Blank & Degraded Image Handling
    # =========================================================================
    print("\n--- 7. Blank & Degraded Image Handling ---")
    blank_img = Image.new("RGB", (100, 100), color=(0, 0, 0))  # Pure black image
    blank_res = scan_label(blank_img)
    record("blank_img: returns structured dict without exception", isinstance(blank_res, dict))
    record("blank_img: quality is UNREADABLE or LOW", blank_res["readability"]["quality"] in ["UNREADABLE", "LOW"])
    record("blank_img: missing fields reported", len(blank_res["missing_fields"]) > 0)
    record("blank_img: disclaimer is present", "disclaimer" in blank_res and len(blank_res["disclaimer"]) > 10)
    record("blank_img: calories is None, NOT 0", blank_res["nutrition"]["calories"] is None)

    # =========================================================================
    # 8. Web API Tests: /api/scan-label
    # =========================================================================
    print("\n--- 8. Web API: /api/scan-label ---")
    app.config["TESTING"] = True
    client = app.test_client()

    # Missing file returns 400
    resp_empty = client.post("/api/scan-label")
    record("api_scan: empty request returns 400", resp_empty.status_code == 400)

    # Upload test image via multipart
    test_label_img = create_mock_label_image([
        "BRITANNIA GOOD DAY COOKIES",
        "Serving Size: 25 g",
        "Energy: 120 kcal",
        "Protein: 1.8 g",
        "Carbohydrate: 17.5 g",
        "Total Sugars: 6.5 g",
        "Total Fat: 5.0 g",
        "Sodium: 75 mg",
        "Ingredients: Wheat Flour, Sugar, Butter, Milk Solids.",
        "Contains: Wheat, Milk."
    ])

    buf = io.BytesIO()
    test_label_img.save(buf, format="JPEG")
    buf.seek(0)

    resp_scan = client.post(
        "/api/scan-label",
        data={"file": (buf, "label_test.jpg")},
        content_type="multipart/form-data"
    )
    record("api_scan: multipart returns 200 OK", resp_scan.status_code == 200)
    scan_json = resp_scan.get_json()
    record("api_scan: response success is True", scan_json.get("success") is True)
    scan_data = scan_json.get("data", {})
    record("api_scan: data contains ocr_text", "ocr_text" in scan_data)
    record("api_scan: data contains readability", "readability" in scan_data)
    record("api_scan: data contains nutrition dict", "nutrition" in scan_data)
    record("api_scan: data contains disclaimer", "disclaimer" in scan_data)

    # Base64 upload test
    buf2 = io.BytesIO()
    test_label_img.save(buf2, format="JPEG")
    buf2.seek(0)
    b64_str = base64.b64encode(buf2.read()).decode("utf-8")
    resp_b64 = client.post(
        "/api/scan-label",
        json={"image_base64": f"data:image/jpeg;base64,{b64_str}"}
    )
    record("api_scan: base64 JSON returns 200 OK", resp_b64.status_code == 200)

    # =========================================================================
    # 9. Web API: /api/confirm-label (User Corrections & Health Scoring)
    # =========================================================================
    print("\n--- 9. Web API: /api/confirm-label ---")

    # Confirm extracted values without correction
    confirm_payload = {
        "user_profile": "adult_male_sedentary",
        "nutrition": {
            "calories": {"value": 120.0, "unit": "kcal", "ocr_extracted": True},
            "protein": {"value": 1.8, "unit": "g", "ocr_extracted": True},
            "total_carbohydrates": {"value": 17.5, "unit": "g", "ocr_extracted": True},
            "total_sugars": {"value": 6.5, "unit": "g", "ocr_extracted": True},
            "total_fat": {"value": 5.0, "unit": "g", "ocr_extracted": True},
            "sodium": {"value": 75.0, "unit": "mg", "ocr_extracted": True},
            "salt": None
        },
        "user_corrections": {}
    }

    resp_confirm = client.post("/api/confirm-label", json=confirm_payload)
    record("api_confirm: returns 200 OK", resp_confirm.status_code == 200)
    confirm_data = resp_confirm.get_json().get("data", {})
    record("api_confirm: contains health_evaluation", "health_evaluation" in confirm_data)
    he = confirm_data.get("health_evaluation", {})
    record("api_confirm: suitability_score is numeric between 0 and 10", 0.0 <= he.get("suitability_score", -1) <= 10.0)
    record("api_confirm: verdict is present", bool(he.get("verdict")))
    record("api_confirm: contains non-clinical disclaimer", "clinical" in confirm_data.get("disclaimer", "").lower() or "medical" in confirm_data.get("disclaimer", "").lower())

    # User correction test: user edits sugar from 6.5 to 25.0 g
    confirm_corrected_payload = {
        "user_profile": "diabetic_adult",
        "nutrition": {
            "calories": {"value": 250.0, "unit": "kcal", "ocr_extracted": True},
            "protein": {"value": 2.0, "unit": "g", "ocr_extracted": True},
            "total_carbohydrates": {"value": 45.0, "unit": "g", "ocr_extracted": True},
            "total_sugars": {"value": 25.0, "unit": "g", "ocr_extracted": False},
            "total_fat": {"value": 8.0, "unit": "g", "ocr_extracted": True},
            "sodium": {"value": 50.0, "unit": "mg", "ocr_extracted": True},
            "salt": None
        },
        "user_corrections": {
            "total_sugars": {"original": 6.5, "user_value": 25.0, "is_corrected": True}
        }
    }

    resp_corr = client.post("/api/confirm-label", json=confirm_corrected_payload)
    corr_data = resp_corr.get_json().get("data", {})
    record("api_confirm: user corrections tracked", "total_sugars" in corr_data.get("user_corrections", {}))
    corr_alerts = corr_data.get("health_evaluation", {}).get("dietary_alerts", [])
    has_sugar_alert = any("sugar" in a.get("message", "").lower() or "carb" in a.get("message", "").lower() for a in corr_alerts)
    record("api_confirm: high sugar triggers dietary alert for diabetic profile", has_sugar_alert)

    # Salt declared but no sodium -> verifies SALT_NOT_CONVERTED warning
    salt_warn_payload = {
        "user_profile": "hypertensive_adult",
        "nutrition": {
            "calories": {"value": 150.0, "unit": "kcal", "ocr_extracted": True},
            "protein": {"value": 3.0, "unit": "g", "ocr_extracted": True},
            "total_carbohydrates": {"value": 20.0, "unit": "g", "ocr_extracted": True},
            "sodium": None,  # Not declared
            "salt": {"value": 2.2, "unit": "g", "ocr_extracted": True}
        },
        "user_corrections": {}
    }
    resp_salt = client.post("/api/confirm-label", json=salt_warn_payload)
    salt_alerts = resp_salt.get_json().get("data", {}).get("health_evaluation", {}).get("dietary_alerts", [])
    has_salt_info = any(a.get("type") == "SALT_NOT_CONVERTED" for a in salt_alerts)
    record("api_confirm: flags SALT_NOT_CONVERTED without inventing sodium", has_salt_info)

    # Summary
    total_tests = len(test_results)
    passed_tests = sum(1 for _, ok in test_results if ok)
    failed_tests = total_tests - passed_tests
    print("\n" + "=" * 65)
    print(f"PACKAGED FOOD LABEL SCANNER TESTS: {passed_tests} PASSED / {failed_tests} FAILED / {total_tests} TOTAL")
    print("=" * 65)

    if failed_tests > 0:
        raise AssertionError(f"{failed_tests} test(s) failed in test_label_scanner.py")


if __name__ == "__main__":
    run_tests()
