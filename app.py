"""
app.py
NutriSense AI - Phase 6 Web Application Server

Flask backend integrating:
- MobileNetV2 43-Class Food Recognition Inference
- Master Nutrition Knowledge Base Lookup
- Interactive Ambiguous Subtype Disambiguation
- Composite Recipe Ingredient Explorer
- ICMR-NIN 2020 RDA Clinical Health Suitability Scoring
- Dynamic Portion Scaling & Allergen Alerts
- Transparent Data Quality Flags & Medical Disclaimer
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import io
import json
from pathlib import Path
import sys
from typing import Dict, Any, Optional, List, Tuple, Union

from flask import Flask, jsonify, render_template, request, send_from_directory
import numpy as np
import pandas as pd
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.inference_pipeline import NutriSenseInferenceEngine, ALLERGEN_MAP
from src.nutrient_scoring_engine import compute_suitability_score, ICMR_RDA_PROFILES
from src.label_scanner import scan_label
from src.nutrition_analyzer import (
    get_analyzer,
    calculate_nutrients,
    convert_portion_to_grams,
    search_foods,
    lookup_food,
    get_available_servings,
    PHASE7_DISCLAIMER,
)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max image upload

# Initialize Inference Engine
print("=" * 65)
print("INITIALIZING NUTRISENSE AI INFERENCE ENGINE...")
print("=" * 65)
engine = NutriSenseInferenceEngine()
print("ENGINE READY FOR WEB SERVING!")

SAMPLES_DIR = WORKSPACE_ROOT / "static" / "samples"
MODELS_DIR = WORKSPACE_ROOT / "models"


@app.route("/")
def index():
    """Serves the main single-page web application."""
    return render_template("index.html")


@app.route("/reports/figures/<path:filename>")
def serve_report_figure(filename):
    """Serves generated benchmark charts and figures."""
    figures_dir = WORKSPACE_ROOT / "reports" / "figures"
    return send_from_directory(figures_dir, filename)


@app.route("/api/health", methods=["GET"])
def health():
    """Returns system status, active model info, and hardware context."""
    return jsonify({
        "status": "healthy",
        "service": "NutriSense AI Web Application",
        "model_architecture": "MobileNetV2 (ImageNet-1K pretrained + 43-class fine-tuned)",
        "classes_count": len(engine.class_to_idx),
        "device": str(engine.device),
        "is_trained": engine.is_trained,
        "checkpoint_path": str(engine.checkpoint_path),
        "disclaimer": "NutriSense AI provides evidence-based nutritional estimations based on ICMR-NIN standards. This information is intended for educational and dietary awareness and does not constitute clinical medical advice."
    })


@app.route("/api/profiles", methods=["GET"])
def get_profiles():
    """Returns available ICMR-NIN 2020 RDA user health profiles."""
    profiles_meta = []
    for key, p in ICMR_RDA_PROFILES.items():
        profiles_meta.append({
            "key": key,
            "title": key.replace("_", " ").title(),
            "description": p.get("description", ""),
            "calories_target": p.get("calories_kcal", 2000),
            "protein_target_g": p.get("protein_g", 50),
            "carbs_target_g": p.get("carbs_g", 250),
            "fat_target_g": p.get("fat_g", 25),
            "fibre_target_g": p.get("fibre_g", 30),
            "sodium_max_mg": p.get("sodium_mg_max", 2000),
            "clinical_condition": p.get("clinical_condition", "GENERAL_WELLNESS")
        })
    return jsonify(profiles_meta)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Returns list of curated test food samples with thumbnails."""
    sample_items = [
        {"class_name": "samosa", "title": "Samosa", "description": "Crispy fried pastry (Ambiguous: Aloo / Keema / Paneer)", "url": "/static/samples/samosa.jpg"},
        {"class_name": "biryani", "title": "Biryani", "description": "Fragrant spiced rice layered dish (Ambiguous)", "url": "/static/samples/biryani.jpg"},
        {"class_name": "palak_paneer", "title": "Palak Paneer", "description": "Cottage cheese in spiced spinach gravy (Dairy)", "url": "/static/samples/palak_paneer.jpg"},
        {"class_name": "jalebi", "title": "Jalebi", "description": "Crispy pretzel-shaped sweet soaked in saffron syrup", "url": "/static/samples/jalebi.jpg"},
        {"class_name": "idli", "title": "Idli", "description": "Steamed fermented rice and lentil cakes (Healthy)", "url": "/static/samples/idli.jpg"},
        {"class_name": "aloo_gobi", "title": "Aloo Gobi", "description": "Potato and cauliflower stir-fry (Composite Recipe)", "url": "/static/samples/aloo_gobi.jpg"},
        {"class_name": "chicken_curry", "title": "Chicken Curry", "description": "Tender chicken simmered in onion-tomato gravy (Non-Veg)", "url": "/static/samples/chicken_curry.jpg"},
        {"class_name": "momos", "title": "Momos", "description": "Steamed Himalayan dumplings (Ambiguous filling)", "url": "/static/samples/momos.jpg"},
        {"class_name": "modak", "title": "Modak", "description": "Steamed sweet dumplings with coconut-jaggery (Nuts)", "url": "/static/samples/modak.jpg"},
        {"class_name": "dhokla", "title": "Dhokla", "description": "Steamed savory fermented gram flour snack", "url": "/static/samples/dhokla.jpg"},
    ]
    return jsonify(sample_items)


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Returns Phase 5 model evaluation metrics and benchmark stats."""
    eval_file = MODELS_DIR / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Evaluation results not found"}), 404


@app.route("/api/food-catalog", methods=["GET"])
def get_food_catalog():
    """Returns complete catalog of 43 foods with nutrition and flags."""
    foods = []
    for c, row in engine.master_kb_map.items():
        foods.append({
            "class_name": c,
            "display_name": row.get("display_name", c),
            "match_type": row.get("match_type", "VALID_DIRECT"),
            "serving_size_g": row.get("serving_size_g"),
            "serving_unit": row.get("serving_unit"),
            "calories_kcal": row.get("calories_kcal"),
            "protein_g": row.get("protein_g"),
            "fat_g": row.get("fat_g"),
            "carbs_g": row.get("carbs_g"),
            "sodium_mg": row.get("sodium_mg"),
            "sodium_flag": row.get("sodium_flag", "ok"),
            "calorie_flag": row.get("calorie_flag", "ok"),
            "allergens": ALLERGEN_MAP.get(c, []),
            "has_subtypes": bool(row.get("has_subtypes", False))
        })
    return jsonify(sorted(foods, key=lambda x: x["display_name"]))


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receives food image via file upload or base64 string, runs vision model,
    and returns top predictions, ambiguous subtypes, scaled nutrients,
    clinical health scores, and allergen alerts.
    """
    try:
        user_profile = request.form.get("user_profile") or request.args.get("user_profile") or "adult_male_sedentary"
        portion_g_raw = request.form.get("portion_g") or request.args.get("portion_g")
        portion_g = float(portion_g_raw) if portion_g_raw else None
        selected_subtype_id = request.form.get("selected_subtype_id") or request.args.get("selected_subtype_id")

        pil_image = None

        # Check multipart file upload
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "":
                pil_image = Image.open(file.stream).convert("RGB")

        # Check JSON payload (e.g. base64 or sample path)
        if pil_image is None and request.is_json:
            json_data = request.get_json()
            if not json_data:
                return jsonify({"error": "Empty JSON payload"}), 400

            user_profile = json_data.get("user_profile", user_profile)
            portion_g = json_data.get("portion_g", portion_g)
            selected_subtype_id = json_data.get("selected_subtype_id", selected_subtype_id)

            # Sample path reference
            if "sample_path" in json_data:
                sample_path = WORKSPACE_ROOT / json_data["sample_path"].lstrip("/")
                if sample_path.exists():
                    pil_image = Image.open(sample_path).convert("RGB")

            # Base64 string
            elif "image_base64" in json_data:
                b64_data = json_data["image_base64"]
                if "," in b64_data:
                    b64_data = b64_data.split(",")[1]
                img_bytes = base64.b64decode(b64_data)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        if pil_image is None:
            return jsonify({"error": "No valid image provided via 'file' or 'image_base64'"}), 400

        # Execute full analysis pipeline
        result = engine.analyze_food(
            pil_image,
            top_k=5,
            user_profile=user_profile,
            portion_g=portion_g,
            selected_subtype_id=selected_subtype_id
        )

        return jsonify({
            "success": True,
            "data": result,
            "disclaimer": "Nutritional estimations are for informational and lifestyle awareness. Not clinical medical advice."
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/adjust-nutrition", methods=["POST"])
def adjust_nutrition():
    """
    Recalculates scaled nutrition, health suitability score, and alerts
    when user modifies portion size or selects a specific subtype,
    without requiring vision re-inference.
    """
    try:
        data = request.get_json() or {}
        class_name = data.get("class_name")
        if not class_name or class_name not in engine.master_kb_map:
            return jsonify({"error": f"Invalid or missing class_name: {class_name}"}), 400

        user_profile = data.get("user_profile", "adult_male_sedentary")
        portion_g = data.get("portion_g")
        selected_subtype_id = data.get("selected_subtype_id")

        # Create dummy image tensor to leverage engine.analyze_food with exact subtype
        # Or run targeted lookup directly:
        base_kb = engine.master_kb_map[class_name].copy()
        chosen_profile = base_kb.copy()

        # Check subtypes
        subtype_info = None
        requires_confirmation = False

        if class_name in engine.subtypes_registry:
            family_meta = engine.subtypes_registry[class_name]
            subtypes_list = family_meta.get("subtypes", [])
            default_sub = next((s for s in subtypes_list if s.get("is_default")), subtypes_list[0])

            if selected_subtype_id:
                match_sub = next((s for s in subtypes_list if s.get("subtype_id") == selected_subtype_id), None)
                if match_sub:
                    chosen_profile.update(match_sub)
                    chosen_profile["selected_subtype"] = selected_subtype_id
                    requires_confirmation = False
                else:
                    chosen_profile.update(default_sub)
                    requires_confirmation = True
            else:
                chosen_profile.update(default_sub)
                requires_confirmation = True

            subtype_info = {
                "requires_confirmation": requires_confirmation,
                "disambiguation_prompt": family_meta.get("disambiguation_prompt", "Please confirm food subtype:"),
                "default_subtype": default_sub,
                "available_subtypes": [
                    {
                        "subtype_id": s["subtype_id"],
                        "display_name": s["display_name"],
                        "calories_kcal": s.get("calories_kcal"),
                        "protein_g": s.get("protein_g"),
                        "fat_g": s.get("fat_g"),
                        "carbs_g": s.get("carbs_g"),
                        "serving_size_g": s.get("serving_size_g"),
                        "serving_unit": s.get("serving_unit")
                    } for s in subtypes_list
                ]
            }

        # Portion scaling
        default_serving_g = float(chosen_profile.get("serving_size_g", 100.0))
        if pd.isna(default_serving_g) or default_serving_g <= 0:
            default_serving_g = engine.serving_weights_map.get(class_name, 100.0)

        actual_portion_g = float(portion_g) if portion_g is not None and float(portion_g) > 0 else default_serving_g
        scale = actual_portion_g / default_serving_g if default_serving_g > 0 else 1.0

        def _scale_val(val):
            if pd.isna(val) or val is None:
                return None
            return round(float(val) * scale, 2)

        scaled_nutrition = {
            "serving_size_g": actual_portion_g,
            "serving_unit": chosen_profile.get("serving_unit", "standard serving"),
            "portion_scale_factor": round(scale, 3),
            "calories_kcal": _scale_val(chosen_profile.get("calories_kcal")),
            "carbs_g": _scale_val(chosen_profile.get("carbs_g")),
            "protein_g": _scale_val(chosen_profile.get("protein_g")),
            "fat_g": _scale_val(chosen_profile.get("fat_g")),
            "fibre_g": _scale_val(chosen_profile.get("fibre_g")),
            "sodium_mg": _scale_val(chosen_profile.get("sodium_mg")),
            "calcium_mg": _scale_val(chosen_profile.get("calcium_mg")),
            "iron_mg": _scale_val(chosen_profile.get("iron_mg")),
            "vitamin_c_mg": _scale_val(chosen_profile.get("vitamin_c_mg")),
            "folate_ug": _scale_val(chosen_profile.get("folate_ug")),
        }

        # Missing micros
        missing_micros = [m for m in ["calcium_mg", "iron_mg", "vitamin_c_mg", "folate_ug"] if scaled_nutrition[m] is None or pd.isna(chosen_profile.get(m))]

        # Flags & health scoring
        sodium_flag = str(chosen_profile.get("sodium_flag", "ok"))
        calorie_flag = str(chosen_profile.get("calorie_flag", "ok"))

        score_input = {
            "calories_kcal": scaled_nutrition["calories_kcal"] or 0,
            "carbs_g": scaled_nutrition["carbs_g"] or 0,
            "protein_g": scaled_nutrition["protein_g"] or 0,
            "fat_g": scaled_nutrition["fat_g"] or 0,
            "fibre_g": scaled_nutrition["fibre_g"] or 1.0,
            "sodium_mg": scaled_nutrition["sodium_mg"] or 200.0,
            "sodium_flag": sodium_flag,
            "calorie_flag": calorie_flag,
            "display_name": chosen_profile.get("display_name", class_name),
            "class_name": class_name
        }
        health_score, verdict = compute_suitability_score(score_input, profile_key=user_profile)

        # Alerts
        alerts = []
        if sodium_flag == "high_sodium_unverified":
            alerts.append({
                "severity": "WARNING",
                "type": "SODIUM_DATA_LIMITATION",
                "message": f"Source sodium is extremely high ({chosen_profile.get('sodium_mg')} mg/serving) and unverified in source literature. Exercise caution."
            })
        if calorie_flag == "discrepant_atwater_over_25pct":
            alerts.append({
                "severity": "INFO",
                "type": "CALORIE_DISCREPANCY_FLAG",
                "message": "Dish has >25% discrepancy between reported calories and Atwater 4-4-9 sum in source literature."
            })
        if user_profile == "hypertensive_adult" and (scaled_nutrition["sodium_mg"] or 0) > 500.0:
            alerts.append({
                "severity": "HIGH",
                "type": "HYPERTENSION_WARNING",
                "message": "High sodium portion exceeds DASH recommendation for hypertensive individuals."
            })
        if user_profile == "diabetic_adult" and (scaled_nutrition["carbs_g"] or 0) > 35.0:
            alerts.append({
                "severity": "MODERATE",
                "type": "DIABETES_CARB_ALERT",
                "message": "High carbohydrate content may cause glycemic spike. Consider pairing with fiber/salad."
            })

        return jsonify({
            "success": True,
            "data": {
                "class_name": class_name,
                "display_name": chosen_profile.get("display_name", class_name),
                "nutrition": scaled_nutrition,
                "metadata": {
                    "match_type": chosen_profile.get("match_type", "VALID_DIRECT"),
                    "data_source": chosen_profile.get("data_source", "master_food_nutrition_kb.csv"),
                    "sodium_flag": sodium_flag,
                    "calorie_flag": calorie_flag,
                    "missing_micronutrients": missing_micros
                },
                "ambiguous_subtype": subtype_info,
                "health_evaluation": {
                    "user_profile": user_profile,
                    "suitability_score": health_score,
                    "verdict": verdict,
                    "dietary_alerts": alerts,
                    "detected_allergens": ALLERGEN_MAP.get(class_name, [])
                }
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Phase 7: Nutrition Analysis Module Endpoints
# =============================================================================

@app.route("/api/nutrition/search", methods=["GET"])
def search_nutrition_foods():
    """
    Manual food search fallback when image recognition fails or user searches directly.
    Searches across Master Food Nutrition KB and Unified Food Lookup (1,556 items).
    """
    try:
        query = request.args.get("q", "").strip()
        limit = int(request.args.get("limit", 15))
        if not query:
            return jsonify({"success": True, "count": 0, "data": []})

        results = search_foods(query, limit=limit)
        return jsonify({
            "success": True,
            "query": query,
            "count": len(results),
            "data": results
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/nutrition/servings", methods=["GET"])
def get_food_servings():
    """Returns list of defined serving options for a given food name or class."""
    try:
        food_name = request.args.get("food_name", "").strip()
        if not food_name:
            return jsonify({"error": "Missing 'food_name' query parameter"}), 400

        servings = get_available_servings(food_name)
        return jsonify({
            "success": True,
            "food_name": food_name,
            "servings": servings
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/nutrition/analyze", methods=["POST"])
def analyze_nutrition_item():
    """
    Phase 7 Core Endpoint: Calculates scaled nutrition using the per-100g standard formula:
      nutrient_amount = (portion_grams / 100.0) * nutrient_value_per_100g
    Handles portion conversion from grams or defined servings (piece, katori, plate, etc.).
    Preserves missing nutrients as null and reports data source & quality flags.
    """
    try:
        data = request.get_json() or {}
        food_name = data.get("food_name") or data.get("class_name")
        if not food_name:
            return jsonify({"error": "Missing 'food_name' in request body"}), 400

        portion_val = data.get("portion_value")
        if portion_val is None:
            portion_val = data.get("portion_g", 100.0)

        unit = data.get("unit", "g")
        subtype_id = data.get("subtype_id") or data.get("selected_subtype_id")
        user_profile = data.get("user_profile", "adult_male_sedentary")

        # 1. Convert portion to grams
        try:
            portion_grams = convert_portion_to_grams(portion_val, unit=unit, food_name=food_name)
        except ValueError as ve:
            return jsonify({"error": f"Invalid portion: {str(ve)}"}), 400

        # 2. Calculate scaled nutrients using standard per-100g formula
        try:
            analysis = calculate_nutrients(food_name, portion_grams, subtype_id=subtype_id)
        except KeyError as ke:
            return jsonify({"error": str(ke)}), 404

        # 3. ICMR-NIN RDA health evaluation
        nut = analysis["nutrients"]
        score_input = {
            "calories_kcal": nut["calories_kcal"]["value"] or 0,
            "carbs_g": nut["carbs_g"]["value"] or 0,
            "protein_g": nut["protein_g"]["value"] or 0,
            "fat_g": nut["fat_g"]["value"] or 0,
            "fibre_g": nut["fibre_g"]["value"] or 1.0,
            "sodium_mg": nut["sodium_mg"]["value"] or 200.0,
            "sodium_flag": analysis["data_quality_flags"]["sodium_flag"],
            "calorie_flag": analysis["data_quality_flags"]["calorie_flag"],
            "display_name": analysis["display_name"],
            "class_name": analysis["class_name"]
        }
        health_score, verdict = compute_suitability_score(score_input, profile_key=user_profile)

        # Generate dietary alerts
        alerts = []
        if analysis["data_quality_flags"]["sodium_flag"] == "high_sodium_unverified":
            alerts.append({
                "severity": "WARNING",
                "type": "SODIUM_DATA_LIMITATION",
                "message": "Source sodium is extremely high and unverified in laboratory literature. Exercise caution."
            })
        if analysis["data_quality_flags"]["calorie_flag"] == "discrepant_atwater_over_25pct":
            alerts.append({
                "severity": "INFO",
                "type": "CALORIE_DISCREPANCY_FLAG",
                "message": "Dish has >25% discrepancy between reported calories and Atwater 4-4-9 sum in source literature."
            })
        if user_profile == "hypertensive_adult" and (nut["sodium_mg"]["value"] or 0) > 500.0:
            alerts.append({
                "severity": "HIGH",
                "type": "HYPERTENSION_WARNING",
                "message": "Portion sodium exceeds DASH guideline for hypertensive individuals."
            })
        if user_profile == "diabetic_adult" and (nut["carbs_g"]["value"] or 0) > 35.0:
            alerts.append({
                "severity": "MODERATE",
                "type": "DIABETES_CARB_ALERT",
                "message": "High carbohydrate portion may cause glycemic spike. Pair with fiber/salad."
            })

        analysis["health_evaluation"] = {
            "user_profile": user_profile,
            "suitability_score": health_score,
            "verdict": verdict,
            "dietary_alerts": alerts,
            "detected_allergens": ALLERGEN_MAP.get(analysis["class_name"], [])
        }

        # Available defined servings for frontend dropdown
        analysis["available_servings"] = get_available_servings(food_name)

        return jsonify({"success": True, "data": analysis})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/scan-label", methods=["POST"])
def scan_packaged_label():
    """
    Accepts a product packaging photo via multipart file upload or base64 JSON.
    Runs EasyOCR and structured parsing, returning all extracted fields.
    Missing fields are reported as null — never invented or zero-filled.
    """
    try:
        pil_image = None

        if "file" in request.files:
            f = request.files["file"]
            if f and f.filename:
                pil_image = Image.open(f.stream).convert("RGB")

        if pil_image is None and request.is_json:
            json_data = request.get_json() or {}
            if "image_base64" in json_data:
                import base64, io as _io
                b64 = json_data["image_base64"]
                if "," in b64:
                    b64 = b64.split(",")[1]
                pil_image = Image.open(_io.BytesIO(base64.b64decode(b64))).convert("RGB")

        if pil_image is None:
            return jsonify({"success": False, "error": "No image provided (use 'file' multipart or 'image_base64' JSON)."}), 400

        result = scan_label(pil_image)
        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/confirm-label", methods=["POST"])
def confirm_packaged_label():
    """
    Accepts user-confirmed (possibly corrected) label data and evaluates
    ICMR-NIN 2020 RDA suitability for the active user health profile.
    Tracks which fields were user-corrected vs OCR-extracted.
    """
    try:
        data = request.get_json() or {}
        user_profile = data.get("user_profile", "adult_male_sedentary")
        nutrition_confirmed = data.get("nutrition", {})
        corrections = data.get("user_corrections", {})

        def _parse(field):
            """Parse numeric value from confirmed nutrition dict entry."""
            entry = nutrition_confirmed.get(field)
            if entry is None:
                return None
            if isinstance(entry, dict):
                try:
                    return float(entry.get("value", 0) or 0)
                except (TypeError, ValueError):
                    return None
            try:
                return float(entry)
            except (TypeError, ValueError):
                return None

        # Build scoring input from confirmed data
        cal = _parse("calories")
        protein = _parse("protein")
        carbs = _parse("total_carbohydrates")
        fat = _parse("total_fat")
        fibre = _parse("dietary_fibre")
        sodium = _parse("sodium")
        salt = _parse("salt")

        # Use sodium if present; if only salt declared, note conversion is NOT automatic
        effective_sodium = sodium
        sodium_source = "sodium"
        if effective_sodium is None and salt is not None:
            effective_sodium = None  # Do NOT auto-convert salt to sodium
            sodium_source = "salt_not_converted"

        health_score, verdict = None, None
        alerts = []

        if cal is not None:
            score_input = {
                "calories_kcal": cal or 0,
                "carbs_g": carbs or 0,
                "protein_g": protein or 0,
                "fat_g": fat or 0,
                "fibre_g": fibre or 1.0,
                "sodium_mg": effective_sodium or 200.0,
                "sodium_flag": "ok",
                "calorie_flag": "ok",
                "display_name": data.get("product", {}).get("product_name", "Packaged Product"),
                "class_name": "packaged_product"
            }
            health_score, verdict = compute_suitability_score(score_input, profile_key=user_profile)

            if user_profile == "hypertensive_adult" and effective_sodium and effective_sodium > 500:
                alerts.append({
                    "severity": "HIGH",
                    "type": "HYPERTENSION_WARNING",
                    "message": f"Declared sodium ({effective_sodium:.0f} mg/serving) exceeds DASH threshold for hypertensive individuals."
                })
            if user_profile == "diabetic_adult" and carbs and carbs > 35:
                alerts.append({
                    "severity": "MODERATE",
                    "type": "DIABETES_CARB_ALERT",
                    "message": "High carbohydrate content may cause glycemic spike. Review serving size carefully."
                })
            if salt is not None and sodium is None:
                alerts.append({
                    "severity": "INFO",
                    "type": "SALT_NOT_CONVERTED",
                    "message": (
                        f"Label declares salt ({salt} g/serving). Sodium not auto-converted from salt — "
                        "sodium value remains unavailable. Check product label for declared sodium."
                    )
                })

        return jsonify({
            "success": True,
            "data": {
                "user_profile": user_profile,
                "confirmed_nutrition": nutrition_confirmed,
                "user_corrections": corrections,
                "health_evaluation": {
                    "suitability_score": health_score,
                    "verdict": verdict,
                    "dietary_alerts": alerts,
                    "note": (
                        "Health score is based on user-confirmed nutritional values from packaging. "
                        "Not applicable if nutrition panel was not visible or data is incomplete."
                        if health_score is not None
                        else "Health score not computed — calorie value is unavailable."
                    )
                },
                "sodium_source": sodium_source,
                "disclaimer": (
                    "Nutritional assessment is based on values extracted or confirmed from the product label. "
                    "Always verify against the physical product. Not clinical medical advice."
                )
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



# =============================================================================
# Phase 8: Personalization and User Profiling Endpoints
# =============================================================================

from src.user_profile_manager import (
    create_profile,
    get_profile,
    update_profile,
    delete_profile,
    list_profiles,
    get_active_profile,
    set_active_profile,
    get_personalized_context,
    get_breakfast_suggestions,
    calculate_bmi,
    validate_profile_data,
    PHASE8_DISCLAIMER,
)


@app.route("/api/profiles/user", methods=["GET"])
def list_user_profiles():
    """Returns lightweight list of all user profiles."""
    try:
        profiles = list_profiles()
        active = get_active_profile()
        return jsonify({
            "success": True,
            "count": len(profiles),
            "active_profile_id": active["user_id"] if active else None,
            "profiles": profiles,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/user", methods=["POST"])
def create_user_profile():
    """
    Creates a new user profile.
    Required: display_name.
    Optional: age, gender, height_cm, weight_kg, dietary_goal, activity_level,
              dietary_preference, allergies, health_conditions, budget,
              max_cooking_time_minutes, avoid_ingredients, dietary_notes,
              morning_routine_preference, breakfast_preference.
    """
    try:
        data = request.get_json() or {}
        if not data.get("display_name"):
            return jsonify({"error": "'display_name' is required."}), 400

        profile = create_profile(data)
        return jsonify({
            "success": True,
            "message": f"Profile '{profile['display_name']}' created successfully.",
            "data": profile,
            "disclaimer": PHASE8_DISCLAIMER,
        }), 201

    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/user/<user_id>", methods=["GET"])
def get_user_profile(user_id):
    """Returns full profile for a given user_id."""
    try:
        profile = get_profile(user_id)
        if not profile:
            return jsonify({"error": f"Profile '{user_id}' not found."}), 404
        return jsonify({"success": True, "data": profile})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/user/<user_id>", methods=["PUT"])
def update_user_profile(user_id):
    """
    Updates an existing user profile (partial update supported).
    Only provided fields are changed; others remain unchanged.
    """
    try:
        data = request.get_json() or {}
        if not data:
            return jsonify({"error": "No fields provided for update."}), 400

        updated = update_profile(user_id, data)
        return jsonify({
            "success": True,
            "message": "Profile updated successfully.",
            "data": updated,
            "disclaimer": PHASE8_DISCLAIMER,
        })

    except KeyError as ke:
        return jsonify({"success": False, "error": str(ke)}), 404
    except ValueError as ve:
        return jsonify({"success": False, "error": str(ve)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/user/<user_id>", methods=["DELETE"])
def delete_user_profile(user_id):
    """Deletes a user profile by user_id."""
    try:
        deleted = delete_profile(user_id)
        if not deleted:
            return jsonify({"error": f"Profile '{user_id}' not found."}), 404
        return jsonify({"success": True, "message": "Profile deleted successfully."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/user/<user_id>/activate", methods=["POST"])
def activate_user_profile(user_id):
    """Sets a profile as the active profile for the session."""
    try:
        success = set_active_profile(user_id)
        if not success:
            return jsonify({"error": f"Profile '{user_id}' not found."}), 404
        return jsonify({
            "success": True,
            "message": f"Profile '{user_id}' set as active.",
            "active_profile_id": user_id,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/active", methods=["GET"])
def get_active_user_profile():
    """Returns the currently active user profile."""
    try:
        profile = get_active_profile()
        if not profile:
            return jsonify({
                "success": True,
                "active_profile": None,
                "message": "No active profile set. Create a profile to get personalised recommendations.",
            })
        return jsonify({"success": True, "active_profile": profile})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/user/<user_id>/context", methods=["GET"])
def get_user_personalized_context(user_id):
    """
    Returns full personalisation context for a user:
      - ICMR-NIN profile key mapping
      - Daily calorie and protein targets
      - BMI info (with limitation disclaimer)
      - Breakfast suggestions based on morning routine + preferences
      - Portion guidance based on activity level and goal
      - Allergen avoidance codes
    """
    try:
        ctx = get_personalized_context(user_id)
        return jsonify({"success": True, "data": ctx})
    except KeyError as ke:
        return jsonify({"success": False, "error": str(ke)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/user/<user_id>/breakfast", methods=["GET"])
def get_user_breakfast_suggestions(user_id):
    """Returns personalised morning breakfast suggestions for a user."""
    try:
        profile = get_profile(user_id)
        if not profile:
            return jsonify({"error": f"Profile '{user_id}' not found."}), 404

        suggestions = get_breakfast_suggestions(profile)
        routine = profile.get("morning_routine_preference", "regular")
        routine_label = {
            "early_bird": "Early Bird (before 7 am)",
            "regular": "Regular (7–9 am)",
            "late_riser": "Late Riser (after 9 am)"
        }.get(routine, routine)

        return jsonify({
            "success": True,
            "user_id": user_id,
            "display_name": profile.get("display_name"),
            "morning_routine": routine_label,
            "suggestions": suggestions,
            "disclaimer": PHASE8_DISCLAIMER,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/profiles/bmi", methods=["POST"])
def compute_bmi():
    """
    Calculates BMI from provided height_cm and weight_kg.
    Always returns the BMI limitation note.
    """
    try:
        data = request.get_json() or {}
        height = data.get("height_cm")
        weight = data.get("weight_kg")

        if height is None or weight is None:
            return jsonify({"error": "'height_cm' and 'weight_kg' are required."}), 400

        try:
            height = float(height)
            weight = float(weight)
        except (ValueError, TypeError):
            return jsonify({"error": "'height_cm' and 'weight_kg' must be numeric."}), 400

        if not (50 <= height <= 250):
            return jsonify({"error": "'height_cm' must be between 50 and 250 cm."}), 400
        if not (20 <= weight <= 300):
            return jsonify({"error": "'weight_kg' must be between 20 and 300 kg."}), 400

        result = calculate_bmi(height, weight)
        return jsonify({"success": True, "data": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Phase 9: Recommendation System Engine
from src.recommendation_engine import (
    RECOMMENDATION_DISCLAIMER,
    SAFETY_OVERRIDE_NOTE,
    SCORE_WEIGHTS,
    RecipeSafetyFilter,
    SuitabilityScorer,
    get_recipe_recommendations,
    get_ingredient_substitutions,
    get_alternative_foods,
    generate_daily_meal_plan,
    generate_weekly_meal_plan,
)


@app.route("/api/profiles/validate", methods=["POST"])
def validate_user_profile():
    """Validates profile data without saving. Returns errors if invalid."""
    try:
        data = request.get_json() or {}
        is_valid, errors = validate_profile_data(data)
        return jsonify({
            "success": True,
            "valid": is_valid,
            "errors": errors,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================================================================
# Phase 9 — Recommendation System & Meal Planning Endpoints
# =========================================================================

def _resolve_profile_for_recommendations(user_id_param: Optional[str] = None) -> Dict[str, Any]:
    """Helper to fetch target profile or fallback to active or default profile."""
    if user_id_param:
        prof = get_profile(user_id_param)
        if prof:
            return prof
    active = get_active_profile()
    if active:
        return active
    return {
        "display_name": "Guest User",
        "dietary_goal": "general_wellness",
        "dietary_preference": "vegetarian",
        "allergies": [],
        "activity_level": "sedentary",
        "budget": "medium",
        "cooking_time_minutes": 35,
        "daily_calorie_target": 2000.0
    }


@app.route("/api/recommendations/formula", methods=["GET"])
def get_recommendation_formula():
    """Returns transparent scoring weights, formula explanation, and non-medical disclaimers."""
    return jsonify({
        "success": True,
        "scale": "0 to 100",
        "weights": SCORE_WEIGHTS,
        "formula_description": (
            "Suitability Score (0-100) = (Goal Alignment × 0.30) + "
            "(Macro & Micro Balance × 0.25) + (Content Preference Similarity × 0.20) + "
            "(Time Feasibility × 0.15) + (Budget Fit × 0.10)."
        ),
        "safety_precedence_rule": SAFETY_OVERRIDE_NOTE,
        "disclaimer": RECOMMENDATION_DISCLAIMER
    })


@app.route("/api/recommendations/recipes", methods=["GET"])
def get_recipes_recommendations_endpoint():
    """Returns safe, personalized recipe recommendations ranked by suitability score."""
    try:
        user_id = request.args.get("user_id")
        course = request.args.get("course")
        cuisine = request.args.get("cuisine")
        max_time = request.args.get("max_cooking_time")
        top_n = int(request.args.get("top_n", 10))

        profile = _resolve_profile_for_recommendations(user_id)
        time_limit = int(max_time) if max_time and max_time.isdigit() else None

        recommendations = get_recipe_recommendations(
            user_profile=profile,
            course=course,
            cuisine=cuisine,
            max_cooking_time=time_limit,
            top_n=top_n
        )

        return jsonify({
            "success": True,
            "count": len(recommendations),
            "user_profile": {
                "display_name": profile.get("display_name"),
                "dietary_goal": profile.get("dietary_goal"),
                "dietary_preference": profile.get("dietary_preference"),
                "allergies": profile.get("allergies", [])
            },
            "recommendations": recommendations,
            "disclaimer": RECOMMENDATION_DISCLAIMER
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommendations/score", methods=["POST"])
def score_recipe_endpoint():
    """Calculates 0-100 suitability score for an arbitrary recipe against user profile."""
    try:
        body = request.get_json() or {}
        recipe = body.get("recipe")
        if not recipe or not isinstance(recipe, dict):
            return jsonify({"error": "'recipe' object is required in request body."}), 400

        user_id = body.get("user_id")
        profile = body.get("user_profile") or _resolve_profile_for_recommendations(user_id)

        score_res = SuitabilityScorer.score_recipe(recipe, profile)
        return jsonify({
            "success": True,
            "data": score_res
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommendations/alternatives", methods=["GET"])
def get_alternatives_endpoint():
    """Provides healthy swaps and reasoning for an unfavorable food dish."""
    try:
        food_name = request.args.get("food_name", "").strip()
        if not food_name:
            return jsonify({"error": "Query parameter 'food_name' is required."}), 400

        user_id = request.args.get("user_id")
        profile = _resolve_profile_for_recommendations(user_id)
        goal = profile.get("dietary_goal")

        result = get_alternative_foods(food_name, user_goal=goal)
        return jsonify({
            "success": True,
            "data": result
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommendations/substitutions", methods=["GET"])
def get_substitutions_endpoint():
    """Returns curated culinary ingredient substitutions filtered by query, goal, or allergy."""
    try:
        q = request.args.get("query")
        goal = request.args.get("target_goal")
        allergy = request.args.get("allergy_tag")

        substitutions = get_ingredient_substitutions(query=q, target_goal=goal, allergy_tag=allergy)
        return jsonify({
            "success": True,
            "count": len(substitutions),
            "substitutions": substitutions,
            "disclaimer": RECOMMENDATION_DISCLAIMER
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommendations/meal-plan/daily", methods=["GET"])
def get_daily_meal_plan_endpoint():
    """Generates a personalized 4-meal daily plan adhering to calorie targets & safety."""
    try:
        user_id = request.args.get("user_id")
        cals_param = request.args.get("target_calories")
        profile = _resolve_profile_for_recommendations(user_id)

        target_cals = float(cals_param) if cals_param else None
        plan = generate_daily_meal_plan(profile, target_calories=target_cals)

        return jsonify({
            "success": True,
            "data": plan
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Phase 10: Backend Database & Meal Diary Repository
from src.database import (
    init_db,
    seed_initial_data,
    log_meal,
    get_meal_history,
    delete_meal_log,
    get_meal_totals,
    save_meal_plan_db,
    get_latest_meal_plan_db,
    record_feedback_db,
    get_feedback_list_db,
    get_database_stats,
    db_save_profile,
    db_get_profile,
    db_list_profiles,
    db_delete_profile,
    db_set_active_profile,
    db_get_active_profile,
)

# Auto-initialize and seed SQLite tables on startup
init_db()
seed_initial_data()


@app.route("/api/recommendations/meal-plan/weekly", methods=["GET"])
def get_weekly_meal_plan_endpoint():
    """Generates a 7-day meal plan with inter-day variety and grocery list."""
    try:
        user_id = request.args.get("user_id")
        cals_param = request.args.get("target_calories")
        profile = _resolve_profile_for_recommendations(user_id)

        target_cals = float(cals_param) if cals_param else None
        weekly_plan = generate_weekly_meal_plan(profile, target_calories=target_cals)

        # Persist generated weekly plan in SQLite
        uid = profile.get("user_id") or "guest"
        try:
            save_meal_plan_db(uid, "weekly", weekly_plan)
        except Exception:
            pass

        return jsonify({
            "success": True,
            "data": weekly_plan
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================================================================
# Phase 10 — Database, Meal Diary & Feedback Endpoints
# =========================================================================

@app.route("/api/meals/log", methods=["POST"])
def log_meal_endpoint():
    """
    Logs a consumed food dish or recipe into SQLite meal diary.
    Requires user_id (or active profile), food_name, portion_g, calories,
    protein_g, carbs_g, fat_g, and meal_slot.
    """
    try:
        payload = request.get_json() or {}
        user_id = payload.get("user_id")
        if not user_id:
            active = get_active_profile()
            user_id = active.get("user_id") if active else "guest_user"
        payload["user_id"] = user_id

        # If user is guest, auto-create guest profile in DB if needed
        if user_id == "guest_user":
            try:
                db_save_profile({
                    "user_id": "guest_user",
                    "display_name": "Guest User",
                    "daily_calorie_target": 2000.0,
                    "dietary_goal": "general_wellness",
                    "dietary_preference": "vegetarian"
                })
            except Exception:
                pass

        saved_log = log_meal(payload)
        return jsonify({
            "success": True,
            "message": "Meal logged successfully",
            "log": saved_log
        }), 201

    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/meals/history", methods=["GET"])
def get_meal_history_endpoint():
    """Retrieves logged meal records with optional date range and meal slot filtering."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            active = get_active_profile()
            user_id = active.get("user_id") if active else "guest_user"

        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")
        meal_slot = request.args.get("meal_slot")
        limit = int(request.args.get("limit", 100))

        history = get_meal_history(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            meal_slot=meal_slot,
            limit=limit
        )

        return jsonify({
            "success": True,
            "count": len(history),
            "user_id": user_id,
            "meals": history
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/meals/totals", methods=["GET"])
def get_meal_totals_endpoint():
    """Calculates daily nutritional totals and progress toward user's calorie/RDA target."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            active = get_active_profile()
            user_id = active.get("user_id") if active else "guest_user"

        date_str = request.args.get("date")
        totals_data = get_meal_totals(user_id=user_id, date_str=date_str)

        return jsonify({
            "success": True,
            "data": totals_data
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/meals/<log_id>", methods=["DELETE"])
def delete_meal_log_endpoint(log_id: str):
    """Deletes a logged meal record from SQLite."""
    try:
        user_id = request.args.get("user_id")
        success = delete_meal_log(log_id, user_id=user_id)
        if success:
            return jsonify({"success": True, "message": f"Meal log {log_id} deleted."}), 200
        return jsonify({"success": False, "error": "Meal log not found or already deleted."}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def submit_feedback_endpoint():
    """Records user feedback on food recognition, suitability scores, or recommendations."""
    try:
        body = request.get_json() or {}
        if not body.get("feedback_type"):
            return jsonify({"error": "Field 'feedback_type' is required."}), 400

        result = record_feedback_db(body)
        return jsonify({
            "success": True,
            "message": "Feedback recorded. Thank you for helping improve NutriSense AI!",
            "data": result
        }), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/feedback", methods=["GET"])
def get_feedback_endpoint():
    """Retrieves recent user feedback entries."""
    try:
        limit = int(request.args.get("limit", 50))
        entries = get_feedback_list_db(limit=limit)
        return jsonify({
            "success": True,
            "count": len(entries),
            "feedback": entries
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/db/stats", methods=["GET"])
def get_db_stats_endpoint():
    """Returns SQLite database health, table row counts, and storage metrics."""
    try:
        stats = get_database_stats()
        return jsonify({
            "success": True,
            "database": stats
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

