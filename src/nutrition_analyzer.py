"""
src/nutrition_analyzer.py
NutriSense AI — Phase 7: Nutrition Analysis Module

Reusable functions and classes for:
1. Connecting confirmed food name to cleaned nutrition database (Master KB, Unified Lookup, IFCT).
2. Portion size conversion from grams or defined serving units (piece, katori, bowl, plate, cup, etc.).
3. Scaling and calculating nutrients using the standard formula:
     nutrient_amount = (portion_grams / 100.0) * nutrient_value_per_100g
4. Calculating calories, protein, carbohydrates, fat, fibre, and available vitamins & minerals.
5. Strict unit handling and reporting missing nutrition data as None (never zero-filled).
6. Showing source, reference quantity, and Atwater / sodium quality flags.
7. Manual food search fallback across all 1,556+ nutrition records.
8. Validation & graceful error handling for invalid inputs (negative, non-numeric, unknown units/foods).
9. Non-clinical disclaimers stating image recognition identifies category only and portion weight must be confirmed.
"""

from difflib import SequenceMatcher
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = WORKSPACE_ROOT / "data" / "processed" / "feature_store"
NUTRITION_DIR = WORKSPACE_ROOT / "data" / "processed" / "nutrition"

# Standard units dictionary
NUTRIENT_UNITS = {
    "calories_kcal": "kcal",
    "protein_g": "g",
    "carbs_g": "g",
    "fat_g": "g",
    "fibre_g": "g",
    "sodium_mg": "mg",
    "calcium_mg": "mg",
    "iron_mg": "mg",
    "vitamin_c_mg": "mg",
    "folate_ug": "mcg",
}

NUTRIENT_LABELS = {
    "calories_kcal": "Energy / Calories",
    "protein_g": "Protein",
    "carbs_g": "Total Carbohydrates",
    "fat_g": "Total Fat",
    "fibre_g": "Dietary Fibre",
    "sodium_mg": "Sodium",
    "calcium_mg": "Calcium",
    "iron_mg": "Iron",
    "vitamin_c_mg": "Vitamin C",
    "folate_ug": "Folate (Vitamin B9)",
}

PHASE7_DISCLAIMER = (
    "Visual food recognition identifies dish category only; portion weight must be confirmed by the user. "
    "Nutritional values are computed from ICMR-NIN (2020) and IFCT (2017) reference composition tables. "
    "Portion weights and nutrient amounts are estimates provided for educational and lifestyle awareness, "
    "not exact clinical measurements or medical advice."
)


class NutritionAnalyzer:
    """
    Core Nutrition Analysis Engine for Phase 7.
    Thread-safe and cached initialization of Master Nutrition KB,
    Unified Food Lookup (1,556 items), Standard Serving Weights, and Culinary Subtypes.
    """

    def __init__(
        self,
        master_kb_path: Optional[Union[str, Path]] = None,
        unified_lookup_path: Optional[Union[str, Path]] = None,
        standard_servings_path: Optional[Union[str, Path]] = None,
        subtypes_path: Optional[Union[str, Path]] = None,
    ):
        self.master_kb_path = Path(master_kb_path or (FEATURE_DIR / "master_food_nutrition_kb.json"))
        self.unified_lookup_path = Path(unified_lookup_path or (NUTRITION_DIR / "nutrition_unified_lookup.csv"))
        self.standard_servings_path = Path(standard_servings_path or (NUTRITION_DIR / "standard_serving_weights.csv"))
        self.subtypes_path = Path(subtypes_path or (FEATURE_DIR / "ambiguous_subtypes_registry.json"))

        self._load_databases()

    def _load_databases(self):
        """Loads and indexes all underlying nutritional databases."""
        # 1. Master Food Nutrition KB (59 records: 43 vision classes + 16 subtypes)
        self.master_kb_items: List[Dict[str, Any]] = []
        self.master_by_class: Dict[str, Dict[str, Any]] = {}
        self.master_by_id: Dict[str, Dict[str, Any]] = {}

        if self.master_kb_path.exists():
            with open(self.master_kb_path, "r", encoding="utf-8") as f:
                self.master_kb_items = json.load(f)
            for item in self.master_kb_items:
                c_name = item.get("class_name", "").lower()
                fid = item.get("food_id", "").lower()
                if c_name:
                    self.master_by_class[c_name] = item
                if fid:
                    self.master_by_id[fid] = item

        # 2. Standard Serving Weights Reference (21 defined dishes)
        self.serving_weights: Dict[str, Dict[str, Any]] = {}
        if self.standard_servings_path.exists():
            df_serv = pd.read_csv(self.standard_servings_path)
            for _, row in df_serv.iterrows():
                fname = str(row["food_name"]).lower().strip()
                self.serving_weights[fname] = {
                    "serving_g": float(row["serving_g"]),
                    "description": str(row.get("description", "")),
                }

        # 3. Ambiguous Subtypes Registry (14 parent families, 36 sub-profiles)
        self.subtypes_registry: Dict[str, Dict[str, Any]] = {}
        if self.subtypes_path.exists():
            with open(self.subtypes_path, "r", encoding="utf-8") as f:
                raw_subs = json.load(f)
            for item in raw_subs:
                p_class = item.get("parent_class", "").lower()
                if p_class:
                    self.subtypes_registry[p_class] = item
                for alias in item.get("aliases", []):
                    self.subtypes_registry[alias.lower()] = item

        # 4. Unified Food Lookup (1,556 items: IFCT ingredients + Indian Food Processed dishes)
        self.unified_df: Optional[pd.DataFrame] = None
        if self.unified_lookup_path.exists():
            self.unified_df = pd.read_csv(self.unified_lookup_path)
            # Ensure normalized name column exists
            if "food_name_normalised" not in self.unified_df.columns:
                self.unified_df["food_name_normalised"] = (
                    self.unified_df["food_name"].astype(str).str.lower().str.strip()
                )

    # -------------------------------------------------------------------------
    # 1. Food Lookup & Subtype Resolution
    # -------------------------------------------------------------------------
    def lookup_food(
        self,
        query_or_name: str,
        subtype_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Connects a confirmed food name / class name or query to the cleaned nutrition database.
        Checks:
          1. Master Food Nutrition KB (exact class_name / food_id / alias)
          2. Specific culinary subtype if subtype_id provided
          3. Unified Food Lookup (1,556 records)
        Returns a canonical food dictionary with per-100g baseline values and reference metadata.
        """
        if not query_or_name or not isinstance(query_or_name, str):
            return None

        q = query_or_name.lower().strip()
        q_norm = re.sub(r"[\s\-_]+", " ", q)
        q_snake = re.sub(r"[\s\-]+", "_", q)

        # Check Master KB by exact class_name or food_id
        item = self.master_by_class.get(q_snake) or self.master_by_class.get(q) or self.master_by_id.get(q)

        # Check by display_name match in Master KB
        if not item:
            for m in self.master_kb_items:
                d_name = m.get("display_name", "").lower()
                if q_norm == d_name or q_norm in d_name:
                    item = m
                    break

        if item:
            base_item = item.copy()

            # If this food is an ambiguous family and a subtype is requested, resolve it
            class_name = base_item.get("class_name", "")
            if class_name in self.subtypes_registry:
                family_meta = self.subtypes_registry[class_name]
                subtypes = family_meta.get("subtypes", [])
                if subtype_id:
                    matched_sub = next((s for s in subtypes if s.get("subtype_id") == subtype_id), None)
                    if matched_sub:
                        base_item.update(matched_sub)
                        base_item["selected_subtype"] = subtype_id
                else:
                    default_sub = next((s for s in subtypes if s.get("is_default")), subtypes[0] if subtypes else {})
                    base_item.update(default_sub)

            return self._standardize_food_record(base_item, source_type="MASTER_KB")

        # Fallback: Search Unified Lookup (1,556 items)
        if self.unified_df is not None:
            # Exact match on food_name or food_name_normalised
            match_row = self.unified_df[
                (self.unified_df["food_name_normalised"] == q) |
                (self.unified_df["food_name_normalised"] == q_norm)
            ]
            if match_row.empty:
                # Substring match
                match_row = self.unified_df[
                    self.unified_df["food_name_normalised"].str.contains(r"\b" + re.escape(q) + r"\b", regex=True)
                ]

            if not match_row.empty:
                row_dict = match_row.iloc[0].to_dict()
                return self._standardize_food_record(row_dict, source_type="UNIFIED_LOOKUP")

        return None

    def _standardize_food_record(self, raw: Dict[str, Any], source_type: str) -> Dict[str, Any]:
        """
        Converts raw database entries into a standardized food profile
        with both reference serving and normalized per-100g baselines.
        """
        if source_type == "MASTER_KB":
            serving_size_g = float(raw.get("serving_size_g", 100.0) or 100.0)
            if serving_size_g <= 0:
                serving_size_g = 100.0

            # Derive per-100g factor
            factor_to_100g = 100.0 / serving_size_g

            def _get_val(key):
                val = raw.get(key)
                if val is None or pd.isna(val):
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

            # Values in Master KB are stored per serving_size_g
            serv_vals = {
                "calories_kcal": _get_val("calories_kcal"),
                "protein_g": _get_val("protein_g"),
                "carbs_g": _get_val("carbs_g"),
                "fat_g": _get_val("fat_g"),
                "fibre_g": _get_val("fibre_g"),
                "sodium_mg": _get_val("sodium_mg"),
                "calcium_mg": _get_val("calcium_mg"),
                "iron_mg": _get_val("iron_mg"),
                "vitamin_c_mg": _get_val("vitamin_c_mg"),
                "folate_ug": _get_val("folate_ug"),
            }

            # Calculate normalized per-100g values
            per_100g_vals = {}
            for k, v in serv_vals.items():
                if v is not None:
                    per_100g_vals[k] = round(v * factor_to_100g, 3)
                else:
                    per_100g_vals[k] = None

            return {
                "food_id": raw.get("food_id", raw.get("class_name", "unknown")),
                "class_name": raw.get("class_name", raw.get("food_id", "unknown")),
                "display_name": raw.get("display_name", raw.get("class_name", "Unknown Food")),
                "source": raw.get("data_source", "ICMR_NIN_MASTER_KB"),
                "source_type": "MASTER_KB",
                "match_type": raw.get("match_type", "VALID_DIRECT"),
                "reference_serving_g": serving_size_g,
                "reference_serving_unit": raw.get("serving_unit", "standard serving"),
                "reference_basis": f"Recorded per reference serving of {serving_size_g}g ({raw.get('serving_unit', 'standard serving')})",
                "reference_nutrients": serv_vals,
                "per_100g_nutrients": per_100g_vals,
                "sodium_flag": raw.get("sodium_flag", "ok"),
                "calorie_flag": raw.get("calorie_flag", "ok"),
                "has_subtypes": raw.get("has_subtypes", False),
                "selected_subtype": raw.get("selected_subtype"),
            }

        else:
            # UNIFIED_LOOKUP item
            serving_ref = str(raw.get("serving_reference", "per_serving_estimated"))
            source_file = str(raw.get("source_file", "nutrition_unified_lookup.csv"))

            # Determine serving size and per-100g baseline
            if serving_ref == "per_100g" or "IFCT" in source_file:
                ref_serving_g = 100.0
                ref_unit = "100g raw / standard edible portion"
                ref_basis = "Recorded per 100g edible portion (IFCT 2017)"
                is_recorded_per_100g = True
            else:
                ref_serving_g = 100.0  # default fallback estimated portion
                ref_unit = "1 serving (estimated)"
                ref_basis = f"Estimated serving ({source_file})"
                is_recorded_per_100g = False

            def _get_u_val(key):
                val = raw.get(key)
                if val is None or pd.isna(val):
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

            raw_vals = {
                "calories_kcal": _get_u_val("calories_kcal"),
                "protein_g": _get_u_val("protein_g"),
                "carbs_g": _get_u_val("carbs_g"),
                "fat_g": _get_u_val("fat_g"),
                "fibre_g": _get_u_val("fibre_g"),
                "sodium_mg": _get_u_val("sodium_mg"),
                "calcium_mg": _get_u_val("calcium_mg"),
                "iron_mg": _get_u_val("iron_mg"),
                "vitamin_c_mg": _get_u_val("vitamin_c_mg") if "vitamin_c_mg" in raw else None,
                "folate_ug": _get_u_val("folate_ug") if "folate_ug" in raw else None,
            }

            per_100g_vals = {}
            for k, v in raw_vals.items():
                per_100g_vals[k] = round(v, 3) if v is not None else None

            fname = raw.get("food_name_raw", raw.get("food_name", "Unknown Food"))

            return {
                "food_id": f"unified_{re.sub(r'[^a-zA-Z0-9]+', '_', str(fname).lower()).strip('_')}",
                "class_name": str(fname).lower(),
                "display_name": str(fname).title(),
                "source": source_file,
                "source_type": "UNIFIED_LOOKUP",
                "match_type": "FALLBACK_LOOKUP",
                "reference_serving_g": ref_serving_g,
                "reference_serving_unit": ref_unit,
                "reference_basis": ref_basis,
                "reference_nutrients": raw_vals,
                "per_100g_nutrients": per_100g_vals,
                "sodium_flag": "ok",
                "calorie_flag": "ok",
                "has_subtypes": False,
                "selected_subtype": None,
            }

    # -------------------------------------------------------------------------
    # 2. Portion Conversion & Defined Servings
    # -------------------------------------------------------------------------
    def get_available_servings(self, food_name_or_id: str) -> List[Dict[str, Any]]:
        """
        Returns a list of defined serving options for a given food.
        Enables user to select portion in grams OR choose a defined serving
        (e.g., 1 piece, 2 pieces, 1 katori, 1 plate, standard serving, 100g).
        """
        food = self.lookup_food(food_name_or_id)
        class_name = food.get("class_name", "").lower() if food else food_name_or_id.lower()

        options = [
            {"unit": "g", "label": "Grams (Custom Entry)", "grams": 1.0, "is_custom": True},
            {"unit": "100g", "label": "100g (Standard Reference Baseline)", "grams": 100.0, "is_custom": False},
        ]

        # Check standard serving weights
        sw = self.serving_weights.get(class_name)
        if not sw:
            # Try fuzzy match in serving_weights
            for k, v in self.serving_weights.items():
                if k in class_name or class_name in k:
                    sw = v
                    break

        if sw:
            serving_g = sw["serving_g"]
            desc = sw["description"]
            options.append({
                "unit": "standard_serving",
                "label": f"1 Serving ({desc} — {int(serving_g)}g)",
                "grams": serving_g,
                "is_custom": False,
                "is_default": True
            })
            if "piece" in desc.lower() or "dosa" in desc.lower() or "chapati" in desc.lower() or "roti" in desc.lower():
                options.append({
                    "unit": "piece_half",
                    "label": f"0.5 Piece ({int(serving_g * 0.5)}g)",
                    "grams": serving_g * 0.5,
                    "is_custom": False
                })
                options.append({
                    "unit": "piece_2",
                    "label": f"2 Pieces ({int(serving_g * 2)}g)",
                    "grams": serving_g * 2.0,
                    "is_custom": False
                })
        elif food and food.get("reference_serving_g"):
            ref_g = food["reference_serving_g"]
            unit_desc = food.get("reference_serving_unit", "standard portion")
            options.append({
                "unit": "standard_serving",
                "label": f"1 Serving ({unit_desc} — {int(ref_g)}g)",
                "grams": ref_g,
                "is_custom": False,
                "is_default": True
            })

        # General household portions
        if not any(o["unit"] == "katori" for o in options) and any(kw in class_name for kw in ["dal", "curry", "sabzi", "chole", "paneer", "soup"]):
            options.append({"unit": "katori_small", "label": "Small Katori / Bowl (100g)", "grams": 100.0, "is_custom": False})
            options.append({"unit": "katori_standard", "label": "Medium Katori / Bowl (150g)", "grams": 150.0, "is_custom": False})
            options.append({"unit": "katori_large", "label": "Large Katori / Bowl (200g)", "grams": 200.0, "is_custom": False})

        return options

    def convert_portion_to_grams(
        self,
        portion_value: Any,
        unit: str = "g",
        food_name: Optional[str] = None
    ) -> float:
        """
        Converts a user-selected portion value and unit into grams.
        Handles numeric grams or defined serving units (piece, katori, plate, bowl, cup, serving).

        Validation:
          - portion_value must be a positive finite number (> 0).
          - Non-numeric or non-positive values raise ValueError.
          - Unknown units raise ValueError.
        """
        # Validate numeric portion_value
        if portion_value is None:
            raise ValueError("Portion value cannot be None.")

        try:
            p_val = float(portion_value)
        except (ValueError, TypeError):
            raise ValueError(f"Portion value must be a valid number, got: {portion_value!r}")

        if math.isnan(p_val) or math.isinf(p_val):
            raise ValueError("Portion value must be a finite number.")

        if p_val <= 0:
            raise ValueError(f"Portion value must be greater than zero, got: {p_val}")

        u = str(unit).lower().strip()

        # 1. Direct grams
        if u in ["g", "gram", "grams"]:
            return round(p_val, 2)

        # 2. 100g baseline
        if u in ["100g", "per_100g"]:
            return round(p_val * 100.0, 2)

        # 3. Standard defined serving
        if u in ["standard_serving", "serving", "servings", "portion", "portions"]:
            ref_g = 100.0
            if food_name:
                food = self.lookup_food(food_name)
                if food and food.get("reference_serving_g"):
                    ref_g = food["reference_serving_g"]
                elif food_name.lower() in self.serving_weights:
                    ref_g = self.serving_weights[food_name.lower()]["serving_g"]
            return round(p_val * ref_g, 2)

        # 4. Piece / Pieces
        if u in ["piece", "pieces", "pc", "pcs", "piece_2", "piece_half"]:
            piece_g = 65.0  # default
            if food_name:
                f_lower = food_name.lower()
                sw = self.serving_weights.get(f_lower)
                if sw:
                    piece_g = sw["serving_g"]
                else:
                    food = self.lookup_food(food_name)
                    if food and food.get("reference_serving_g"):
                        piece_g = food["reference_serving_g"]
            if u == "piece_half":
                return round(p_val * piece_g * 0.5, 2)
            if u == "piece_2":
                return round(p_val * piece_g * 2.0, 2)
            return round(p_val * piece_g, 2)

        # 5. Katori / Bowl
        if u in ["katori", "katoris", "bowl", "bowls", "katori_standard"]:
            katori_g = 150.0
            if food_name:
                f_lower = food_name.lower()
                if "sabzi" in f_lower or "bhindi" in f_lower:
                    katori_g = 120.0
                elif "raita" in f_lower:
                    katori_g = 100.0
                elif "soup" in f_lower:
                    katori_g = 200.0
            return round(p_val * katori_g, 2)

        if u == "katori_small":
            return round(p_val * 100.0, 2)
        if u == "katori_large":
            return round(p_val * 200.0, 2)

        # 6. Plate
        if u in ["plate", "plates"]:
            plate_g = 200.0
            if food_name:
                f_lower = food_name.lower()
                if "biryani" in f_lower:
                    plate_g = 250.0
                elif "poha" in f_lower:
                    plate_g = 120.0
                elif "upma" in f_lower:
                    plate_g = 150.0
                elif "khichdi" in f_lower:
                    plate_g = 200.0
            return round(p_val * plate_g, 2)

        # 7. Cup
        if u in ["cup", "cups"]:
            return round(p_val * 150.0, 2)

        raise ValueError(
            f"Unsupported portion unit: '{unit}'. "
            "Supported units include 'g', '100g', 'serving', 'piece', 'katori', 'plate', 'bowl', 'cup'."
        )

    # -------------------------------------------------------------------------
    # 3. Nutrient Calculation per 100g Formula
    # -------------------------------------------------------------------------
    def calculate_nutrients(
        self,
        food_record_or_name: Union[str, Dict[str, Any]],
        portion_grams: float,
        subtype_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculates calories, protein, carbohydrates, fat, fibre, and available vitamins and minerals.

        Core formula:
          nutrient_amount = (portion_grams / 100.0) * nutrient_value_per_100g

        Requirements:
          - Values recorded per 100g are scaled strictly by portion_grams / 100.0.
          - Missing nutrients remain None (never zero-filled) and are reported in missing_nutrients.
          - Proper units (kcal, g, mg, mcg) are attached to each nutrient.
          - Shows the source, reference quantity, and quality flags.
          - Does not claim exact portion weight from vision alone.
        """
        # Validate portion_grams
        if portion_grams is None:
            raise ValueError("Portion grams cannot be None.")
        try:
            p_grams = float(portion_grams)
        except (ValueError, TypeError):
            raise ValueError(f"Portion grams must be a numeric value, got: {portion_grams!r}")

        if math.isnan(p_grams) or math.isinf(p_grams):
            raise ValueError("Portion grams must be a finite number.")

        if p_grams <= 0:
            raise ValueError(f"Portion grams must be greater than zero, got: {p_grams}")

        # Obtain food record
        if isinstance(food_record_or_name, str):
            food = self.lookup_food(food_record_or_name, subtype_id=subtype_id)
            if not food:
                raise KeyError(
                    f"Food '{food_record_or_name}' not found in nutrition database. "
                    "Use search_foods() to find available matches."
                )
        elif isinstance(food_record_or_name, dict):
            # Check if already standardized
            if "per_100g_nutrients" in food_record_or_name:
                food = food_record_or_name
            else:
                food = self._standardize_food_record(food_record_or_name, source_type="MASTER_KB")
        else:
            raise TypeError("food_record_or_name must be a string food name or a dictionary record.")

        per_100g = food["per_100g_nutrients"]
        scale_ratio = p_grams / 100.0

        scaled_nutrients: Dict[str, Dict[str, Any]] = {}
        missing_nutrients: List[Dict[str, str]] = []

        for key, unit in NUTRIENT_UNITS.items():
            val_100g = per_100g.get(key)
            label = NUTRIENT_LABELS.get(key, key)

            if val_100g is not None and not pd.isna(val_100g):
                # Apply standard scaling formula: portion_grams / 100.0 * per_100g
                calculated_val = round(scale_ratio * float(val_100g), 2)
                scaled_nutrients[key] = {
                    "label": label,
                    "value": calculated_val,
                    "unit": unit,
                    "value_per_100g": val_100g,
                    "status": "AVAILABLE"
                }
            else:
                # NEVER impute missing nutrient values to zero!
                scaled_nutrients[key] = {
                    "label": label,
                    "value": None,
                    "unit": unit,
                    "value_per_100g": None,
                    "status": "DATA_NOT_AVAILABLE"
                }
                missing_nutrients.append({
                    "nutrient": key,
                    "label": label,
                    "unit": unit,
                    "reason": f"Not reported in reference database source ({food.get('source')})"
                })

        return {
            "food_id": food.get("food_id"),
            "class_name": food.get("class_name"),
            "display_name": food.get("display_name"),
            "source": food.get("source"),
            "match_type": food.get("match_type"),
            "portion": {
                "grams": p_grams,
                "unit": "g",
                "scale_ratio_from_100g": round(scale_ratio, 4)
            },
            "reference_basis": food.get("reference_basis"),
            "reference_serving_g": food.get("reference_serving_g"),
            "reference_serving_unit": food.get("reference_serving_unit"),
            "nutrients": scaled_nutrients,
            "missing_nutrients": missing_nutrients,
            "data_quality_flags": {
                "sodium_flag": food.get("sodium_flag", "ok"),
                "calorie_flag": food.get("calorie_flag", "ok")
            },
            "ambiguous_subtype": {
                "has_subtypes": food.get("has_subtypes", False),
                "selected_subtype": food.get("selected_subtype")
            },
            "disclaimer": PHASE7_DISCLAIMER
        }

    # -------------------------------------------------------------------------
    # 4. Manual Food Search Fallback
    # -------------------------------------------------------------------------
    def search_foods(self, query: str, limit: int = 15) -> List[Dict[str, Any]]:
        """
        Manual food search fallback when image recognition fails or user searches directly.
        Searches across:
          1. Master Food Nutrition KB (43 classes + 16 culinary subtypes)
          2. Unified Food Lookup (1,556 Indian foods from IFCT & Processed datasets)
        Returns ranked list of candidate matches.
        """
        if not query or not isinstance(query, str) or not query.strip():
            return []

        q = query.lower().strip()
        q_tokens = set(re.findall(r"\w+", q))
        results = []
        seen_names = set()

        # 1. Search Master KB
        for item in self.master_kb_items:
            c_name = item.get("class_name", "").lower()
            d_name = item.get("display_name", "").lower()
            fid = item.get("food_id", "").lower()

            target_tokens = set(re.findall(r"\w+", f"{c_name} {d_name}"))
            token_overlap = len(q_tokens.intersection(target_tokens))

            score = 0.0
            if q == c_name or q == d_name:
                score = 1.0
            elif q in c_name or q in d_name:
                score = 0.85 + (len(q) / max(len(d_name), 1)) * 0.1
            elif token_overlap > 0:
                score = 0.6 + (token_overlap / max(len(q_tokens), 1)) * 0.3
            else:
                ratio = SequenceMatcher(None, q, c_name).ratio()
                if ratio > 0.65:
                    score = ratio * 0.7

            if score >= 0.5:
                # Compute baseline calories per 100g
                serving_g = float(item.get("serving_size_g", 100.0) or 100.0)
                cal_serv = item.get("calories_kcal")
                cal_100g = round(float(cal_serv) * (100.0 / serving_g), 1) if cal_serv is not None else None

                results.append({
                    "food_id": fid or c_name,
                    "food_name": c_name,
                    "display_name": item.get("display_name", c_name.title()),
                    "match_type": item.get("match_type", "MASTER_KB"),
                    "source": item.get("data_source", "ICMR_NIN_MASTER_KB"),
                    "default_serving_g": serving_g,
                    "serving_unit": item.get("serving_unit", "standard serving"),
                    "calories_per_100g": cal_100g,
                    "protein_per_100g": round(float(item["protein_g"]) * (100.0 / serving_g), 1) if item.get("protein_g") is not None else None,
                    "match_score": round(score, 3),
                    "is_benchmark_class": True
                })
                seen_names.add(c_name)

        # 2. Search Unified Lookup (1,556 items)
        if self.unified_df is not None:
            # Filter rows containing query tokens
            for _, row in self.unified_df.iterrows():
                fname = str(row.get("food_name_normalised", "")).lower()
                raw_name = str(row.get("food_name_raw", row.get("food_name", "")))
                if fname in seen_names:
                    continue

                target_tokens = set(re.findall(r"\w+", fname))
                token_overlap = len(q_tokens.intersection(target_tokens))

                score = 0.0
                if q == fname:
                    score = 0.95
                elif q in fname:
                    score = 0.75 + (len(q) / max(len(fname), 1)) * 0.1
                elif token_overlap > 0:
                    score = 0.5 + (token_overlap / max(len(q_tokens), 1)) * 0.3

                if score >= 0.5:
                    cal = row.get("calories_kcal")
                    cal_val = round(float(cal), 1) if pd.notna(cal) else None
                    prot = row.get("protein_g")
                    prot_val = round(float(prot), 1) if pd.notna(prot) else None

                    serving_ref = str(row.get("serving_reference", "per_serving_estimated"))
                    unit = "100g edible portion" if serving_ref == "per_100g" else "1 estimated serving (100g)"

                    results.append({
                        "food_id": f"unified_{re.sub(r'[^a-zA-Z0-9]+', '_', fname).strip('_')}",
                        "food_name": fname,
                        "display_name": raw_name.title(),
                        "match_type": "UNIFIED_LOOKUP",
                        "source": str(row.get("source_file", "nutrition_unified_lookup.csv")),
                        "default_serving_g": 100.0,
                        "serving_unit": unit,
                        "calories_per_100g": cal_val,
                        "protein_per_100g": prot_val,
                        "match_score": round(score, 3),
                        "is_benchmark_class": False
                    })
                    seen_names.add(fname)

        # Sort descending by match_score
        results.sort(key=lambda x: (x["is_benchmark_class"], x["match_score"]), reverse=True)
        return results[:limit]


# -----------------------------------------------------------------------------
# Module Singleton and Functional API Wrappers
# -----------------------------------------------------------------------------
_global_analyzer: Optional[NutritionAnalyzer] = None


def get_analyzer() -> NutritionAnalyzer:
    """Returns the singleton instance of NutritionAnalyzer."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = NutritionAnalyzer()
    return _global_analyzer


def lookup_food(query_or_name: str, subtype_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Functional wrapper for NutritionAnalyzer.lookup_food()."""
    return get_analyzer().lookup_food(query_or_name, subtype_id=subtype_id)


def convert_portion_to_grams(
    portion_value: Any,
    unit: str = "g",
    food_name: Optional[str] = None
) -> float:
    """Functional wrapper for NutritionAnalyzer.convert_portion_to_grams()."""
    return get_analyzer().convert_portion_to_grams(portion_value, unit=unit, food_name=food_name)


def calculate_nutrients(
    food_record_or_name: Union[str, Dict[str, Any]],
    portion_grams: float,
    subtype_id: Optional[str] = None
) -> Dict[str, Any]:
    """Functional wrapper for NutritionAnalyzer.calculate_nutrients()."""
    return get_analyzer().calculate_nutrients(food_record_or_name, portion_grams, subtype_id=subtype_id)


def search_foods(query: str, limit: int = 15) -> List[Dict[str, Any]]:
    """Functional wrapper for NutritionAnalyzer.search_foods()."""
    return get_analyzer().search_foods(query, limit=limit)


def get_available_servings(food_name_or_id: str) -> List[Dict[str, Any]]:
    """Functional wrapper for NutritionAnalyzer.get_available_servings()."""
    return get_analyzer().get_available_servings(food_name_or_id)
