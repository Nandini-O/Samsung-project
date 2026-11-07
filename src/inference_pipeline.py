"""
src/inference_pipeline.py
Phase 5: End-to-End NutriSense AI Inference Pipeline

Integrates:
1. MobileNetV2 Vision Model (43-class Enhanced Indian Food classifier).
2. Master Nutrition Knowledge Base (59 discrete profiles with quality flags).
3. Ambiguous Subtype Disambiguation Engine (14 parent families, 36 discrete profiles).
4. Composite Recipe Nutrition Explorer (15 recipe-derived classes).
5. Evidence-based Dietary Scoring & Safety Alerts (ICMR-NIN 2020 RDA).
6. Portion-size Scaling & Allergen Detection.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

import sys
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.nutrient_scoring_engine import compute_suitability_score, ICMR_RDA_PROFILES
from src.vision_dataset import IMAGENET_MEAN, IMAGENET_STD

MODELS_DIR = WORKSPACE_ROOT / "models"
FEATURE_DIR = WORKSPACE_ROOT / "data" / "processed" / "feature_store"
NUTRITION_DIR = WORKSPACE_ROOT / "data" / "processed" / "nutrition"


ALLERGEN_MAP = {
    # Dairy / Lactose
    "kadai_paneer": ["DAIRY", "LACTOSE"],
    "palak_paneer": ["DAIRY", "LACTOSE"],
    "shahi_paneer": ["DAIRY", "LACTOSE"],
    "dal_makhani": ["DAIRY", "BUTTER"],
    "butter_naan": ["DAIRY", "GLUTEN"],
    "kheer": ["DAIRY", "LACTOSE"],
    "kulfi": ["DAIRY", "LACTOSE"],
    "lassi": ["DAIRY", "LACTOSE"],
    "ras_malai": ["DAIRY", "LACTOSE"],
    "gulab_jamun": ["DAIRY", "GLUTEN"],
    "ghevar": ["DAIRY", "GLUTEN"],
    # Gluten / Wheat
    "chapati": ["GLUTEN", "WHEAT"],
    "chole_bhature": ["GLUTEN", "WHEAT"],
    "kaathi_rolls": ["GLUTEN", "WHEAT"],
    "momos": ["GLUTEN", "WHEAT"],
    "samosa": ["GLUTEN", "WHEAT"],
    "burger": ["GLUTEN", "WHEAT"],
    "pizza": ["GLUTEN", "DAIRY", "WHEAT"],
    # Non-Vegetarian
    "chicken_curry": ["POULTRY", "NON_VEG"],
    "fish_curry": ["SEAFOOD", "FISH", "NON_VEG"],
    "mutton_curry": ["RED_MEAT", "NON_VEG"],
    # Tree Nuts
    "modak": ["COCONUT", "NUTS"],
    "coconut_chutney": ["COCONUT"]
}


class NutriSenseInferenceEngine:
    """
    End-to-end inference engine combining computer vision, nutrition knowledge base,
    interactive ambiguous-subtype confirmation, and clinical health scoring.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 1. Load Vision Model Checkpoint
        if checkpoint_path is None:
            checkpoint_path = MODELS_DIR / "mobilenetv2_food43_best.pth"
        self.checkpoint_path = Path(checkpoint_path)

        self._load_knowledge_bases()
        self._build_and_load_model()

        # Image Preprocessing Transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def _load_knowledge_bases(self):
        """Loads master nutrition KB, subtypes registry, composite recipes, and serving weights."""
        # Class-to-idx
        c2i_path = MODELS_DIR / "class_to_idx.json"
        if not c2i_path.exists():
            raise FileNotFoundError(f"Missing {c2i_path}")
        with open(c2i_path, "r", encoding="utf-8") as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Master Nutrition KB
        kb_csv = FEATURE_DIR / "master_food_nutrition_kb.csv"
        if not kb_csv.exists():
            raise FileNotFoundError(f"Missing {kb_csv}")
        self.master_kb_df = pd.read_csv(kb_csv)
        self.master_kb_map = {row["class_name"]: row.to_dict() for _, row in self.master_kb_df.iterrows()}

        # Ambiguous Subtypes Registry
        subtypes_json = FEATURE_DIR / "ambiguous_subtypes_registry.json"
        if subtypes_json.exists():
            with open(subtypes_json, "r", encoding="utf-8") as f:
                raw_subs = json.load(f)
                self.subtypes_registry = {}
                for item in raw_subs:
                    self.subtypes_registry[item["parent_class"]] = item
                    for alias in item.get("aliases", []):
                        self.subtypes_registry[alias] = item
        else:
            self.subtypes_registry = {}

        # Composite Recipes Nutrition
        comp_json = FEATURE_DIR / "composite_recipes_nutrition.json"
        if comp_json.exists():
            with open(comp_json, "r", encoding="utf-8") as f:
                raw_comp = json.load(f)
                self.composite_recipes = {item["class_name"]: item for item in raw_comp}
        else:
            self.composite_recipes = {}

        # Standard Serving Size Weights
        serv_csv = NUTRITION_DIR / "serving_size_weights.csv"
        if serv_csv.exists():
            self.serving_weights_df = pd.read_csv(serv_csv)
            self.serving_weights_map = {
                row["food_name"]: float(row["serving_g"]) for _, row in self.serving_weights_df.iterrows()
            }
        else:
            self.serving_weights_map = {}

    def _build_and_load_model(self):
        """Builds MobileNetV2 architecture and loads trained weights if checkpoint exists."""
        self.model = mobilenet_v2(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, len(self.class_to_idx))
        )

        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.is_trained = True
            print(f"Loaded model checkpoint from {self.checkpoint_path} (Epoch {checkpoint.get('epoch', 'N/A')})")
        else:
            self.is_trained = False
            print(f"WARNING: Checkpoint {self.checkpoint_path} not found. Running with uninitialized weights.")

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_input: Union[str, Path, Image.Image, torch.Tensor]) -> torch.Tensor:
        """Converts diverse image inputs into standardized torch tensor (1, 3, 224, 224)."""
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if not p.exists():
                raise FileNotFoundError(f"Image not found at {p}")
            img = Image.open(p).convert("RGB")
            tensor = self.transform(img).unsqueeze(0)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
            tensor = self.transform(img).unsqueeze(0)
        elif isinstance(image_input, torch.Tensor):
            if image_input.dim() == 3:
                tensor = image_input.unsqueeze(0)
            elif image_input.dim() == 4:
                tensor = image_input
            else:
                raise ValueError(f"Invalid tensor shape: {image_input.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return tensor.to(self.device)

    def predict_vision_probabilities(self, image_tensor: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """Runs vision model inference and returns top-k predicted classes with confidences."""
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(self.class_to_idx)))
        results = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.idx_to_class[idx.item()]
            kb_meta = self.master_kb_map.get(class_name, {})
            results.append({
                "class_name": class_name,
                "display_name": kb_meta.get("display_name", class_name.replace("_", " ").title()),
                "confidence": round(float(prob.item()), 4),
                "class_idx": int(idx.item())
            })
        return results

    def analyze_food(
        self,
        image_input: Union[str, Path, Image.Image, torch.Tensor],
        top_k: int = 5,
        user_profile: str = "adult_male_sedentary",
        portion_g: Optional[float] = None,
        selected_subtype_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end food analysis:
        1. Predicts vision class and top-k alternatives.
        2. Retrieves nutrition from Master KB or Subtype Registry.
        3. Flags ambiguous subtypes for user confirmation if required.
        4. Scales nutrition to portion size.
        5. Evaluates clinical health suitability score and dietary alerts.
        """
        # 1. Vision Prediction
        tensor = self.preprocess_image(image_input)
        top_k_preds = self.predict_vision_probabilities(tensor, top_k=top_k)
        primary = top_k_preds[0]
        class_name = primary["class_name"]

        # 2. Nutrition Lookup from Master KB
        base_kb = self.master_kb_map.get(class_name, {})
        match_type = base_kb.get("match_type", "UNKNOWN")
        is_ambiguous = (match_type == "AMBIGUOUS_SUBTYPE") or bool(base_kb.get("has_subtypes", False))

        # 3. Subtype Handling
        subtype_info = None
        chosen_profile = base_kb.copy()
        requires_confirmation = False

        if is_ambiguous and class_name in self.subtypes_registry:
            family_meta = self.subtypes_registry[class_name]
            subtypes_list = family_meta.get("subtypes", [])
            default_sub = next((s for s in subtypes_list if s.get("is_default")), subtypes_list[0])

            if selected_subtype_id:
                # User has confirmed specific subtype
                match_sub = next((s for s in subtypes_list if s.get("subtype_id") == selected_subtype_id), None)
                if match_sub:
                    chosen_profile.update(match_sub)
                    chosen_profile["selected_subtype"] = selected_subtype_id
                    requires_confirmation = False
                else:
                    chosen_profile.update(default_sub)
                    requires_confirmation = True
            else:
                # Default profile with confirmation prompt
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

        # 4. Standard Serving Size & Portion Scaling
        default_serving_g = float(chosen_profile.get("serving_size_g", 100.0))
        if pd.isna(default_serving_g) or default_serving_g <= 0:
            default_serving_g = self.serving_weights_map.get(class_name, 100.0)

        actual_portion_g = float(portion_g) if portion_g is not None and portion_g > 0 else default_serving_g
        scale = actual_portion_g / default_serving_g if default_serving_g > 0 else 1.0

        # Scale nutritional values
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

        # 5. Missing Micronutrient Transparency Check
        missing_micros = []
        for micro in ["calcium_mg", "iron_mg", "vitamin_c_mg", "folate_ug"]:
            if scaled_nutrition[micro] is None or pd.isna(chosen_profile.get(micro)):
                missing_micros.append(micro)

        # 6. Safety Flags Preservation
        sodium_flag = str(chosen_profile.get("sodium_flag", "ok"))
        calorie_flag = str(chosen_profile.get("calorie_flag", "ok"))

        # 7. Evidence-based Health Suitability Scoring
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

        # 8. Clinical and Dietary Alerts
        dietary_alerts = []
        if sodium_flag == "high_sodium_unverified":
            dietary_alerts.append({
                "severity": "WARNING",
                "type": "SODIUM_DATA_LIMITATION",
                "message": f"Source sodium is extremely high ({chosen_profile.get('sodium_mg')} mg/serving) and unverified in source data. Exercise caution."
            })

        if calorie_flag == "discrepant_atwater_over_25pct":
            dietary_alerts.append({
                "severity": "INFO",
                "type": "CALORIE_DISCREPANCY_FLAG",
                "message": "Dish has >25% discrepancy between reported calories and Atwater 4-4-9 sum in source literature."
            })

        if user_profile == "hypertensive_adult" and (scaled_nutrition["sodium_mg"] or 0) > 500.0:
            dietary_alerts.append({
                "severity": "HIGH",
                "type": "HYPERTENSION_WARNING",
                "message": "High sodium portion exceeds DASH recommendation for hypertensive individuals."
            })

        if user_profile == "diabetic_adult" and (scaled_nutrition["carbs_g"] or 0) > 35.0:
            dietary_alerts.append({
                "severity": "MODERATE",
                "type": "DIABETES_CARB_ALERT",
                "message": "High carbohydrate content may cause glycemic spike. Consider pairing with fiber/salad."
            })

        # Allergens
        detected_allergens = ALLERGEN_MAP.get(class_name, [])

        # Composite recipe context
        composite_info = None
        if match_type == "COMPOSITE_RECIPE" and class_name in self.composite_recipes:
            comp_data = self.composite_recipes[class_name]
            ingredients_list = [
                ing.get("ingredient", "") for ing in comp_data.get("ingredients_breakdown", [])
            ]
            composite_info = {
                "source_type": comp_data.get("source_type", "COMPOSITE_CALCULATED"),
                "sample_ingredients": ingredients_list,
                "culinary_notes": comp_data.get("culinary_notes", "")
            }

        return {
            "prediction": {
                "class_name": class_name,
                "display_name": chosen_profile.get("display_name", class_name.replace("_", " ").title()),
                "confidence": primary["confidence"],
                "top_k": top_k_preds
            },
            "nutrition": scaled_nutrition,
            "metadata": {
                "match_type": match_type,
                "data_source": chosen_profile.get("data_source", "master_food_nutrition_kb.csv"),
                "data_quality_note": chosen_profile.get("data_quality_note", ""),
                "sodium_flag": sodium_flag,
                "calorie_flag": calorie_flag,
                "missing_micronutrients": missing_micros
            },
            "ambiguous_subtype": subtype_info,
            "composite_recipe": composite_info,
            "health_evaluation": {
                "user_profile": user_profile,
                "suitability_score": health_score,
                "verdict": verdict,
                "dietary_alerts": dietary_alerts,
                "detected_allergens": detected_allergens
            }
        }
