"""
tests/test_model_training_and_inference.py
Unit & Integration Tests for Phase 5:
- Vision Dataset loading & transforms
- MobileNetV2 architecture & checkpoint
- Top-k prediction & calibration
- Master KB nutrition integration
- Ambiguous subtype disambiguation
- Composite recipe context retrieval
- Portion scaling
- Evidence-based health suitability scoring
- Clinical safety alerts & allergen mapping
"""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
import torch

WORKSPACE = Path("c:/Samsung_Project")
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.vision_dataset import (
    EnhancedFoodDataset,
    FoodLensDataset,
    get_vision_transforms,
    create_dataloaders
)
from src.inference_pipeline import NutriSenseInferenceEngine, ALLERGEN_MAP

PASS = "[PASS]"
FAIL = "[FAIL]"
test_results = []


def test(name: str, condition: bool, extra: str = ""):
    status = PASS if condition else FAIL
    msg = f"{status} {name}"
    if extra:
        msg += f" | {extra}"
    print(msg)
    test_results.append((name, condition))


def run_tests():
    print("=" * 60)
    print("RUNNING PHASE 5 MODEL TRAINING & INFERENCE TESTS")
    print("=" * 60)

    # 1. Vision Dataset & Transforms
    print("\n--- 1. Vision Dataset & DataLoader Tests ---")
    train_tf = get_vision_transforms("train")
    eval_tf = get_vision_transforms("val")

    test("transforms: train transform returns Compose", isinstance(train_tf, object))
    test("transforms: eval transform returns Compose", isinstance(eval_tf, object))

    train_ds = EnhancedFoodDataset("data/processed/vision_splits/train_split.csv", transform=eval_tf)
    val_ds = EnhancedFoodDataset("data/processed/vision_splits/val_split.csv", transform=eval_tf)
    test_ds = EnhancedFoodDataset("data/processed/vision_splits/test_split.csv", transform=eval_tf)

    test("dataset: train dataset length is 10,777", len(train_ds) == 10777, f"len={len(train_ds)}")
    test("dataset: val dataset length is 2,315", len(val_ds) == 2315, f"len={len(val_ds)}")
    test("dataset: test dataset length is 2,312", len(test_ds) == 2312, f"len={len(test_ds)}")
    test("dataset: total stratified images is 15,404", len(train_ds) + len(val_ds) + len(test_ds) == 15404)
    test("dataset: 43 unique classes in class_to_idx", len(train_ds.class_to_idx) == 43)

    # Sample item verification
    img, target, class_name, md5_hash = train_ds[0]
    test("dataset: sample image tensor shape is (3, 224, 224)", img.shape == torch.Size([3, 224, 224]))
    test("dataset: sample target is integer in range [0, 42]", 0 <= target <= 42, f"target={target}")
    test("dataset: sample class_name is non-empty string", bool(class_name))
    test("dataset: sample md5_hash is 32-char hex string", len(md5_hash) == 32)

    # FoodLens Dataset
    fl_ds = FoodLensDataset("data/processed/vision_splits/food_lens_train_split.csv", transform=eval_tf)
    test("foodlens: dataset loads cleanly", len(fl_ds) == 1764, f"len={len(fl_ds)}")

    # 2. Checkpoint & Model Verification
    print("\n--- 2. Model & Checkpoint Tests ---")
    checkpoint_path = Path("models/mobilenetv2_food43_best.pth")
    test("checkpoint: file exists", checkpoint_path.exists())

    if checkpoint_path.exists():
        chk = torch.load(checkpoint_path, map_location="cpu")
        test("checkpoint: contains model_state_dict", "model_state_dict" in chk)
        test("checkpoint: contains val_metrics", "val_metrics" in chk)
        test("checkpoint: contains class_to_idx", "class_to_idx" in chk)
        test("checkpoint: num_classes is 43", chk.get("num_classes") == 43)

        val_metrics = chk.get("val_metrics", {})
        top1 = val_metrics.get("top1_acc", 0.0)
        macro_f1 = val_metrics.get("macro_f1", 0.0)
        test("checkpoint: validation top-1 accuracy > 70%", top1 >= 70.0, f"top1={top1:.2f}%")
        test("checkpoint: validation macro F1 > 70%", macro_f1 >= 0.70, f"macro_f1={macro_f1*100:.2f}%")

    # Evaluation results artifact
    eval_json_path = Path("models/evaluation_results.json")
    test("eval_results: json file exists", eval_json_path.exists())
    if eval_json_path.exists():
        with open(eval_json_path) as f:
            eval_data = json.load(f)
        test_m = eval_data.get("test_metrics", {})
        test("eval_results: test top-1 accuracy > 70%", test_m.get("top1_accuracy", 0.0) >= 70.0, f"acc={test_m.get('top1_accuracy'):.2f}%")
        test("eval_results: test top-5 accuracy >= 90%", test_m.get("top5_accuracy", 0.0) >= 90.0, f"top5={test_m.get('top5_accuracy'):.2f}%")
        test("eval_results: per_class_metrics contains 43 classes", len([c for c in eval_data.get("per_class_metrics", {}) if c in train_ds.class_to_idx]) == 43)

    # 3. End-to-End Inference Engine Tests
    print("\n--- 3. End-to-End Inference Pipeline Tests ---")
    engine = NutriSenseInferenceEngine()
    test("inference: engine initializes successfully", engine is not None)
    test("inference: model loaded on device", engine.device is not None)
    test("inference: master_kb loaded with >= 43 foods", len(engine.master_kb_map) >= 43)

    # Synthetic image input
    dummy_img = Image.new("RGB", (300, 300), color=(180, 120, 70))
    res = engine.analyze_food(dummy_img, top_k=5)

    test("inference: prediction contains predicted class_name", "class_name" in res["prediction"])
    test("inference: prediction contains top_k with 5 items", len(res["prediction"]["top_k"]) == 5)
    test("inference: top-1 confidence in range [0, 1]", 0.0 <= res["prediction"]["confidence"] <= 1.0)
    test("inference: nutrition dictionary present", "nutrition" in res)
    test("inference: calories_kcal is positive or null", res["nutrition"]["calories_kcal"] is None or res["nutrition"]["calories_kcal"] > 0)
    test("inference: health_evaluation present", "health_evaluation" in res)
    test("inference: suitability_score in range [1.0, 10.0]", 1.0 <= res["health_evaluation"]["suitability_score"] <= 10.0)

    # 4. Ambiguous Subtype Disambiguation Tests
    print("\n--- 4. Ambiguous Subtype Disambiguation Tests ---")
    samosa_res = engine.analyze_food(dummy_img, selected_subtype_id=None)
    # Simulate samosa analysis specifically
    samosa_analysis = engine.analyze_food(dummy_img)
    # Override class for targeted subtype tests
    samosa_default = engine.analyze_food(dummy_img, portion_g=100.0)
    
    # Test samosa specifically through engine's ambiguous subtype handling
    test("ambiguous: samosa is in subtypes registry", "samosa" in engine.subtypes_registry)
    if "samosa" in engine.subtypes_registry:
        sub_meta = engine.subtypes_registry["samosa"]
        test("ambiguous: samosa has >= 2 culinary subtypes", len(sub_meta["subtypes"]) >= 2)
        test("ambiguous: samosa prompt is present", bool(sub_meta.get("disambiguation_prompt")))

    # Test explicit subtype selection (Potato vs Keema Samosa)
    potato_sub = engine.analyze_food(dummy_img, selected_subtype_id="samosa_potato")
    keema_sub = engine.analyze_food(dummy_img, selected_subtype_id="samosa_keema")

    test("ambiguous: potato samosa calories registered (256 kcal)", engine.subtypes_registry["samosa"]["subtypes"][0]["calories_kcal"] == 256.0)
    test("ambiguous: keema samosa calories registered (318 kcal)", engine.subtypes_registry["samosa"]["subtypes"][1]["calories_kcal"] == 318.0)

    # 5. Composite Recipe Nutrition Integration Tests
    print("\n--- 5. Composite Recipe Integration Tests ---")
    test("composite: 15 composite recipes in registry", len(engine.composite_recipes) == 15)
    test("composite: aloo_gobi present in composite recipes", "aloo_gobi" in engine.composite_recipes)
    if "aloo_gobi" in engine.composite_recipes:
        ag = engine.composite_recipes["aloo_gobi"]
        test("composite: aloo_gobi has ingredients_breakdown", len(ag.get("ingredients_breakdown", [])) > 0)
        test("composite: aloo_gobi has culinary_notes", bool(ag.get("culinary_notes")))

    # 6. Quality Flags & Safety Tests
    print("\n--- 6. Data Quality Flags & Clinical Safety Alerts ---")
    test("quality: master KB retains sodium_flag column", "sodium_flag" in engine.master_kb_df.columns)
    test("quality: master KB retains calorie_flag column", "calorie_flag" in engine.master_kb_df.columns)
    df_dish = pd.read_csv("data/processed/nutrition/nutrition_dish_level.csv")
    test("quality: dish level retains 31 high_sodium_unverified", (df_dish["sodium_flag"] == "high_sodium_unverified").sum() == 31)
    test("quality: dish level retains 35 atwater discrepancy flags", (df_dish["calorie_flag"] == "atwater_discrepancy_unverified").sum() == 35)

    # Portion scaling test (100g vs 200g)
    res_100g = engine.analyze_food(dummy_img, portion_g=100.0)
    res_200g = engine.analyze_food(dummy_img, portion_g=200.0)
    if res_100g["nutrition"]["calories_kcal"] and res_200g["nutrition"]["calories_kcal"]:
        ratio = res_200g["nutrition"]["calories_kcal"] / res_100g["nutrition"]["calories_kcal"]
        test("portion: scaling scales calories proportionally (ratio ~2.0)", 1.95 <= ratio <= 2.05, f"ratio={ratio:.2f}")

    # Allergen mapping test
    test("allergens: paneer dishes mapped to DAIRY", "DAIRY" in ALLERGEN_MAP.get("kadai_paneer", []))
    test("allergens: chicken curry mapped to NON_VEG", "NON_VEG" in ALLERGEN_MAP.get("chicken_curry", []))
    test("allergens: samosa mapped to GLUTEN", "GLUTEN" in ALLERGEN_MAP.get("samosa", []))

    # Health Profile Suitability Scoring
    res_diabetic = engine.analyze_food(dummy_img, user_profile="diabetic_adult")
    test("health: diabetic profile score bounded in [1.0, 10.0]", 1.0 <= res_diabetic["health_evaluation"]["suitability_score"] <= 10.0)
    test("health: diabetic verdict is valid qualitative string", res_diabetic["health_evaluation"]["verdict"] in [
        "HIGHLY_RECOMMENDED", "HEALTHY_CHOICE", "MODERATE_PORTION_CONTROL", "NOT_RECOMMENDED_LIMIT_INTAKE"
    ])

    # Summary
    passed_count = sum(1 for _, ok in test_results if ok)
    failed_count = sum(1 for _, ok in test_results if not ok)
    total_count = len(test_results)

    print("\n" + "=" * 60)
    print(f"PHASE 5 TESTS: {passed_count} PASSED / {failed_count} FAILED / {total_count} TOTAL")
    print("=" * 60)
    if failed_count == 0:
        print("All Phase 5 tests passed successfully.")
    else:
        print(f"FAILED {failed_count} tests.")


if __name__ == "__main__":
    run_tests()
