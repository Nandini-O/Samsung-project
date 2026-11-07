# Phase 4: Feature Engineering and Dataset Preparation Audit Report — NutriSense AI

## 1. Executive Summary
Phase 4 (Feature Engineering and Dataset Preparation) has undergone an exhaustive audit, verification, and targeted review in the `c:\Samsung_Project` workspace. All raw datasets (`data/raw/`) remain preserved and untouched.

Key audit outcomes and pipeline confirmations:
1. **Dataset Count Discrepancy Resolved**: The difference of 2,521 images between `combined_vision_manifest.csv` (17,925 images) and the primary vision splits (15,404 images) was traced to the active subset of the FoodLens dataset. Dedicated split files are established for both benchmarks, and `combined_vision_manifest.csv` explicitly documents `benchmark_role` (`primary_43class_benchmark` vs. `auxiliary_transfer_foodlens`).
2. **FoodLens Image & Hash Count Verified**: The 2,521 active FoodLens images correspond to exactly **2,520 unique real MD5 hashes**. The single duplicate hash within active FoodLens is an identical image pair located entirely within the `train` split for `chappati` (`BUM71V45K4BA.jpg` and `YVRKYNVI4KFD.jpg`, MD5: `e3227974416659a054d5123e54769b60`). Cross-split leakage remains strictly **0%**.
3. **Zero Cross-Split & Cross-Dataset Leakage**: 0% hash leakage across train, validation, and test splits within Enhanced Indian Food (15,404 images) and FoodLens (2,521 active images), with 0 hash collision between the two datasets.
4. **Composite Nutrition Profiles Grounded**: All 15 previously unmatched image classes have verified ingredient-based profiles satisfying the Atwater thermodynamic consistency check ($\le 3.43\%$ maximum error, well below the 5% threshold).
5. **Ambiguous Subtypes Registry Audited**: 36 discrete culinary sub-profiles across 14 parent families (covering all 18 ambiguous image classes from Phase 3 EDA) are catalogued with clinical baselines and UI disambiguation prompts.
6. **Master Nutrition Knowledge Base Audited**: Exactly 59 entries providing 100% coverage of the 43 vision classes, preserving sodium and calorie flags and retaining missing micronutrients as `NaN`.
7. **Scoring Engine & Missing Nutrient Handling Verified**: Scoring engine evaluated against ICMR-NIN 2020 RDA benchmarks (1.0–10.0 scale). Tested and verified to safely handle foods with missing/NaN micronutrients without failure.
8. **Recipe Feature Store Audited**: 5,381 recipes tagged with cooking complexity and ingredient tokens, with explicit transparency caveats regarding categorical health tags.
9. **Test Verification**: The test suite stands at **87 / 87 PASSED** in [`test_feature_engineering.py`](file:///c:/Samsung_Project/tests/test_feature_engineering.py) and **249 / 249 PASSED** across all project test suites.

---

## 2. Task 1: Dataset Count & Hash Discrepancy Resolution

### 2.1 The Initial Discrepancy
- Combined vision manifest: 17,925 images
- Training split: 10,777 images
- Validation split: 2,315 images
- Test split: 2,312 images
- Sum of the three splits: 15,404 images
- Discrepancy: $17,925 - 15,404 = 2,521$ images

### 2.2 Root Cause Investigation & Tracing
1. **Source Tracking**:
   - `data/processed/images/enhanced_indian_food_stratified_manifest.csv`: Contains exactly **15,404 images** across 43 classes. This was split cleanly into 10,777 train (70.0%), 2,315 validation (15.0%), and 2,312 test (15.0%) images.
   - `data/processed/images/food_lens_manifest.csv`: Contains **2,523 total raw images** across 20 classes. In Phase 2, exactly **2 cross-split duplicate images** were identified and flagged with `exclude_leakage = True` (`02a5cf3882.jpg` in val, `d34b35e26b.jpg` in test, duplicating train images). This left exactly **2,521 active, clean FoodLens images**.
   - $15,404 + 2,521 = 17,925$ total active images across both datasets.
2. **FoodLens Image vs. Unique Hash Count Audit**:
   - Total active FoodLens images: **2,521**
   - Unique real MD5 hashes of active files on disk: **2,520**
   - Root Cause of Difference ($2,521 - 2,520 = 1$): Exactly one duplicate image pair exists within the `train` split for the `chappati` class:
     - File 1: `BUM71V45K4BA.jpg` (train)
     - File 2: `YVRKYNVI4KFD.jpg` (train)
     - Both share MD5: `e3227974416659a054d5123e54769b60`
     - Because both files reside in the same split (`train`), there is zero cross-split data leakage.
3. **Cross-Split and Cross-Dataset Leakage Verification**:
   - **Enhanced Cross-Split Leakage**: 0 shared hashes between train, validation, and test.
   - **FoodLens Active Cross-Split Leakage**: 0 shared hashes between train (1,763 unique hashes), val (375 unique hashes), and test (382 unique hashes).
   - **Cross-Dataset Collision**: The 15,311 unique MD5 hashes in Enhanced Indian Food and the 2,520 unique real hashes in FoodLens share **0 hashes** in common (0 collision).

### 2.3 Manifest Architecture & Inclusion Policy
- **Primary 43-Class Benchmark (Enhanced Indian Food)**:
  - [`train_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/train_split.csv): 10,777 images (70.0%)
  - [`val_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/val_split.csv): 2,315 images (15.0%)
  - [`test_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/test_split.csv): 2,312 images (15.0%)
  - Total: 15,404 images across all 43 classes.
- **Auxiliary Transfer Benchmark (FoodLens)**:
  - [`food_lens_train_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/food_lens_train_split.csv): 1,764 images (1,763 unique hashes)
  - [`food_lens_val_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/food_lens_val_split.csv): 375 images
  - [`food_lens_test_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/food_lens_test_split.csv): 382 images
  - Total: 2,521 active images (2,520 unique hashes).
- **Master Manifest**:
  - [`combined_vision_manifest.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/combined_vision_manifest.csv): 17,925 images with explicit `benchmark_role` tags and authentic file MD5 hashes.

---

## 3. Task 2: Composite Nutrition Profiles Audit

### 3.1 Traceable Sources & Serving Sizes for 15 Dishes
[`src/composite_nutrition_builder.py`](file:///c:/Samsung_Project/src/composite_nutrition_builder.py) synthesizes ingredient-level nutrition for the 15 dishes absent from `nutrition_dish_level.csv`:

| Class Name | Display Name | Serving Size (g) | Serving Unit | Calories (kcal) | Protein (g) | Fat (g) | Carbs (g) | Fiber (g) | Atwater Error (%) | Traceable Reference Source |
|---|---|---|---|---|---|---|---|---|---|---|
| `aloo_gobi` | Aloo Gobi | 150.0g | 1 katori / bowl | 112.0 | 2.7 | 5.4 | 14.0 | 2.8 | 3.04% | IFCT 2017: Potato (Solanum tuberosum) + Cauliflower |
| `aloo_masala` | Aloo Masala | 100.0g | 1 portion / filling | 105.0 | 2.2 | 4.2 | 15.5 | 2.1 | 3.43% | IFCT 2017: Boiled potato + Big onion + Green peas |
| `bhindi_masala` | Bhindi Masala | 120.0g | 1 katori / bowl | 115.0 | 2.5 | 8.2 | 8.6 | 3.8 | 2.78% | IFCT 2017: Okra (Abelmoschus esculentus) + Aromatics |
| `chole` | Chole | 150.0g | 1 katori / bowl | 195.0 | 7.5 | 8.5 | 22.5 | 5.5 | 0.77% | IFCT 2017: Kabuli chana B005 + Tomato-onion gravy |
| `chole_bhature` | Chole Bhature | 310.0g | 1 plate (150g chole + 2 bhature 160g) | 645.0 | 16.0 | 30.5 | 76.5 | 6.5 | 0.08% | Composite: IFCT Chole + Fried refined wheat flour A012 |
| `ghevar` | Ghevar | 80.0g | 1 medium piece | 365.0 | 3.2 | 20.5 | 42.0 | 0.8 | 0.08% | IFCT 2017: Refined flour A012 + Desi cow ghee + Sugar |
| `jalebi` | Jalebi | 100.0g | 1 serving (4-5 spirals) | 390.0 | 2.8 | 15.5 | 60.0 | 0.5 | 0.18% | IFCT 2017: Fermented maida batter + Cane sugar syrup |
| `kadai_paneer` | Kadai Paneer | 150.0g | 1 katori / bowl | 285.0 | 12.5 | 21.0 | 11.5 | 2.2 | 0.00% | IFCT 2017: Whole milk paneer + Green capsicum |
| `modak` | Modak | 70.0g | 2 pieces | 260.0 | 5.2 | 11.5 | 34.0 | 1.4 | 0.12% | Packaged Benchmark: Chitale Bandhu FSSAI standard |
| `momos` | Momos | 120.0g | 1 plate (6 steamed pieces) | 210.0 | 5.8 | 5.2 | 35.0 | 2.8 | 0.00% | Packaged Benchmark: Prasuma FSSAI nutrition label |
| `mutton_curry` | Mutton Curry | 180.0g | 1 bowl (70g meat + 110g gravy) | 310.0 | 19.5 | 21.5 | 8.5 | 1.8 | 1.45% | IFCT 2017: Goat meat D001 + Spiced aromatics gravy |
| `ras_malai` | Ras Malai | 120.0g | 2 pieces with cardamom milk | 245.0 | 7.8 | 11.2 | 28.5 | 0.5 | 0.41% | IFCT 2017: Whole milk chhena + Reduced whole milk |
| `bisibelebath` | Bisi Bele Bath | 200.0g | 1 plate / bowl | 260.0 | 6.8 | 8.5 | 39.5 | 4.2 | 0.65% | IFCT 2017: Milled rice + Toor dal B021 + Veg |
| `paniyaram` | Kuzhi Paniyaram | 100.0g | 4 pieces | 165.0 | 4.2 | 5.4 | 25.0 | 2.1 | 0.24% | IFCT 2017: Fermented rice-urad batter + Sunflower oil |
| `ven_pongal` | Ven Pongal | 180.0g | 1 plate / bowl | 305.0 | 7.2 | 14.8 | 36.5 | 2.8 | 0.98% | IFCT 2017: Milled rice + Moong dal B016 + Cow ghee |

### 3.2 Atwater Thermodynamic Consistency vs Nutritional Accuracy
- **Thermodynamic Formula**: $\text{Energy}_{\text{calc}} = (4 \times \text{Carbs}) + (4 \times \text{Protein}) + (9 \times \text{Fat})$.
- **Verification**: Maximum Atwater caloric discrepancy across all 15 dishes is **3.43%** (for `aloo_masala`), comfortably below the $\le 5\%$ scientific threshold.
- **Scientific Limitation**: Atwater consistency verifies that the macro-nutrient math is thermodynamically sound. It does **not** prove that every restaurant or home preparation has identical calories, as gravy composition and cooking oil absorption vary across culinary preparations.

---

## 4. Task 3: Ambiguous Subtypes Audit

[`src/ambiguous_subtypes_store.py`](file:///c:/Samsung_Project/src/ambiguous_subtypes_store.py) manages the 14 unique culinary parent families (covering all 18 ambiguous image classes from Phase 3 EDA):
- **36 Discrete Profiles**: Catalogued with distinct macronutrient densities (e.g., Potato Samosa 256 kcal, 3.5g protein vs. Keema Samosa 318 kcal, 12.8g protein; Sweet Lassi 185 kcal vs. Salted Chaas 65 kcal).
- **Default Baseline Profiles**: Every parent family has exactly 1 default baseline profile (`is_default = True`) ensuring safe fallbacks if the user does not disambiguate.
- **UI Prompt Strings**: Specific confirmation queries (e.g., *"Which type of biryani is on your plate?"*) and selection options are persisted in [`ambiguous_subtypes_registry.json`](file:///c:/Samsung_Project/data/processed/feature_store/ambiguous_subtypes_registry.json).

---

## 5. Task 4: Master Food & Nutrition Knowledge Base Audit

[`src/master_nutrition_kb.py`](file:///c:/Samsung_Project/src/master_nutrition_kb.py) provides the unified nutrition lookup:
- **Total Entries**: Exactly **59 master records** (26 Valid Direct, 18 Ambiguous Subtype Defaults, 15 Composite Recipes).
- **Class Coverage**: 43 out of 43 Enhanced Indian Food vision classes covered (100% coverage, 0 unmapped classes).
- **Preservation of Quality Flags**: All 31 high-sodium warnings (`sodium_flag == 'high_sodium_unverified'`) and 35 Atwater discrepancy warnings (`calorie_flag == 'discrepancy_gt_25pct'`) are preserved.
- **Missing Micronutrients Handled Transparently**: Unknown micronutrient values (`calcium_mg`, `iron_mg`, `vitamin_c_mg`, `folate_ug`) remain strictly `NaN` rather than being falsely imputed to zero.

---

## 6. Task 5 & 6: Vision Splits & Scoring Engine Audit

### 6.1 Vision Splits & Class Balancing
- **Class Weights**: Inverse-frequency weights in [`models/class_weights.json`](file:///c:/Samsung_Project/models/class_weights.json) range from 0.44 (`chole_bhature`, 814 images) to 2.76 (`paani_puri`, 130 images), matching the 43-class mapping in [`models/class_to_idx.json`](file:///c:/Samsung_Project/models/class_to_idx.json).
- **Leakage Integrity**: Verified 0 shared hashes across all splits.

### 6.2 Scoring Engine & Missing Nutrient Safety
- **Reference**: ICMR-NIN 2020 RDA benchmarks across 5 adult profiles on a 1.0–10.0 scale.
- **Safety Safeguards**:
  - Hypertensive profile: Dishes flagged `high_sodium_unverified` are capped at $\le 2.0 / 10.0$.
  - Diabetic profile: High-sugar desserts receive substantial deductions (`ghevar` scores 1.0, `jalebi` scores 1.3).
  - Missing Nutrient Resilience: Explicitly tested with all-NaN micronutrients; scoring safely falls back to standard macro-based evaluation without crashing.
- **Disclaimer**: Food suitability scores represent general nutritional guidance heuristics, not clinically validated diagnostic recommendations.

---

## 7. Task 7: Recipe Feature Store Audit

[`src/recipe_features.py`](file:///c:/Samsung_Project/src/recipe_features.py) organizes **5,381 recipes**:
- **Complexity Breakdown**: Easy: 1,592; Moderate: 2,811; Advanced: 978.
- **Ingredient Similarity**: Extracted stopword-cleaned `ingredient_tokens` for recommendation similarity matching in Phase 5.
- **Transparency Caveat**: Category tags (e.g., "diabetic_friendly") reflect source-provided labels and are not treated as verified clinical certifications.

---

## 8. Comprehensive Test Suite Results

All workspace tests were executed directly in Python:

| Test Suite | Purpose | Tests Passed | Status |
|---|---|---|---|
| [`test_preprocessing.py`](file:///c:/Samsung_Project/tests/test_preprocessing.py) | Raw cleaning & dataset formatting | **60 / 60** | **PASS** |
| [`test_phase2_review.py`](file:///c:/Samsung_Project/tests/test_phase2_review.py) | Quality flags, serving sizes, leakage | **45 / 45** | **PASS** |
| [`test_eda.py`](file:///c:/Samsung_Project/tests/test_eda.py) | Exploratory analysis, stratified splits, mapping | **57 / 57** | **PASS** |
| [`test_feature_engineering.py`](file:///c:/Samsung_Project/tests/test_feature_engineering.py) | Composites, subtypes, KB, splits, scoring, recipes, hash audit | **87 / 87** | **PASS** |
| **Workspace Total** | **End-to-End Phases 1–4 Verification** | **249 / 249** | **100% PASS** |

---

## 9. Modified and Generated Files Inventory

### Source Files Modified
- [`src/dataset_prep_vision.py`](file:///c:/Samsung_Project/src/dataset_prep_vision.py): Updated with genuine file MD5 hashing for FoodLens, explicit export of both benchmarks, and zero-leakage cross-checks.
- [`tests/test_feature_engineering.py`](file:///c:/Samsung_Project/tests/test_feature_engineering.py): Added targeted audit tests for FoodLens split tracking, 2,520 unique real hashes, within-train duplicate verification, missing nutrient resilience in scoring, and combined reconciliation (expanded from 67 to 87 tests).

### Artifacts and Generated Data Files
- [`data/processed/vision_splits/train_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/train_split.csv) (10,777 images)
- [`data/processed/vision_splits/val_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/val_split.csv) (2,315 images)
- [`data/processed/vision_splits/test_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/test_split.csv) (2,312 images)
- [`data/processed/vision_splits/food_lens_train_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/food_lens_train_split.csv) (1,764 images)
- [`data/processed/vision_splits/food_lens_val_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/food_lens_val_split.csv) (375 images)
- [`data/processed/vision_splits/food_lens_test_split.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/food_lens_test_split.csv) (382 images)
- [`data/processed/vision_splits/combined_vision_manifest.csv`](file:///c:/Samsung_Project/data/processed/vision_splits/combined_vision_manifest.csv) (17,925 images)
- [`models/class_to_idx.json`](file:///c:/Samsung_Project/models/class_to_idx.json) (43 classes)
- [`models/class_weights.json`](file:///c:/Samsung_Project/models/class_weights.json) (43 classes)
- [`data/processed/feature_store/composite_recipes_nutrition.csv`](file:///c:/Samsung_Project/data/processed/feature_store/composite_recipes_nutrition.csv) & [`.json`](file:///c:/Samsung_Project/data/processed/feature_store/composite_recipes_nutrition.json)
- [`data/processed/feature_store/ambiguous_subtypes_registry.csv`](file:///c:/Samsung_Project/data/processed/feature_store/ambiguous_subtypes_registry.csv) & [`.json`](file:///c:/Samsung_Project/data/processed/feature_store/ambiguous_subtypes_registry.json)
- [`data/processed/feature_store/master_food_nutrition_kb.csv`](file:///c:/Samsung_Project/data/processed/feature_store/master_food_nutrition_kb.csv) & [`.json`](file:///c:/Samsung_Project/data/processed/feature_store/master_food_nutrition_kb.json)
- [`data/processed/feature_store/food_suitability_benchmarks.csv`](file:///c:/Samsung_Project/data/processed/feature_store/food_suitability_benchmarks.csv) & [`user_health_profiles.json`](file:///c:/Samsung_Project/data/processed/feature_store/user_health_profiles.json)
- [`data/processed/feature_store/recipe_feature_store.csv`](file:///c:/Samsung_Project/data/processed/feature_store/recipe_feature_store.csv)
- [`reports/phase4_feature_engineering_report.md`](file:///c:/Samsung_Project/reports/phase4_feature_engineering_report.md)
