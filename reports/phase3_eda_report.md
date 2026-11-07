# Phase 3: Exploratory Data Analysis (EDA) Report (Corrected)
**Project:** NutriSense AI — AI-Based Personalized Dietary Assessment and Nutrient-Gap Recommendation System for Indian Meals  
**Status:** Phase 3 Corrections Completed  
**Generated/Updated:** 2026-10-25  

---

## Executive Summary
This report presents the verified and corrected Exploratory Data Analysis (EDA) conducted across the Phase 2 preprocessed datasets. The analysis spans across 14 processed datasets covering **17,927 food images**, **4,019 nutrition records**, and **5,381 Indian recipes**.

Key corrections, verified findings, data quality flags, and architectural solutions for Phase 4 are documented below.

---

## 1. Dataset Dimensions, Types & Missing Values

### Summary Inventory
Across all processed datasets, raw files were preserved without modification. Cleaned data integrity is verified.

| Dataset Name | Domain | Rows | Columns | % Missing Cells | Memory (MB) |
|---|---|---|---|---|---|
| `nutrition_dish_level` | nutrition | 1,014 | 18 | 0.90% | 0.54 MB |
| `nutrition_ifct_ingredients` | nutrition | 542 | 19 | 13.99% | 0.24 MB |
| `nutrition_daily_general` | nutrition | 587 | 15 | 0.00% | 0.26 MB |
| `nutrition_packaged_foods` | nutrition | 843 | 27 | 7.91% | 0.84 MB |
| `nutrition_unified_lookup` | nutrition | 1,556 | 13 | 12.06% | 0.66 MB |
| `standard_serving_weights` | nutrition | 21 | 3 | 0.00% | 0.0 MB |
| `food_lens_manifest` | images | 2,523 | 9 | 0.00% | 0.89 MB |
| `food_lens_class_manifest` | images | 60 | 3 | 0.00% | 0.01 MB |
| `enhanced_indian_food_label_manifest` | images | 43 | 5 | 0.00% | 0.0 MB |
| `enhanced_stratified_split_summary` | images | 43 | 8 | 0.00% | 0.0 MB |
| `recipes_indian_food_dataset` | recipes | 4,075 | 17 | 0.00% | 16.43 MB |
| `recipes_ndtv` | recipes | 746 | 14 | 0.47% | 0.69 MB |
| `recipes_indian_food_meta` | recipes | 255 | 12 | 2.84% | 0.14 MB |
| `recipes_regional` | recipes | 305 | 15 | 6.14% | 0.21 MB |

### Missing Value Handling & Micronutrient Integrity
- **Primary Macronutrients Completeness**: Zero missing values in primary macronutrients (`calories_kcal`, `carbs_g`, `protein_g`, `fat_g`) in `nutrition_dish_level.csv`.
- **Micronutrient Preservation**: Missing micronutrients (`vitamin_c_mg`, `folate_ug`, `thiamine_b1_mg`) remain explicitly cataloged as `NaN` (missing) rather than imputed as zero. This preserves the essential nutritional distinction between unmeasured laboratory parameters and verified zero values (e.g. pure oils genuinely containing 0 mg Vitamin C).
- **Packaged foods**: High completeness for mandatory FSSAI facts; voluntary items exhibit expected missingness.

Visual references:
- Dimensions Chart: `reports/figures/01_dataset_dimensions.png`
- Missing Rates Chart: `reports/figures/02_missing_values_summary.png`

---

## 2. Food-Image Class Distribution, Root Cause of Splits & Stratified Plan

### Enhanced Indian Food Classification (15,404 Images across 43 Classes)
- **Root Cause of 21 Train-Only Classes**: Upstream HuggingFace dataset creators took an existing 22-class benchmark (which had train, validation, and test shards) and appended 21 additional Indian dish classes (200 or 400 images each) strictly to the train parquet shards (`train-00000`, `train-00001`, `train-00002`) without generating validation or test splits.
- **Raw Parquet Leakage Detected**: 39 exact MD5 duplicate images cross splits in the upstream raw benchmark.
- **Reproducible Stratified Splitting Plan**:
  - Implemented in `src/image_split_analysis_and_plan.py` (seed=42).
  - Groups identical images strictly by `md5_hash` prior to splitting, guaranteeing **0 cross-split duplicates (0% leakage)**.
  - Stratifies all 43 classes into **70% train / 15% validation / 15% test**.
  - **All 43 classes can be split reliably**: Smallest class (`paani_puri`, 130 total images) provides 92 train, 19 val, and 19 test images, exceeding statistical evaluation requirements.
  - Manifest exported to: `data/processed/images/enhanced_indian_food_stratified_manifest.csv`.

Visual reference:
- Enhanced Distribution (Raw vs Stratified Plan): `reports/figures/03_enhanced_image_class_distribution.png`

---

## 3. FoodLens Dataset & Imbalance Management (2,521 Active Clean Images)

- **Leakage Prevention**: Exactly 2 evaluation cross-split duplicate images (`val` Dhokla and `test` Samosa) are flagged with `exclude_leakage=True`. Active clean images: **2,521**.
- **Verified Imbalance**: `samosa` accounts for **1070 images (42.44%)** of the entire active dataset, whereas the other 19 classes median at 75 images.
- **Concrete Phase 4 Training Solutions**:
  1. **Class-Weighted CrossEntropyLoss**: Weights inversely proportional to class frequencies ($w_c = \frac{N}{K \cdot N_c}$).
  2. **Focal Loss**: Focuses gradient updates on hard, non-samosa examples ($\gamma = 2.0$).
  3. **WeightedRandomSampler**: Generates balanced mini-batches in PyTorch DataLoader without discarding valid images or fabricating synthetic data.
  4. **Targeted Data Augmentation**: RandAugment applied dynamically to minority classes.

Visual reference:
- FoodLens Distribution: `reports/figures/04_food_lens_class_distribution.png`

---

## 4. Nutrition Distributions, Sodium Flags & Atwater Discrepancies

### Summary Statistics
Summary statistics were exported to `reports/eda_nutrition_stats.csv`:
- **Calories**: Median = 252.0 kcal/serving (IQR: 148 – 376 kcal).
- **Carbohydrates**: Median = 28.5 g (IQR: 14.2 – 48.0 g).
- **Protein**: Median = 6.8 g (IQR: 3.2 – 12.5 g).
- **Fat**: Median = 11.2 g (IQR: 4.8 – 21.0 g).

### Dual Data Quality Flags in `nutrition_dish_level.csv`
1. **Sodium Flag (`sodium_flag`)**:
   - Preserved **31 flagged records** (`high_sodium_unverified`, > 5,000 mg).
   - 983 normal records (`ok`, median: 98.1 mg).
2. **Calorie Atwater Flag (`calorie_flag`)**:
   - Flagged **35 records** (`atwater_discrepancy_unverified`) where reported calories diverge by > 25% from $(4 \times \text{C} + 4 \times \text{P} + 9 \times \text{F})$.
   - 979 normal records (`ok`).
3. **Systematic Overlap Discovery**:
   - **28 of the 35 Atwater discrepant dishes** are the exact same dishes flagged for extreme sodium (predominantly restaurant soups).
   - *Culinary Root Cause*: The soup sub-table in the source dataset suffers from systematic multi-portion recipe batch inflation (recording recipe-level sodium and inverted/unscaled calories).
   - Raw values are strictly retained; detailed explanations are appended to `data_quality_note`.

Visual references:
- Macronutrient Distributions: `reports/figures/05_nutrition_macro_distributions.png`
- Sodium Distribution & Flags: `reports/figures/06_sodium_distribution_and_flags.png`
- Atwater Identity & Flagged Discrepancies: `reports/figures/07_atwater_calorie_validation.png`

---

## 5. Recipe Coverage & Food Category Distribution

- **Total Corpus**: **5,381 recipes** across 4 curated sources.
- **Diet**: 70.0% Vegetarian, 11.8% Vegetarian High-Protein, 5.2% Non-Vegetarian.
- **Course**: Lunch (33.7%), Side Dish (17.2%), Snack (12.8%), Dinner (9.6%), Dessert (7.6%), South Indian Breakfast (5.6%).
- **Times**: Median Prep Time = 10 mins, Median Cook Time = 30 mins, Median Total Time = 40 mins.

Visual references:
- Diet & Course: `reports/figures/09_recipe_diet_and_course_distribution.png`
- Cooking Times: `reports/figures/10_recipe_cooking_times.png`

---

## 6. Image-to-Nutrition Mapping Assessment

Across all 61 unique image classes (43 Enhanced + 20 FoodLens, accounting for spelling aliases), mapping was evaluated against nutrition datasets:

| Match Category | Class Count | Percentage | Description & Examples | Phase 4 Resolution Strategy |
|---|---|---|---|---|
| **VALID_DIRECT** | **28 classes** | **45.9%** | Exact culinary and macronutrient match (e.g. *dhokla, idli, dal makhani, pav bhaji, masala dosa, poha, shahi paneer, chicken curry, pizza*). | Direct lookup from `nutrition_dish_level.csv` using standard serving weights. |
| **AMBIGUOUS_SUBTYPE** | **18 classes** | **29.5%** | Broad category with multiple sub-types having divergent nutritional profiles (e.g. *samosa* [potato vs keema], *biryani* [veg vs mutton], *chai* [sweet vs low-sugar], *lassi* [sweet vs salted]). | Present user disambiguation modal in UI (defaulting to Indian standard). |
| **UNMATCHED_GAP** | **15 classes** | **24.6%** | Prepared dish is absent from `dish_level.csv` (e.g. *chole bhature, jalebi, modak, momos, aloo gobi, ghevar*). | Synthesize composite recipe from IFCT ingredients or use verified packaged food references. |

Full mapping catalog exported to:
- CSV: `reports/image_to_nutrition_mapping.csv`
- JSON: `reports/image_to_nutrition_mapping.json`

Visual reference:
- Image Mapping Breakdown: `reports/figures/11_image_to_nutrition_mapping.png`

---

## 7. Verified Corrections vs Remaining Limitations

### Verified Corrections
1. **Root cause of 21 train-only classes identified**: Upstream benchmark compilation artifact.
2. **Stratified split plan prepared**: Deterministic seed=42, 70/15/15 ratio, 0% duplicate leakage, all 43 classes reliably splittable.
3. **FoodLens samosa imbalance verified (42.4%)**: Concrete class-weighted loss and sampling plans documented.
4. **Nutrition quality flags updated**: 31 high-sodium warnings preserved; 35 Atwater discrepancies flagged (`calorie_flag`), with 28 overlapping soups documented.
5. **Micronutrient missingness preserved**: `NaN` maintained, zero-filling avoided.
6. **Principled image-to-nutrition mapping established**: 28 direct matches, 18 ambiguous subtypes, 15 dataset gaps.

### Remaining Limitations to Address in Phase 4
1. **Unmatched Image Classes**: 15 classes need composite recipe definitions in Phase 4 feature engineering.
2. **Ambiguous Subtypes**: 18 classes require user disambiguation in application logic.
3. **External Archive**: `Food/dataset.tar.gz` remains unextracted (optional expansion to 104 classes).
