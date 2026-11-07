# Phase 7: Nutrition Analysis Module — Comprehensive Report

## 1. Overview & Objectives

Phase 7 connects the confirmed food item to the cleaned nutrition database (Master Food Nutrition KB, Unified Food Lookup with 1,556 items, IFCT 2017 composition tables, and standard serving weights reference).

### Core Requirements Implemented:
1. **Confirmed Food Connection**: Connects vision predictions, culinary subtypes, or manually selected dishes to exact laboratory nutrition profiles.
2. **Portion Size Selection**: Allows users to select or enter a portion size in grams or choose a defined serving unit (piece, 2 pieces, half piece, katori, plate, bowl, cup, standard serving).
3. **Standard Formula Scaling**:
   $$\text{Nutrient Amount} = \frac{\text{Portion Grams}}{100.0} \times \text{Nutrient Value per } 100\,\text{g}$$
4. **Complete Nutrient Spectrum**: Calculates calories (`kcal`), protein (`g`), carbohydrates (`g`), fat (`g`), fibre (`g`), and available micronutrients: sodium (`mg`), calcium (`mg`), iron (`mg`), vitamin C (`mg`), and folate (`mcg`).
5. **Missing Nutrition Data Integrity**: Never imputes missing nutrient values to zero. Missing fields remain `None` (`null`) and are explicitly reported in `missing_nutrients` with reason and laboratory source.
6. **Data Source & Reference Basis Transparency**: Reports source composition table (IFCT 2017, Composite Recipe, Indian Food Nutrition Processed) and basis ("Recorded per 100g edible portion", "Per reference serving of X grams").
7. **Manual Food Search Fallback**: Fast fuzzy/token-based search across all 1,556+ nutrition records when image recognition fails or user searches directly.
8. **Explicit Non-Claim Disclaimer**: Prominently states that visual image recognition identifies dish category only, and portion weight must be confirmed by the user.

---

## 2. Mathematical Formulation & Portion Conversion

### A. Normalized Baseline per 100g
For any food record with reference serving $S_{\text{ref}}$ (in grams) and declared nutrient $N_{\text{ref}}$:
$$N_{100} = N_{\text{ref}} \times \left(\frac{100.0}{S_{\text{ref}}}\right)$$

### B. User-Selected Portion Scaling
Given user portion $P$ (in grams):
$$N_{\text{portion}} = \frac{P}{100.0} \times N_{100} = P \times \left(\frac{N_{\text{ref}}}{S_{\text{ref}}}\right)$$

### C. Defined Serving Conversions (`standard_serving_weights.csv`)
| Dish Category | Defined Serving Unit | Reference Weight ($g$) | Basis Source |
| :--- | :--- | :---: | :--- |
| **Samosa** | 1 Piece | 65.0 | Approximate standard |
| **Samosa** | 2 Pieces | 130.0 | Calculated ($2 \times 65\text{g}$) |
| **Samosa** | 0.5 Piece | 32.5 | Calculated ($0.5 \times 65\text{g}$) |
| **Idli** | 1 Piece | 80.0 | ICMR-NIN reference |
| **Dosa** | 1 Dosa | 80.0 | Approximate standard |
| **Chapati** | 1 Chapati | 35.0 | ICMR-NIN reference |
| **Roti** | 1 Roti | 30.0 | Approximate standard |
| **Biryani** | 1 Plate | 250.0 | Standard restaurant plate |
| **Dal** | 1 Katori | 150.0 | Standard 150ml katori |
| **Sabzi / Curry** | 1 Katori | 120.0 | Standard vegetable katori |
| **Poha / Upma** | 1 Plate | 120.0 - 150.0 | Standard breakfast plate |

---

## 3. Architecture & API Endpoints

### A. Core Engine (`src/nutrition_analyzer.py`)
- `NutritionAnalyzer`: Central engine handling database indexing, fuzzy search, defined servings, and portion calculations.
- Functions exposed: `lookup_food()`, `convert_portion_to_grams()`, `calculate_nutrients()`, `search_foods()`, `get_available_servings()`.

### B. Flask API Endpoints (`app.py`)
1. `GET /api/nutrition/search?q={query}&limit={15}`:
   - Full-text search across 59 Master KB dishes and 1,556 Unified Lookup foods.
2. `GET /api/nutrition/servings?food_name={name}`:
   - Returns custom grams, 100g reference, and all dish-specific defined servings.
3. `POST /api/nutrition/analyze`:
   - Accepts `{ food_name, portion_value, unit, subtype_id, user_profile }`.
   - Returns complete scaled nutrient payload, missing nutrient list, data source, health evaluation score, and non-claim disclaimer.

---

## 4. Test Suite & Verification Results

| Test Suite | Scope | Total Tests | Passed | Failed |
| :--- | :--- | :---: | :---: | :---: |
| [`tests/test_nutrition_analysis.py`](file:///C:/Samsung_Project/tests/test_nutrition_analysis.py) | **Phase 7: Nutrition Analysis Module** | **63** | **63** | **0** |
| [`tests/test_label_scanner.py`](file:///C:/Samsung_Project/tests/test_label_scanner.py) | Packaged Food Label Scanner | 43 | 43 | 0 |
| [`tests/test_web_application.py`](file:///C:/Samsung_Project/tests/test_web_application.py) | Phase 6 Web Application & Integration | 48 | 48 | 0 |
| [`tests/test_model_training_and_inference.py`](file:///C:/Samsung_Project/tests/test_model_training_and_inference.py) | Phase 5 Model Training & Inference | 52 | 52 | 0 |
| [`tests/test_feature_engineering.py`](file:///C:/Samsung_Project/tests/test_feature_engineering.py) | Phase 4 Feature Engineering | 87 | 87 | 0 |
| [`tests/test_eda.py`](file:///C:/Samsung_Project/tests/test_eda.py) | Phase 3 Exploratory Data Analysis | 57 | 57 | 0 |
| [`tests/test_phase2_review.py`](file:///C:/Samsung_Project/tests/test_phase2_review.py) | Phase 2 Review & Anomaly Remediation | 45 | 45 | 0 |
| [`tests/test_preprocessing.py`](file:///C:/Samsung_Project/tests/test_preprocessing.py) | Phase 1 Data Audit & Preprocessing | 60 | 60 | 0 |
| **All Project Tests Combined** | **Phases 1–7 Full Regression Suite** | **455** | **455** | **0** |

---

## 5. Phase Completion Status

All requirements for Phase 7: Nutrition Analysis Module have been successfully implemented, verified on live server, and backed by 100% passing automated test suites.
