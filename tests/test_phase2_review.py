"""
tests/test_phase2_review.py
Verifies the 5 Phase-2 review items:
  1. High-sodium records: sodium_flag column present, 31 flagged, values unchanged
  2. Serving-size: standard_serving_weights.csv exists with >= 20 entries
  3. FoodLens duplicates: cross-split leakage flagged in manifest
  4. Tar assessment: findings documented in audit report
  5. Audit report updated with phase2_review section
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC_NUT = ROOT / "data" / "processed" / "nutrition"
PROC_IMG = ROOT / "data" / "processed" / "images"
LOG_DIR = ROOT / "logs"

PASS = 0
FAIL = 0
results = []


def test(name, condition, detail=""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    msg = f"[{status}] {name}"
    if detail:
        msg += f" | {detail}"
    print(msg)
    results.append({"test": name, "status": status, "detail": detail})


# ---------------------------------------------------------------------------
# ITEM 1: Sodium investigation
# ---------------------------------------------------------------------------
df_dish = pd.read_csv(PROC_NUT / "nutrition_dish_level.csv")

test("sodium_flag: column present", "sodium_flag" in df_dish.columns)
test("sodium_flag: 31 records flagged high_sodium_unverified",
     (df_dish["sodium_flag"] == "high_sodium_unverified").sum() == 31,
     f"flagged={(df_dish['sodium_flag']=='high_sodium_unverified').sum()}")
test("sodium_flag: 983 records ok",
     (df_dish["sodium_flag"] == "ok").sum() == 983,
     f"ok={(df_dish['sodium_flag']=='ok').sum()}")
test("sodium: flagged records all have sodium_mg > 5000",
     df_dish[df_dish["sodium_flag"] == "high_sodium_unverified"]["sodium_mg"].gt(5000).all())
test("sodium: raw values unchanged (all high-sodium records retained)",
     len(df_dish[df_dish["sodium_mg"] > 5000]) == 31,
     "records retained, not deleted")
# Cross-check: flag only where sodium > 5000
test("sodium: no record with sodium<=5000 is flagged",
     df_dish[df_dish["sodium_mg"] <= 5000]["sodium_flag"].eq("ok").all())

# Verify transformation fidelity (raw == processed values)
df_raw = pd.read_csv(ROOT / "data" / "raw" / "Indian_Food_Nutrition_Processed.csv")
df_raw_idx = df_raw.set_index("Dish Name")
mismatches = 0
for _, row in df_dish[df_dish["sodium_flag"] == "high_sodium_unverified"].iterrows():
    raw_name = row["food_name_raw"]
    if raw_name in df_raw_idx.index:
        raw_val = float(df_raw_idx.loc[raw_name, "Sodium (mg)"])
        if abs(raw_val - row["sodium_mg"]) > 1.0:
            mismatches += 1
test("sodium: transformation fidelity -- all flagged values match raw source",
     mismatches == 0, f"mismatches={mismatches}")

# ---------------------------------------------------------------------------
# ITEM 2: Serving-size limitations
# ---------------------------------------------------------------------------
serving_path = PROC_NUT / "standard_serving_weights.csv"
test("serving_weights: file exists", serving_path.exists())

if serving_path.exists():
    df_sv = pd.read_csv(serving_path)
    test("serving_weights: >= 20 entries", len(df_sv) >= 20, f"entries={len(df_sv)}")
    test("serving_weights: food_name column present", "food_name" in df_sv.columns)
    test("serving_weights: serving_g column present", "serving_g" in df_sv.columns)
    test("serving_weights: serving_g values are positive",
         df_sv["serving_g"].gt(0).all())
    test("serving_weights: no null food_name",
         df_sv["food_name"].notna().all())
    # Check key entries are present
    key_foods = ["idli", "chapati", "rice", "dal", "samosa"]
    for food in key_foods:
        test(f"serving_weights: '{food}' entry present",
             df_sv["food_name"].str.contains(food, na=False).any())

test("nutrition_dish_level: serving_reference column present",
     "serving_reference" in df_dish.columns)
test("nutrition_dish_level: no serving_size_g column (correctly absent, not fabricated)",
     "serving_size_g" not in df_dish.columns)

# Packaged foods still has serving_size_g
df_pack = pd.read_csv(PROC_NUT / "nutrition_packaged_foods.csv")
test("packaged_foods: serving_size_g column present (this source has it)",
     "serving_size_g" in df_pack.columns)

# ---------------------------------------------------------------------------
# ITEM 3: FoodLens duplicate split check
# ---------------------------------------------------------------------------
df_manifest = pd.read_csv(PROC_IMG / "food_lens_manifest.csv")

test("manifest: exclude_leakage column present", "exclude_leakage" in df_manifest.columns)
test("manifest: exactly 2 cross-split duplicates flagged",
     df_manifest["exclude_leakage"].sum() == 2,
     f"flagged={df_manifest['exclude_leakage'].sum()}")
test("manifest: flagged images are in val or test splits",
     df_manifest[df_manifest["exclude_leakage"]]["split"].isin(["val", "test"]).all())
test("manifest: flagged images are dhokla and samosa classes",
     set(df_manifest[df_manifest["exclude_leakage"]]["class"]).issubset({"dhokla", "samosa"}),
     f"classes={set(df_manifest[df_manifest['exclude_leakage']]['class'])}")
test("manifest: raw files NOT deleted (all 2523 rows remain)",
     len(df_manifest) == 2523, f"rows={len(df_manifest)}")

# Verify: if we exclude leakage, val+test have no cross-split dups with train
clean_manifest = df_manifest[~df_manifest["exclude_leakage"]]
test("manifest: after exclusion, no filename appears in both train and val",
     clean_manifest.groupby("filename")["split"].nunique().le(1).all())
test("manifest: after exclusion, no filename appears in both train and test",
     clean_manifest.groupby("filename")["split"].nunique().le(1).all())

# Verify findings log
dup_log = LOG_DIR / "food_lens_duplicate_images.json"
test("duplicate_log: file exists", dup_log.exists())
if dup_log.exists():
    with open(dup_log) as f:
        dups = json.load(f)
    test("duplicate_log: exactly 3 total duplicates recorded", len(dups) == 3,
         f"count={len(dups)}")

# ---------------------------------------------------------------------------
# ITEM 4: Tar assessment documented in audit report
# ---------------------------------------------------------------------------
audit_path = LOG_DIR / "dataset_audit_report.json"
test("audit_report: file exists", audit_path.exists())

if audit_path.exists():
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)

    test("audit_report: phase2_review section present", "phase2_review" in audit)

    if "phase2_review" in audit:
        findings = audit["phase2_review"].get("findings", {})
        test("audit_report: item4_tar section present", "item4_tar" in findings)

        if "item4_tar" in findings:
            tar = findings["item4_tar"]
            test("audit_report: tar archive_size_gb recorded",
                 "archive_size_gb" in tar and float(tar["archive_size_gb"]) > 5)
            test("audit_report: tar decision recorded",
                 "decision" in tar and tar["decision"] in ("EXTRACTION_OPTIONAL", "DO_NOT_EXTRACT"))
            test("audit_report: tar unique_to_tar >= 50 unique classes",
                 int(tar.get("unique_to_tar", 0)) >= 50,
                 f"unique_labels={tar.get('unique_to_tar')}")
            test("audit_report: disk_free_gb recorded",
                 "disk_free_gb" in tar)

# ---------------------------------------------------------------------------
# ITEM 5: Review findings file saved
# ---------------------------------------------------------------------------
review_path = LOG_DIR / "phase2_review_findings.json"
test("review_findings: file exists", review_path.exists())

if review_path.exists():
    with open(review_path, encoding="utf-8") as f:
        rev = json.load(f)
    test("review_findings: item1_sodium present", "item1_sodium" in rev)
    test("review_findings: item2_serving present", "item2_serving" in rev)
    test("review_findings: item3_duplicates present", "item3_duplicates" in rev)
    test("review_findings: item4_tar present", "item4_tar" in rev)

    if "item3_duplicates" in rev:
        dup3 = rev["item3_duplicates"]
        test("review_findings: 2 cross-split leakage pairs documented",
             int(dup3.get("cross_split_leakage", -1)) == 2,
             f"cross_split={dup3.get('cross_split_leakage')}")
        test("review_findings: 1 same-split duplicate documented",
             int(dup3.get("same_split_dups", -1)) == 1,
             f"same_split={dup3.get('same_split_dups')}")

# ---------------------------------------------------------------------------
# Check standard_serving_weights.csv is readable as expected
# ---------------------------------------------------------------------------
test("serving_weights: description column present for transparency",
     "description" in pd.read_csv(PROC_NUT / "standard_serving_weights.csv").columns
     if (PROC_NUT / "standard_serving_weights.csv").exists() else False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print(f"PHASE 2 REVIEW TESTS: {PASS} PASSED / {FAIL} FAILED / {PASS+FAIL} TOTAL")
print("=" * 60)
if FAIL > 0:
    print("FAILED tests:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  - {r['test']}: {r['detail']}")
    sys.exit(1)
else:
    print("All review tests passed.")
