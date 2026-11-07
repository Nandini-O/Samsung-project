"""
run_all_tests.py
Master Test Suite Runner for NutriSense AI (Phases 1 through 9)

Executes all 10 comprehensive test suites and generates an aggregated verification scorecard:
1. Phase 1 Preprocessing Tests (tests/test_preprocessing.py)
2. Phase 2 Review & Remediation Tests (tests/test_phase2_review.py)
3. Phase 3 EDA Tests (tests/test_eda.py)
4. Phase 4 Feature Engineering Tests (tests/test_feature_engineering.py)
5. Phase 5 Model & Inference Tests (tests/test_model_training_and_inference.py)
6. Phase 6 Web Application Tests (tests/test_web_application.py)
7. Phase 7 Nutrition Analysis Tests (tests/test_nutrition_analysis.py)
8. Packaged Food Label Scanner Tests (tests/test_label_scanner.py)
9. Phase 8 Personalization & User Profiling Tests (tests/test_user_profiling.py)
10. Phase 9 Recommendation System Tests (tests/test_recommendation_system.py)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent

TEST_SUITES = [
    ("Phase 1: Preprocessing & Data Cleaning", "tests/test_preprocessing.py"),
    ("Phase 2: Review & Anomaly Remediation", "tests/test_phase2_review.py"),
    ("Phase 3: Exploratory Data Analysis (EDA)", "tests/test_eda.py"),
    ("Phase 4: Feature Engineering & Knowledge Base", "tests/test_feature_engineering.py"),
    ("Phase 5: AI Model Training & Inference Engine", "tests/test_model_training_and_inference.py"),
    ("Phase 6: Web Application & REST APIs", "tests/test_web_application.py"),
    ("Phase 7: Nutrition Analysis Module", "tests/test_nutrition_analysis.py"),
    ("Packaged Food Label Scanner Module", "tests/test_label_scanner.py"),
    ("Phase 8: Personalization & User Profiling", "tests/test_user_profiling.py"),
    ("Phase 9: Recommendation System Module", "tests/test_recommendation_system.py"),
]


def run_master_test_suite():
    print("=" * 75)
    print("NUTRISENSE AI — MASTER VERIFICATION & TEST REGRESSION SUITE")
    print(f"Workspace: {WORKSPACE}")
    print("=" * 75)

    results = []
    overall_start = time.time()

    for name, rel_path in TEST_SUITES:
        script_path = WORKSPACE / rel_path
        print(f"\n>> Executing Suite: {name} ({rel_path})...")
        t0 = time.time()

        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
        )
        duration = time.time() - t0

        passed = proc.returncode == 0
        status_str = "PASSED" if passed else "FAILED"
        print(f"   Result: [{status_str}] in {duration:.2f}s (Exit Code: {proc.returncode})")

        # Parse test counts from output if available
        output = proc.stdout + "\n" + proc.stderr
        passed_count = None
        total_count = None
        for line in output.splitlines():
            line_upper = line.upper()
            if "PASSED" in line_upper and "FAILED" in line_upper and "TOTAL" in line_upper:
                parts = line.split()
                # look for digits around PASSED and TOTAL
                for i, token in enumerate(parts):
                    if "PASS" in token.upper() and i > 0 and parts[i - 1].isdigit():
                        passed_count = int(parts[i - 1])
                    if "TOTAL" in token.upper() and i > 0 and parts[i - 1].isdigit():
                        total_count = int(parts[i - 1])

        results.append({
            "name": name,
            "path": rel_path,
            "passed": passed,
            "duration": duration,
            "passed_count": passed_count,
            "total_count": total_count,
            "output": output,
        })

    total_time = time.time() - overall_start

    print("\n" + "=" * 75)
    print("ALL SUITES EXECUTION SUMMARY SCORECARD")
    print("=" * 75)
    print(f"{'Test Suite':<48} | {'Status':<8} | {'Passed/Total':<14} | {'Time':<6}")
    print("-" * 75)

    all_passed = True
    total_passed_tests = 0
    total_run_tests = 0

    for r in results:
        status_display = "OK" if r["passed"] else "FAIL"
        counts_display = f"{r['passed_count']}/{r['total_count']}" if r["passed_count"] is not None else ("ALL" if r["passed"] else "ERROR")
        if r["passed_count"] is not None:
            total_passed_tests += r["passed_count"]
            total_run_tests += (r["total_count"] or r["passed_count"])
        print(f"{r['name']:<48} | {status_display:<8} | {counts_display:<14} | {r['duration']:>5.2f}s")
        if not r["passed"]:
            all_passed = False

    print("-" * 75)
    print(f"Total Tests Verified: {total_passed_tests} Passed / {total_run_tests} Run in {total_time:.2f}s")
    print(f"Master Verification Status: {'100% ALL SUITES PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 75)

    if not all_passed:
        print("\nFAILURE DETAILS:")
        for r in results:
            if not r["passed"]:
                print(f"\n--- {r['name']} Output ---")
                print(r["output"][-1000:])
        sys.exit(1)


if __name__ == "__main__":
    run_master_test_suite()
