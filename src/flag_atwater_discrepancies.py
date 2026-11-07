"""
src/flag_atwater_discrepancies.py
Flags the 35 dishes in nutrition_dish_level.csv with > 25% discrepancy
between reported calories and Atwater-calculated calories (4*C + 4*P + 9*F).
Preserves all raw values and existing sodium_flag.
"""

from pathlib import Path
import numpy as np
import pandas as pd

FILE_PATH = Path("data/processed/nutrition/nutrition_dish_level.csv")


def flag_discrepancies():
    df = pd.read_csv(FILE_PATH)
    print(f"Loaded {len(df)} records from {FILE_PATH}")

    c = pd.to_numeric(df["carbs_g"], errors="coerce").fillna(0)
    p = pd.to_numeric(df["protein_g"], errors="coerce").fillna(0)
    f = pd.to_numeric(df["fat_g"], errors="coerce").fillna(0)
    cal = pd.to_numeric(df["calories_kcal"], errors="coerce")

    atwater = (4 * c) + (4 * p) + (9 * f)
    diff = cal - atwater
    abs_diff = diff.abs()
    pct_diff = (abs_diff / cal.replace(0, np.nan)) * 100

    discrepant_mask = pct_diff > 25
    print(f"Dishes with > 25% Atwater discrepancy: {discrepant_mask.sum()}")

    df["calorie_flag"] = "ok"
    df.loc[discrepant_mask, "calorie_flag"] = "atwater_discrepancy_unverified"

    # Add explanatory note to data_quality_note
    for idx in df[discrepant_mask].index:
        rep_cal = df.loc[idx, "calories_kcal"]
        atw_cal = round(atwater[idx], 1)
        d_val = round(diff[idx], 1)
        pct_val = round(pct_diff[idx], 1)

        note_parts = [
            f"Atwater discrepancy: reported={rep_cal}kcal, calculated={atw_cal}kcal (diff={d_val:+0.1f}kcal, {pct_val}%)"
        ]
        if df.loc[idx, "sodium_flag"] == "high_sodium_unverified":
            note_parts.append(
                "Overlaps with high_sodium_unverified flag (source soup batch inflation)"
            )

        curr_note = df.loc[idx, "data_quality_note"]
        if pd.notnull(curr_note) and str(curr_note).strip():
            combined = str(curr_note) + "; " + "; ".join(note_parts)
        else:
            combined = "; ".join(note_parts)
        df.loc[idx, "data_quality_note"] = combined

    df.to_csv(FILE_PATH, index=False, encoding="utf-8")
    print(f"Saved updated file -> {FILE_PATH}")
    print("\nVerification:")
    print("  sodium_flag value counts:")
    print(df["sodium_flag"].value_counts())
    print("\n  calorie_flag value counts:")
    print(df["calorie_flag"].value_counts())


if __name__ == "__main__":
    flag_discrepancies()
