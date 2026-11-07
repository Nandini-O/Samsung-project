"""
data_audit.py -- Dataset Audit Utility for NutriSense AI
Reports available files, image counts, CSV columns, missing values, duplicates,
food-name inconsistencies, nutrition units, and recipe coverage.
Original datasets are never modified by this script.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "data_audit.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_read_csv(path):
    for enc in ("utf-8", "latin-1", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip")
        except Exception:
            continue
    log.warning("  Could not read %s", path)
    return None


def audit_dataframe(df, name):
    n_rows, n_cols = df.shape
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    duplicates = int(df.duplicated().sum())
    return {
        "name": name,
        "rows": n_rows,
        "columns": n_cols,
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_total": int(missing.sum()),
        "missing_per_col": {
            c: {"count": int(missing[c]), "pct": float(missing_pct[c])}
            for c in df.columns
            if missing[c] > 0
        },
        "duplicates": duplicates,
    }


def audit_image_directory(root, name):
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}
    corrupt_candidates = []
    root = Path(root)
    if not root.is_dir():
        return {"name": name, "error": f"Directory not found: {root}"}

    subdirs = [d for d in root.iterdir() if d.is_dir()]
    splits = [d for d in subdirs if d.name in ("train", "val", "validation", "test")]

    if splits:
        result = {"name": name, "structure": "split/class", "splits": {}}
        for split in splits:
            split_data = {}
            for cls_dir in sorted(split.iterdir()):
                if cls_dir.is_dir():
                    imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
                    for img in imgs:
                        if img.stat().st_size == 0:
                            corrupt_candidates.append(str(img))
                    cls_name = cls_dir.name.replace("food indian_food ", "").replace("_", " ").strip()
                    split_data[cls_name] = len(imgs)
            result["splits"][split.name] = {
                "classes": len(split_data),
                "total_images": sum(split_data.values()),
                "per_class": split_data,
            }
        result["total_images"] = sum(v["total_images"] for v in result["splits"].values())
        result["corrupt_candidates"] = corrupt_candidates
        return result

    result = {"name": name, "structure": "class", "classes": {}}
    for cls_dir in sorted(root.iterdir()):
        if cls_dir.is_dir():
            imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
            for img in imgs:
                if img.stat().st_size == 0:
                    corrupt_candidates.append(str(img))
            result["classes"][cls_dir.name] = len(imgs)
    result["total_images"] = sum(result["classes"].values())
    result["corrupt_candidates"] = corrupt_candidates
    return result


def audit_parquet_labels(parquet_dir, name):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"name": name, "error": "pyarrow not installed"}

    files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not files:
        return {"name": name, "error": "No parquet files found"}

    label_counts = {}
    split_info = {}
    for f in files:
        split_key = f.stem.split("-")[0]
        try:
            df = pd.read_parquet(f, columns=["label"])
            counts = df["label"].value_counts().to_dict()
            split_info.setdefault(split_key, {})
            for lbl, cnt in counts.items():
                split_info[split_key][lbl] = split_info[split_key].get(lbl, 0) + cnt
                label_counts[lbl] = label_counts.get(lbl, 0) + cnt
        except Exception as e:
            log.warning("  Could not read parquet %s: %s", f.name, e)

    return {
        "name": name,
        "structure": "parquet (HuggingFace)",
        "total_images": sum(label_counts.values()),
        "unique_labels": len(label_counts),
        "all_labels": sorted(label_counts.keys()),
        "splits": {
            k: {"total": sum(v.values()), "classes": len(v), "per_class": v}
            for k, v in split_info.items()
        },
    }


def audit_nutrition_csv(path):
    path = Path(path)
    df = safe_read_csv(path)
    if df is None:
        return {"error": f"Could not read {path.name}"}

    base = audit_dataframe(df, path.name)
    cols_lower = [c.lower() for c in df.columns]
    has_serving = any("serv" in c for c in cols_lower)
    has_per100 = any("100" in c or "per 100" in c or "/100" in c for c in df.columns)

    unit_markers = {}
    for col in df.columns:
        for unit in ["kcal", "kj", "g", "mg", "ug", "%"]:
            if "(" + unit + ")" in col.lower() or "[" + unit + "]" in col.lower():
                unit_markers[col] = unit
                break

    name_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ["name", "dish", "food", "item"])),
        None,
    )
    sample_names = list(df[name_col].dropna().head(5)) if name_col else []

    base.update({
        "source_reference": "per 100g" if has_per100 else ("per serving" if has_serving else "unknown"),
        "detected_units": unit_markers,
        "name_column": name_col,
        "sample_food_names": [str(n) for n in sample_names],
    })
    return base


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit():
    report = {
        "audit_timestamp": datetime.now().isoformat(),
        "workspace_root": str(ROOT),
        "datasets": {},
    }

    log.info("=" * 60)
    log.info("NutriSense AI -- Dataset Audit")
    log.info("=" * 60)

    # Image datasets
    log.info("\n--- IMAGE DATASETS ---")

    enhanced_dir = RAW_DIR / "enhanced-indian-food-classification"
    if enhanced_dir.is_dir():
        log.info("Auditing: enhanced-indian-food-classification (parquet)")
        r = audit_parquet_labels(enhanced_dir, "enhanced-indian-food-classification")
        report["datasets"]["enhanced_indian_food"] = r
        log.info("  Total images: %s | Labels: %s", r.get("total_images"), r.get("unique_labels"))

    food_lens_dir = RAW_DIR / "food_lens" / "images.cv_rdi4no0tot5qp4nyjvgq6" / "data"
    if food_lens_dir.is_dir():
        log.info("Auditing: food_lens (folder-based)")
        r = audit_image_directory(food_lens_dir, "food_lens")
        report["datasets"]["food_lens"] = r
        log.info("  Total images: %s | Corrupt candidates: %s", r.get("total_images"), len(r.get("corrupt_candidates", [])))

    indian_foods_dir = RAW_DIR / "indian-foods-dataset"
    if indian_foods_dir.is_dir():
        log.info("Auditing: indian-foods-dataset (parquet)")
        r = audit_parquet_labels(indian_foods_dir, "indian-foods-dataset")
        report["datasets"]["indian_foods_dataset"] = r
        log.info("  Total images: %s | Labels: %s", r.get("total_images"), r.get("unique_labels"))

    food_tar = RAW_DIR / "Food" / "dataset.tar.gz"
    if food_tar.exists():
        lbl_file = RAW_DIR / "Food" / "labels.txt"
        labels = []
        if lbl_file.exists():
            labels = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
        report["datasets"]["food_dataset_tar"] = {
            "name": "Food/dataset.tar.gz",
            "status": "archive_not_extracted",
            "size_bytes": food_tar.stat().st_size,
            "note": "Contains 80-class Indian food dataset. Requires extraction.",
            "labels": labels,
        }
        log.info("  Food/dataset.tar.gz: archive present but not extracted (%.1f GB)", food_tar.stat().st_size / 1e9)

    # Nutrition datasets
    log.info("\n--- NUTRITION DATASETS ---")
    nutrition_files = {
        "indian_food_nutrition_processed": RAW_DIR / "Indian_Food_Nutrition_Processed.csv",
        "indian_food_composition_table": RAW_DIR / "IndianFoodCompositionTable_In_CSV_Format.csv",
        "daily_food_nutrition": RAW_DIR / "daily_food_nutrition_dataset.csv",
        "packaged_foods_india": RAW_DIR / "packaged_foods_india.csv",
    }
    for key, path in nutrition_files.items():
        if path.exists():
            log.info("Auditing: %s", path.name)
            r = audit_nutrition_csv(path)
            report["datasets"][key] = r
            log.info("  Rows: %s | Missing: %s | Dup: %s | Ref: %s", r.get("rows"), r.get("missing_total"), r.get("duplicates"), r.get("source_reference"))

    # Recipe datasets
    log.info("\n--- RECIPE DATASETS ---")
    recipe_files = {
        "indian_food_recipe_ndtv": RAW_DIR / "INDIAN FOOD RECIPE.csv",
        "recipes_regional": RAW_DIR / "recipes.csv",
        "indian_food_metadata": RAW_DIR / "indian_food.csv",
        "indian_food_dataset_full": RAW_DIR / "IndianFoodDatasetCSV.csv",
    }
    for key, path in recipe_files.items():
        if path.exists():
            log.info("Auditing: %s", path.name)
            df = safe_read_csv(path)
            if df is not None:
                r = audit_dataframe(df, path.name)
                for col in ["Cuisine", "Diet", "diet", "category", "Category"]:
                    if col in df.columns:
                        r[col + "_distribution"] = df[col].value_counts().head(15).to_dict()
                report["datasets"][key] = r
                log.info("  Rows: %s | Missing: %s | Dup: %s", r["rows"], r["missing_total"], r["duplicates"])

    # Out-of-scope datasets
    log.info("\n--- OUT-OF-SCOPE DATASETS ---")
    oos = {
        "diabetes_dataset": RAW_DIR / "diabetes_dataset.csv",
        "pcos_data": RAW_DIR / "PCOS_data.csv",
        "thyroid": RAW_DIR / "thyroid.csv",
        "obesity_train": RAW_DIR / "Obesity" / "obesity_train.csv",
        "nfhs_map": RAW_DIR / "NFHS_map_Children-under-5-stunted-pct_N6_Total.csv",
        "ahs_cab": RAW_DIR / "ahs-cab2014-madhya_pradesh-west_nimar.csv",
    }
    for key, path in oos.items():
        if path.exists():
            try:
                tmp = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
                n = len(tmp)
            except Exception:
                try:
                    tmp = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
                    n = len(tmp)
                except Exception:
                    n = -1
            report["datasets"][key] = {
                "name": Path(path).name,
                "status": "out_of_scope_for_food_features",
                "rows": n,
                "note": "Not used for food classification/nutrition.",
            }
            log.info("  %s: out of scope (%s rows)", Path(path).name, n)

    # XLSX
    xlsx_path = RAW_DIR / "IndianFoodDatasetXLS.xlsx"
    if xlsx_path.exists():
        try:
            df_xl = pd.read_excel(xlsx_path, nrows=3)
            report["datasets"]["indian_food_dataset_xlsx"] = {
                "name": xlsx_path.name,
                "columns": list(df_xl.columns),
                "note": "XLSX version of IndianFoodDatasetCSV -- likely same data",
            }
            log.info("  IndianFoodDatasetXLS.xlsx: columns=%s", list(df_xl.columns))
        except Exception as e:
            log.warning("  Could not read XLSX: %s", e)

    report_path = ROOT / "logs" / "dataset_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str, ensure_ascii=False)
    log.info("\nAudit report saved -> %s", report_path)
    return report


def print_summary(report):
    print("\n" + "=" * 60)
    print("DATASET AUDIT SUMMARY")
    print("=" * 60)
    for key, val in report.get("datasets", {}).items():
        name = val.get("name", key)
        status = val.get("status", "ok")
        rows = val.get("rows", val.get("total_images", "n/a"))
        err = val.get("error", "")
        if err:
            print(f"  ERROR   {name}: {err}")
        elif status == "out_of_scope_for_food_features":
            print(f"  SKIP    {name}: OUT OF SCOPE ({rows} rows)")
        elif status == "archive_not_extracted":
            print(f"  ARCHIVE {name}: not extracted, {val.get('size_bytes', 0)/1e9:.1f} GB")
        else:
            imgs = val.get("total_images", "")
            dup = val.get("duplicates", "")
            miss = val.get("missing_total", "")
            extra = f"images={imgs}" if imgs else f"rows={rows}"
            print(f"  OK      {name}: {extra} | missing={miss} | duplicates={dup}")
    print("=" * 60)


if __name__ == "__main__":
    report = run_audit()
    print_summary(report)
