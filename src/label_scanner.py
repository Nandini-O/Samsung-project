"""
src/label_scanner.py
NutriSense AI - Packaged Food Label Scanner (OCR Engine)

Uses EasyOCR to extract text from a packaged food product photo, then applies
structured regex parsing to recover:
  - Product name and brand
  - Nutrition facts (calories, macros, micros) with declared units and basis
  - Serving size / servings per pack
  - Ingredients list
  - Declared allergens

Design principles:
  - Only reports values that are supported by the OCR output.
  - Missing fields are recorded as None with a reason string.
  - Never substitutes zero for unknown/unreadable values.
  - Tracks which fields are OCR-extracted vs user-corrected.
  - Preserves units as printed (g, mg, %, kcal, kJ, ml, oz).
  - Does NOT confuse sodium with salt or perform undocumented conversions.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy-load EasyOCR reader (first call downloads CRAFT + recognition models).
# Lazy init ensures Flask startup is not blocked when scanner is not used.
# ---------------------------------------------------------------------------
_ocr_reader = None


def _get_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import ssl
        try:
            ssl._create_default_https_context = ssl._create_unverified_context
        except Exception:
            pass
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _ocr_reader


# ---------------------------------------------------------------------------
# Text preprocessing helpers
# ---------------------------------------------------------------------------

def _normalize_text(raw: str) -> str:
    """Unicode-normalize and clean raw OCR output to ASCII-friendly text."""
    text = unicodedata.normalize("NFKD", raw)
    replacements = {
        "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
        "\u00b5": "u",
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "|": "I",
        "\u2022": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def _merge_lines(raw_results: List[Tuple]) -> str:
    """Flatten EasyOCR bounding-box list into a single ordered text block."""
    sorted_items = sorted(raw_results, key=lambda r: (r[0][0][1], r[0][0][0]))
    lines = [_normalize_text(item[1]) for item in sorted_items]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regex pattern helpers
# ---------------------------------------------------------------------------

_NUM = r"(\d[\d\.,]*)"
_UNIT = r"(kcal|kj|g|mg|mcg|%|ml|oz|IU)"


def _find_nutrient(text: str, patterns: List[str]) -> Optional[Dict[str, str]]:
    """Try each regex pattern; return first match as dict or None."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            groups = m.groups()
            value = groups[0].replace(",", "").strip() if groups else None
            unit = groups[1].strip() if len(groups) > 1 and groups[1] else None
            return {"value": value, "unit": unit, "ocr_extracted": True}
    return None


# ---------------------------------------------------------------------------
# Section splitters
# ---------------------------------------------------------------------------

_NUTRITION_SECTION_START = re.compile(
    r"nutrition\s*(facts|information|value[s]?)|amount\s*per\s*serving|"
    r"per\s*(100\s*g|serving|portion|pack|100\s*ml)",
    re.IGNORECASE
)

_INGREDIENTS_START = re.compile(
    r"ingredient[s]?\s*[:]*\s*|contains\s*[:]\s*",
    re.IGNORECASE
)

_ALLERGEN_START = re.compile(
    r"allergen[s]?\s*(information|warning|declaration|alert)?|"
    r"allergy\s+(advice|warning|information)",
    re.IGNORECASE
)


def _split_sections(full_text: str) -> Dict[str, str]:
    """Split OCR text into header, nutrition, ingredients, and allergen blocks."""
    lines = full_text.splitlines()
    sections: Dict[str, List[str]] = {
        "header": [], "nutrition": [], "ingredients": [], "allergen": []
    }
    current = "header"
    for line in lines:
        stripped = line.strip()
        if _NUTRITION_SECTION_START.search(stripped):
            current = "nutrition"
        elif _INGREDIENTS_START.search(stripped):
            current = "ingredients"
        elif _ALLERGEN_START.search(stripped):
            current = "allergen"
        sections[current].append(stripped)
    return {k: "\n".join(v) for k, v in sections.items()}


# ---------------------------------------------------------------------------
# Header / Product identification
# ---------------------------------------------------------------------------

def _extract_product_info(header_text: str, full_text: str) -> Dict[str, Any]:
    """Identify product name and brand from the header block."""
    lines = [ln.strip() for ln in header_text.splitlines() if ln.strip()]
    if not lines:
        lines = [ln.strip() for ln in full_text.splitlines() if ln.strip()][:5]

    candidate_lines = [
        ln for ln in lines
        if len(ln) > 2 and not re.fullmatch(r"[\d\s\.,]+", ln)
    ]

    product_name: Optional[str] = candidate_lines[0] if candidate_lines else None
    brand: Optional[str] = candidate_lines[1] if len(candidate_lines) > 1 else None

    # If second line is shorter and ALL-CAPS, it is more likely the brand
    if product_name and brand:
        if len(brand) < len(product_name) and brand == brand.upper():
            product_name, brand = brand, product_name

    return {
        "product_name": product_name,
        "brand": brand,
        "ocr_extracted": True
    }


# ---------------------------------------------------------------------------
# Serving size
# ---------------------------------------------------------------------------

def _extract_serving(text: str) -> Dict[str, Any]:
    """Extract serving size and servings per container from nutrition block."""
    result: Dict[str, Any] = {
        "serving_size": None,
        "serving_unit": None,
        "servings_per_container": None,
        "nutrition_basis": None,
        "ocr_extracted": True,
        "raw_match": None
    }

    m = re.search(
        r"serving\s*size\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        text, re.IGNORECASE
    )
    if m:
        groups = [g for g in m.groups() if g]
        if len(groups) >= 2:
            result["serving_size"] = groups[0].replace(",", "")
            result["serving_unit"] = groups[1]
        elif len(groups) == 1:
            result["serving_size"] = groups[0]
        result["raw_match"] = m.group(0)

    basis_m = re.search(
        r"\bper\s+(100\s*(?:g|ml)|serving|portion|pack(?:age)?|100g)\b",
        text, re.IGNORECASE
    )
    if basis_m:
        result["nutrition_basis"] = basis_m.group(0).strip()

    spc_m = re.search(
        r"(?:about\s+)?(\d+)\s+servings?\s*(?:per\s*(?:container|pack(?:age)?))?|"
        r"servings?\s*(?:per\s*(?:container|pack(?:age)?))?\s*[:\-]?\s*(\d+)",
        text, re.IGNORECASE
    )
    if spc_m:
        val = spc_m.group(1) or spc_m.group(2)
        if val:
            result["servings_per_container"] = val

    return result


# ---------------------------------------------------------------------------
# Nutrition facts extraction
# ---------------------------------------------------------------------------

_NUTRIENT_PATTERNS: Dict[str, List[str]] = {
    "calories": [
        r"(?:energy|calories?)\s*[:\-]?\s*" + _NUM + r"\s*(kcal|kj|cal)",
        r"(?:energy|calories?)\s*[:\-]?\s*" + _NUM,
    ],
    "total_fat": [
        r"total\s*fat\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bfat\s+total\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "saturated_fat": [
        r"sat(?:urated)?\s*fat\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"of\s+which\s+saturate[sd]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "trans_fat": [
        r"trans\s*fat\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"trans\s*fatty\s*acids?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "total_carbohydrates": [
        r"total\s*carbo?hydrate[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"carbo?hydrate[s]?\s*total\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bcarb[o]?[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "total_sugars": [
        r"total\s*sugar[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bsugar[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "added_sugars": [
        r"added\s*sugar[s]?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "dietary_fibre": [
        r"dietary\s*fi(?:b|br)re?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"\bfi(?:b|br)re?\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "protein": [
        r"protein\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "sodium": [
        r"\bsodium\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "salt": [
        r"\bsalt\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "calcium": [
        r"\bcalcium\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "iron": [
        r"\biron\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "vitamin_c": [
        r"vitamin\s*c\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
        r"ascorbic\s*acid\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "vitamin_d": [
        r"vitamin\s*d\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "potassium": [
        r"\bpotassium\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
    "cholesterol": [
        r"\bcholesterol\s*[:\-]?\s*" + _NUM + r"\s*" + _UNIT,
    ],
}


def _extract_nutrition(text: str) -> Dict[str, Any]:
    """Parse all recognisable nutrition values from the nutrition text block."""
    results: Dict[str, Any] = {}
    for nutrient, patterns in _NUTRIENT_PATTERNS.items():
        found = _find_nutrient(text, patterns)
        results[nutrient] = found  # None if not found / not visible
    return results


# ---------------------------------------------------------------------------
# Ingredients
# ---------------------------------------------------------------------------

def _extract_ingredients(text: str) -> Optional[str]:
    """Return the raw ingredients string or None if not visible."""
    if not text.strip():
        return None
    cleaned = _INGREDIENTS_START.sub("", text, count=1).strip()
    return cleaned if cleaned else None


# ---------------------------------------------------------------------------
# Allergens
# ---------------------------------------------------------------------------

_KNOWN_ALLERGENS = [
    "milk", "dairy", "lactose", "wheat", "gluten", "soy", "soya",
    "peanut", "peanuts", "groundnut", "tree nut", "tree nuts",
    "almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
    "hazelnut", "hazelnuts", "pistachio", "pistachios", "pecan",
    "egg", "eggs", "fish", "shellfish", "shrimp", "prawn", "crab",
    "lobster", "sesame", "mustard", "celery", "lupin", "mollusc",
    "sulphite", "sulphites", "sulfite", "sulfites"
]


def _extract_allergens(allergen_text: str, ingredients_text: str) -> Dict[str, Any]:
    """Extract declared allergens from allergen block and ingredients list."""
    declared: List[str] = []
    may_contain: List[str] = []

    combined = f"{allergen_text}\n{ingredients_text}".lower()

    for allergen in _KNOWN_ALLERGENS:
        if re.search(r"\b" + re.escape(allergen) + r"\b", combined):
            if allergen not in declared:
                declared.append(allergen)

    may_m = re.search(
        r"may\s+contain\s*(traces?\s*of\s*)?([\w\s,]+)",
        combined, re.IGNORECASE
    )
    if may_m:
        trace_items = [t.strip() for t in re.split(r"[,;]", may_m.group(2)) if t.strip()]
        may_contain = trace_items[:8]

    return {
        "declared": declared if declared else None,
        "may_contain": may_contain if may_contain else None,
        "ocr_extracted": True
    }


# ---------------------------------------------------------------------------
# Readability / Confidence Assessment
# ---------------------------------------------------------------------------

def _assess_readability(raw_results: List[Tuple]) -> Dict[str, Any]:
    """Derive overall readability score from EasyOCR per-bbox confidence values."""
    if not raw_results:
        return {"mean_confidence": 0.0, "readable_bbox_count": 0, "quality": "UNREADABLE"}

    confs = [r[2] for r in raw_results if len(r) > 2]
    if not confs:
        return {"mean_confidence": 0.0, "readable_bbox_count": 0, "quality": "UNREADABLE"}

    mean_conf = float(np.mean(confs))
    n_readable = int(sum(1 for c in confs if c > 0.5))

    if mean_conf >= 0.75 and n_readable >= 5:
        quality = "GOOD"
    elif mean_conf >= 0.50 and n_readable >= 2:
        quality = "PARTIAL"
    else:
        quality = "LOW"

    return {
        "mean_confidence": round(mean_conf, 3),
        "readable_bbox_count": n_readable,
        "quality": quality
    }


# ---------------------------------------------------------------------------
# Missing field summary
# ---------------------------------------------------------------------------

def _summarise_missing(
    nutrition: Dict,
    serving: Dict,
    product: Dict,
    ingredients: Optional[str],
    allergens: Dict
) -> List[str]:
    """Collect field names that could not be extracted. Never set to 0."""
    missing: List[str] = []

    if not product.get("product_name"):
        missing.append("product_name")
    if not product.get("brand"):
        missing.append("brand")
    if not serving.get("serving_size"):
        missing.append("serving_size")
    if not serving.get("nutrition_basis"):
        missing.append("nutrition_basis")
    if not ingredients:
        missing.append("ingredients")
    if not allergens.get("declared"):
        missing.append("declared_allergens")

    priority_nutrients = [
        "calories", "total_fat", "total_carbohydrates", "protein", "sodium"
    ]
    for field in priority_nutrients:
        if nutrition.get(field) is None:
            missing.append(f"nutrition.{field}")

    return missing


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "Nutrition information is extracted via OCR from the uploaded product image and "
    "may be incomplete or inaccurate due to image quality, label layout, or text "
    "legibility. Always verify against the physical product label. This is for "
    "educational and dietary awareness only and does not constitute clinical advice."
)


def scan_label(pil_image: Image.Image) -> Dict[str, Any]:
    """
    Main entry point. Given a PIL Image of a packaged food product,
    run EasyOCR and return a structured extraction dict.

    The returned dict contains:
      ocr_text, readability, product, serving, nutrition,
      ingredients, allergens, missing_fields, extraction_note, disclaimer.

    Fields that cannot be extracted are set to None, never to 0 or a default.
    """
    reader = _get_reader()
    img_array = np.array(pil_image.convert("RGB"))
    raw_results = reader.readtext(img_array, detail=1, paragraph=False)

    if not raw_results:
        return {
            "ocr_text": "",
            "readability": {
                "mean_confidence": 0.0,
                "readable_bbox_count": 0,
                "quality": "UNREADABLE"
            },
            "product": {"product_name": None, "brand": None, "ocr_extracted": True},
            "serving": {
                "serving_size": None, "serving_unit": None,
                "nutrition_basis": None, "servings_per_container": None,
                "ocr_extracted": True
            },
            "nutrition": {k: None for k in _NUTRIENT_PATTERNS},
            "ingredients": None,
            "allergens": {"declared": None, "may_contain": None, "ocr_extracted": True},
            "missing_fields": ["ALL — no text detected in image"],
            "extraction_note": (
                "No text was detected in the uploaded image. "
                "The image may be blurry, too dark, or not show a nutrition label."
            ),
            "disclaimer": _DISCLAIMER
        }

    full_text = _merge_lines(raw_results)
    readability = _assess_readability(raw_results)
    sections = _split_sections(full_text)

    product = _extract_product_info(sections["header"], full_text)
    serving = _extract_serving(sections["nutrition"] or full_text)
    nutrition = _extract_nutrition(sections["nutrition"] or full_text)
    ingredients = _extract_ingredients(sections["ingredients"])
    allergens = _extract_allergens(sections["allergen"], sections["ingredients"])

    # Retry serving extraction on full text if section-based found nothing
    if not serving.get("serving_size") and not serving.get("nutrition_basis"):
        serving = _extract_serving(full_text)

    missing = _summarise_missing(nutrition, serving, product, ingredients, allergens)

    notes = []
    if readability["quality"] == "LOW":
        notes.append(
            "Image quality is low — OCR confidence is poor. "
            "Results may be incomplete or inaccurate."
        )
    elif readability["quality"] == "PARTIAL":
        notes.append(
            "Image is partially readable — some fields may not have been extracted."
        )
    if "nutrition.calories" in missing and "nutrition.total_carbohydrates" in missing:
        notes.append("Nutrition panel was not visible or readable in this image.")
    if "ingredients" in missing:
        notes.append("Ingredients list was not visible in this image.")

    extraction_note = " ".join(notes) if notes else "Extraction completed successfully."

    return {
        "ocr_text": full_text,
        "readability": readability,
        "product": product,
        "serving": serving,
        "nutrition": nutrition,
        "ingredients": ingredients,
        "allergens": allergens,
        "missing_fields": missing,
        "extraction_note": extraction_note,
        "disclaimer": _DISCLAIMER
    }
