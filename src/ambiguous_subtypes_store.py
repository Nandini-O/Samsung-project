"""
src/ambiguous_subtypes_store.py
Phase 4: Feature Engineering - Ambiguous Subtypes Registry

Preserves the 18 ambiguous food classes for later user confirmation
rather than assuming they are nutritionally identical.

For each ambiguous class, registers:
- Canonical image class & aliases
- Discrete culinary subtypes with distinct macronutrient and micronutrient profiles
- Default standard profile (used if user skips disambiguation)
- Disambiguation prompt and selection options for the application UI/API

Outputs:
- data/processed/feature_store/ambiguous_subtypes_registry.json
- data/processed/feature_store/ambiguous_subtypes_registry.csv
"""

import json
from pathlib import Path
import pandas as pd

FEATURE_DIR = Path("data/processed/feature_store")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

SUBTYPES_REGISTRY = [
    # 1. Samosa
    {
        "parent_class": "samosa",
        "aliases": ["samosa"],
        "disambiguation_prompt": "What kind of samosa filling did you have?",
        "subtypes": [
            {
                "subtype_id": "samosa_potato",
                "display_name": "Vegetable / Potato Samosa (Standard Aloo)",
                "is_default": True,
                "serving_size_g": 65.0,
                "serving_unit": "1 piece",
                "calories_kcal": 256.0,
                "carbs_g": 28.0,
                "protein_g": 3.5,
                "fat_g": 14.5,
                "fibre_g": 2.2,
                "sodium_mg": 280.0,
                "source": "nutrition_dish_level.csv (potato samosa)"
            },
            {
                "subtype_id": "samosa_keema",
                "display_name": "Minced Meat Samosa (Keema Samosa)",
                "is_default": False,
                "serving_size_g": 70.0,
                "serving_unit": "1 piece",
                "calories_kcal": 318.0,
                "carbs_g": 25.5,
                "protein_g": 12.8,
                "fat_g": 18.2,
                "fibre_g": 1.4,
                "sodium_mg": 340.0,
                "source": "nutrition_dish_level.csv (keema samosa)"
            }
        ]
    },
    # 2. Biryani
    {
        "parent_class": "biryani",
        "aliases": ["biryani", "biriyani"],
        "disambiguation_prompt": "Which type of biryani is on your plate?",
        "subtypes": [
            {
                "subtype_id": "biryani_vegetable",
                "display_name": "Vegetable Biryani",
                "is_default": True,
                "serving_size_g": 200.0,
                "serving_unit": "1 plate / bowl",
                "calories_kcal": 220.0,
                "carbs_g": 34.5,
                "protein_g": 4.5,
                "fat_g": 7.2,
                "fibre_g": 3.5,
                "sodium_mg": 410.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "biryani_chicken",
                "display_name": "Chicken Biryani",
                "is_default": False,
                "serving_size_g": 250.0,
                "serving_unit": "1 plate / bowl",
                "calories_kcal": 330.0,
                "carbs_g": 33.5,
                "protein_g": 22.0,
                "fat_g": 12.0,
                "fibre_g": 2.2,
                "sodium_mg": 480.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "biryani_mutton",
                "display_name": "Mutton Biryani",
                "is_default": False,
                "serving_size_g": 250.0,
                "serving_unit": "1 plate / bowl",
                "calories_kcal": 380.0,
                "carbs_g": 35.0,
                "protein_g": 19.0,
                "fat_g": 18.0,
                "fibre_g": 2.0,
                "sodium_mg": 520.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    },
    # 3. Burger
    {
        "parent_class": "burger",
        "aliases": ["burger"],
        "disambiguation_prompt": "Is this a veggie or chicken burger?",
        "subtypes": [
            {
                "subtype_id": "burger_veg_aloo_tikki",
                "display_name": "Veg Aloo Tikki Burger",
                "is_default": True,
                "serving_size_g": 140.0,
                "serving_unit": "1 burger",
                "calories_kcal": 285.0,
                "carbs_g": 42.0,
                "protein_g": 5.8,
                "fat_g": 10.5,
                "fibre_g": 3.2,
                "sodium_mg": 520.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "burger_chicken",
                "display_name": "Crispy Chicken Burger",
                "is_default": False,
                "serving_size_g": 160.0,
                "serving_unit": "1 burger",
                "calories_kcal": 410.0,
                "carbs_g": 38.0,
                "protein_g": 23.5,
                "fat_g": 18.0,
                "fibre_g": 2.0,
                "sodium_mg": 680.0,
                "source": "nutrition_packaged_foods.csv"
            }
        ]
    },
    # 4. Butter Naan
    {
        "parent_class": "butter_naan",
        "aliases": ["butter_naan", "butternaan"],
        "disambiguation_prompt": "Did your naan have butter applied?",
        "subtypes": [
            {
                "subtype_id": "naan_plain",
                "display_name": "Plain Naan (No Butter)",
                "is_default": False,
                "serving_size_g": 90.0,
                "serving_unit": "1 piece",
                "calories_kcal": 260.0,
                "carbs_g": 49.5,
                "protein_g": 7.2,
                "fat_g": 3.5,
                "fibre_g": 2.4,
                "sodium_mg": 310.0,
                "source": "nutrition_dish_level.csv (naan)"
            },
            {
                "subtype_id": "naan_butter",
                "display_name": "Butter Naan (With Ghee/Butter)",
                "is_default": True,
                "serving_size_g": 100.0,
                "serving_unit": "1 piece",
                "calories_kcal": 340.0,
                "carbs_g": 49.5,
                "protein_g": 7.2,
                "fat_g": 12.5,
                "fibre_g": 2.4,
                "sodium_mg": 360.0,
                "source": "nutrition_dish_level.csv + butter scaling"
            }
        ]
    },
    # 5. Chai
    {
        "parent_class": "chai",
        "aliases": ["chai"],
        "disambiguation_prompt": "How was your chai prepared?",
        "subtypes": [
            {
                "subtype_id": "chai_standard_sweet",
                "display_name": "Masala Chai (Milk + 1.5 tsp Sugar)",
                "is_default": True,
                "serving_size_g": 150.0,
                "serving_unit": "1 cup (150ml)",
                "calories_kcal": 92.0,
                "carbs_g": 12.8,
                "protein_g": 2.6,
                "fat_g": 3.4,
                "fibre_g": 0.0,
                "sodium_mg": 45.0,
                "source": "nutrition_dish_level.csv (hot tea)"
            },
            {
                "subtype_id": "chai_low_sugar",
                "display_name": "Chai without Added Sugar",
                "is_default": False,
                "serving_size_g": 150.0,
                "serving_unit": "1 cup (150ml)",
                "calories_kcal": 52.0,
                "carbs_g": 2.8,
                "protein_g": 2.6,
                "fat_g": 3.4,
                "fibre_g": 0.0,
                "sodium_mg": 45.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "chai_black",
                "display_name": "Black Tea / Green Tea (Unsweetened)",
                "is_default": False,
                "serving_size_g": 150.0,
                "serving_unit": "1 cup (150ml)",
                "calories_kcal": 4.0,
                "carbs_g": 0.8,
                "protein_g": 0.2,
                "fat_g": 0.0,
                "fibre_g": 0.0,
                "sodium_mg": 5.0,
                "source": "IFCT 2017"
            }
        ]
    },
    # 6. Fried Rice
    {
        "parent_class": "fried_rice",
        "aliases": ["fried_rice"],
        "disambiguation_prompt": "What protein was added to your fried rice?",
        "subtypes": [
            {
                "subtype_id": "fried_rice_vegetable",
                "display_name": "Vegetable Fried Rice",
                "is_default": True,
                "serving_size_g": 180.0,
                "serving_unit": "1 plate",
                "calories_kcal": 215.0,
                "carbs_g": 34.0,
                "protein_g": 4.2,
                "fat_g": 7.0,
                "fibre_g": 2.5,
                "sodium_mg": 420.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "fried_rice_egg",
                "display_name": "Egg Fried Rice",
                "is_default": False,
                "serving_size_g": 190.0,
                "serving_unit": "1 plate",
                "calories_kcal": 265.0,
                "carbs_g": 34.5,
                "protein_g": 8.5,
                "fat_g": 10.2,
                "fibre_g": 2.2,
                "sodium_mg": 460.0,
                "source": "nutrition_dish_level.csv + egg composite"
            },
            {
                "subtype_id": "fried_rice_chicken",
                "display_name": "Chicken Fried Rice",
                "is_default": False,
                "serving_size_g": 200.0,
                "serving_unit": "1 plate",
                "calories_kcal": 295.0,
                "carbs_g": 36.0,
                "protein_g": 15.2,
                "fat_g": 9.8,
                "fibre_g": 2.2,
                "sodium_mg": 490.0,
                "source": "nutrition_dish_level.csv + chicken composite"
            }
        ]
    },
    # 7. Gulab Jamun
    {
        "parent_class": "gulab_jamun",
        "aliases": ["gulab_jamun", "gulab jamun"],
        "disambiguation_prompt": "Was the gulab jamun made from traditional khoya or milk powder?",
        "subtypes": [
            {
                "subtype_id": "gulab_jamun_khoya",
                "display_name": "Gulab Jamun (Traditional Khoya Mawa)",
                "is_default": True,
                "serving_size_g": 50.0,
                "serving_unit": "1 piece in syrup",
                "calories_kcal": 150.0,
                "carbs_g": 20.0,
                "protein_g": 2.4,
                "fat_g": 6.8,
                "fibre_g": 0.2,
                "sodium_mg": 28.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "gulab_jamun_milk_powder",
                "display_name": "Gulab Jamun (Instant Milk Powder)",
                "is_default": False,
                "serving_size_g": 50.0,
                "serving_unit": "1 piece in syrup",
                "calories_kcal": 132.0,
                "carbs_g": 21.0,
                "protein_g": 2.8,
                "fat_g": 4.2,
                "fibre_g": 0.2,
                "sodium_mg": 34.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    },
    # 8. Kebab
    {
        "parent_class": "kebab",
        "aliases": ["kebab"],
        "disambiguation_prompt": "Is this a meat kebab or vegetarian kebab?",
        "subtypes": [
            {
                "subtype_id": "kebab_shammi_meat",
                "display_name": "Meat Shammi / Chicken Seekh Kebab",
                "is_default": True,
                "serving_size_g": 80.0,
                "serving_unit": "2 pieces",
                "calories_kcal": 185.0,
                "carbs_g": 6.8,
                "protein_g": 16.5,
                "fat_g": 10.2,
                "fibre_g": 1.5,
                "sodium_mg": 380.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "kebab_soya_veg",
                "display_name": "Soya / Hara Bhara Veg Kebab",
                "is_default": False,
                "serving_size_g": 80.0,
                "serving_unit": "2 pieces",
                "calories_kcal": 145.0,
                "carbs_g": 11.8,
                "protein_g": 12.0,
                "fat_g": 5.5,
                "fibre_g": 3.8,
                "sodium_mg": 320.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    },
    # 9. Kheer
    {
        "parent_class": "kheer",
        "aliases": ["kheer"],
        "disambiguation_prompt": "What base grain was used in your kheer?",
        "subtypes": [
            {
                "subtype_id": "kheer_rice",
                "display_name": "Rice Kheer (Chawal ki Kheer)",
                "is_default": True,
                "serving_size_g": 150.0,
                "serving_unit": "1 bowl",
                "calories_kcal": 210.0,
                "carbs_g": 33.0,
                "protein_g": 4.8,
                "fat_g": 6.5,
                "fibre_g": 0.5,
                "sodium_mg": 65.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "kheer_seviyan",
                "display_name": "Seviyan Kheer (Vermicelli)",
                "is_default": False,
                "serving_size_g": 150.0,
                "serving_unit": "1 bowl",
                "calories_kcal": 230.0,
                "carbs_g": 35.0,
                "protein_g": 5.2,
                "fat_g": 7.8,
                "fibre_g": 0.8,
                "sodium_mg": 70.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "kheer_suji",
                "display_name": "Rava / Suji Kheer",
                "is_default": False,
                "serving_size_g": 150.0,
                "serving_unit": "1 bowl",
                "calories_kcal": 195.0,
                "carbs_g": 31.0,
                "protein_g": 4.5,
                "fat_g": 6.0,
                "fibre_g": 0.7,
                "sodium_mg": 60.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    },
    # 10. Lassi
    {
        "parent_class": "lassi",
        "aliases": ["lassi"],
        "disambiguation_prompt": "Did you drink sweet or salted lassi?",
        "subtypes": [
            {
                "subtype_id": "lassi_sweet",
                "display_name": "Sweet Lassi (Meethi Lassi with Sugar)",
                "is_default": True,
                "serving_size_g": 250.0,
                "serving_unit": "1 tall glass (250ml)",
                "calories_kcal": 185.0,
                "carbs_g": 30.5,
                "protein_g": 5.2,
                "fat_g": 4.8,
                "fibre_g": 0.0,
                "sodium_mg": 95.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "lassi_salted_chaas",
                "display_name": "Salted Lassi / Spiced Buttermilk (Chaas)",
                "is_default": False,
                "serving_size_g": 250.0,
                "serving_unit": "1 glass (250ml)",
                "calories_kcal": 65.0,
                "carbs_g": 6.5,
                "protein_g": 3.8,
                "fat_g": 2.5,
                "fibre_g": 0.0,
                "sodium_mg": 280.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    },
    # 11. Pakode
    {
        "parent_class": "pakode",
        "aliases": ["pakode"],
        "disambiguation_prompt": "What was the main ingredient inside your pakoda?",
        "subtypes": [
            {
                "subtype_id": "pakode_onion",
                "display_name": "Onion Pakoda (Pyaaz ke Pakode)",
                "is_default": True,
                "serving_size_g": 100.0,
                "serving_unit": "1 plate (~4-5 pcs)",
                "calories_kcal": 190.0,
                "carbs_g": 17.5,
                "protein_g": 4.2,
                "fat_g": 11.5,
                "fibre_g": 2.8,
                "sodium_mg": 310.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "pakode_paneer",
                "display_name": "Paneer Pakoda",
                "is_default": False,
                "serving_size_g": 100.0,
                "serving_unit": "1 plate (~3-4 pcs)",
                "calories_kcal": 265.0,
                "carbs_g": 14.8,
                "protein_g": 10.5,
                "fat_g": 18.2,
                "fibre_g": 1.4,
                "sodium_mg": 340.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "pakode_spinach",
                "display_name": "Palak / Spinach Pakoda",
                "is_default": False,
                "serving_size_g": 100.0,
                "serving_unit": "1 plate",
                "calories_kcal": 165.0,
                "carbs_g": 14.5,
                "protein_g": 4.8,
                "fat_g": 9.8,
                "fibre_g": 3.4,
                "sodium_mg": 290.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    },
    # 12. Kaathi Rolls
    {
        "parent_class": "kaathi_rolls",
        "aliases": ["kaathi_rolls", "kathi roll"],
        "disambiguation_prompt": "What filling was inside your kathi roll?",
        "subtypes": [
            {
                "subtype_id": "kaathi_roll_paneer",
                "display_name": "Paneer Kathi Roll (Veg)",
                "is_default": True,
                "serving_size_g": 160.0,
                "serving_unit": "1 roll",
                "calories_kcal": 320.0,
                "carbs_g": 36.5,
                "protein_g": 11.5,
                "fat_g": 14.2,
                "fibre_g": 2.8,
                "sodium_mg": 440.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "kaathi_roll_chicken",
                "display_name": "Chicken Tikka Kathi Roll",
                "is_default": False,
                "serving_size_g": 160.0,
                "serving_unit": "1 roll",
                "calories_kcal": 345.0,
                "carbs_g": 36.5,
                "protein_g": 21.0,
                "fat_g": 12.8,
                "fibre_g": 2.2,
                "sodium_mg": 480.0,
                "source": "nutrition_dish_level.csv + chicken composite"
            },
            {
                "subtype_id": "kaathi_roll_egg",
                "display_name": "Single Egg Kathi Roll",
                "is_default": False,
                "serving_size_g": 150.0,
                "serving_unit": "1 roll",
                "calories_kcal": 290.0,
                "carbs_g": 34.0,
                "protein_g": 13.0,
                "fat_g": 11.5,
                "fibre_g": 2.2,
                "sodium_mg": 410.0,
                "source": "nutrition_dish_level.csv + egg composite"
            }
        ]
    },
    # 13. Chaat
    {
        "parent_class": "chaat",
        "aliases": ["chaat"],
        "disambiguation_prompt": "Which specific chaat dish was this?",
        "subtypes": [
            {
                "subtype_id": "chaat_bhel_puri",
                "display_name": "Bhel Puri (Puffed rice, sev, chutneys)",
                "is_default": True,
                "serving_size_g": 120.0,
                "serving_unit": "1 plate",
                "calories_kcal": 185.0,
                "carbs_g": 30.0,
                "protein_g": 4.2,
                "fat_g": 5.5,
                "fibre_g": 3.0,
                "sodium_mg": 380.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "chaat_sev_puri",
                "display_name": "Sev Puri (Crispy flat puris with potato & sev)",
                "is_default": False,
                "serving_size_g": 120.0,
                "serving_unit": "1 plate (6 pcs)",
                "calories_kcal": 235.0,
                "carbs_g": 28.0,
                "protein_g": 4.5,
                "fat_g": 11.8,
                "fibre_g": 2.5,
                "sodium_mg": 420.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "chaat_papdi",
                "display_name": "Dahi Papdi Chaat (Puris with yogurt & chutneys)",
                "is_default": False,
                "serving_size_g": 140.0,
                "serving_unit": "1 plate",
                "calories_kcal": 250.0,
                "carbs_g": 29.5,
                "protein_g": 5.0,
                "fat_g": 12.5,
                "fibre_g": 2.4,
                "sodium_mg": 390.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    },
    # 14. Halwa
    {
        "parent_class": "halwa",
        "aliases": ["halwa"],
        "disambiguation_prompt": "What type of halwa did you have?",
        "subtypes": [
            {
                "subtype_id": "halwa_sooji",
                "display_name": "Sooji / Rava Halwa (Semolina)",
                "is_default": True,
                "serving_size_g": 100.0,
                "serving_unit": "1 small katori",
                "calories_kcal": 285.0,
                "carbs_g": 39.5,
                "protein_g": 3.8,
                "fat_g": 12.5,
                "fibre_g": 1.1,
                "sodium_mg": 35.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "halwa_gajar",
                "display_name": "Gajar ka Halwa (Carrot Halwa)",
                "is_default": False,
                "serving_size_g": 100.0,
                "serving_unit": "1 small katori",
                "calories_kcal": 230.0,
                "carbs_g": 32.0,
                "protein_g": 4.2,
                "fat_g": 9.5,
                "fibre_g": 2.4,
                "sodium_mg": 55.0,
                "source": "nutrition_dish_level.csv"
            },
            {
                "subtype_id": "halwa_moong_dal",
                "display_name": "Moong Dal Halwa",
                "is_default": False,
                "serving_size_g": 100.0,
                "serving_unit": "1 small katori",
                "calories_kcal": 340.0,
                "carbs_g": 39.5,
                "protein_g": 8.5,
                "fat_g": 16.5,
                "fibre_g": 3.2,
                "sodium_mg": 60.0,
                "source": "nutrition_dish_level.csv"
            }
        ]
    }
]


def build_ambiguous_subtypes():
    print("=" * 65)
    print("BUILDING AMBIGUOUS SUBTYPES REGISTRY")
    print("=" * 65)

    flat_records = []
    total_subtypes = 0

    for item in SUBTYPES_REGISTRY:
        p = item["parent_class"]
        prompt = item["disambiguation_prompt"]
        aliases = ",".join(item["aliases"])
        for sub in item["subtypes"]:
            total_subtypes += 1
            row = dict(sub)
            row["parent_class"] = p
            row["aliases"] = aliases
            row["disambiguation_prompt"] = prompt
            flat_records.append(row)

    df_sub = pd.DataFrame(flat_records)

    # Validate that every parent has exactly ONE default subtype
    default_counts = df_sub[df_sub["is_default"]].groupby("parent_class").size()
    all_have_default = (default_counts == 1).all() and len(default_counts) == len(SUBTYPES_REGISTRY)

    print(f"Parent ambiguous families: {len(SUBTYPES_REGISTRY)}")
    print(f"Total discrete nutritional sub-profiles registered: {total_subtypes}")
    print(f"Every parent family has exactly 1 default profile: {all_have_default}")

    # Export
    json_path = FEATURE_DIR / "ambiguous_subtypes_registry.json"
    csv_path = FEATURE_DIR / "ambiguous_subtypes_registry.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(SUBTYPES_REGISTRY, f, indent=2)

    df_sub.to_csv(csv_path, index=False)

    print(f"\nOutputs saved:")
    print(f"  - JSON Registry: {json_path}")
    print(f"  - CSV Table:     {csv_path}")
    print("=" * 65)
    return df_sub


if __name__ == "__main__":
    build_ambiguous_subtypes()
