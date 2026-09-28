"""
Synthetic Training Data Generator for the NER Model.

Since we don't have thousands of labeled product images yet, we generate
realistic product label text strings with entity annotations.

This is standard practice — train on synthetic data first, then fine-tune
on real labeled data as you collect it from the human review queue.

The generator creates text that looks like what OCR would extract from
real product packaging, including:
  - Standard net weight/volume declarations
  - Multipacks
  - Dual unit labels
  - Promotional text (noise the model must learn to IGNORE)
  - Nutrition information (another noise source)
  - Various languages and formats
"""

import random
import json
import os
from typing import List, Tuple, Dict


# ──────────────────────────────────────────────────────────────
# Building blocks for synthetic label text
# ──────────────────────────────────────────────────────────────

QUALIFIERS = [
    "Net Wt.", "Net Weight", "Net Wt", "NET WT.",
    "Net", "NET", "Net Qty", "Net Quantity",
    "Net Content", "Contents", "Net Vol.",
    "Gross Wt.", "Gross Weight",
    "Drained Weight", "Drained Wt.",
]

WEIGHT_UNITS = ["g", "kg", "mg", "oz", "lb", "gm", "gms", "grams", "kilograms"]
VOLUME_UNITS = ["ml", "mL", "L", "l", "fl oz", "cl", "litre", "litres", "millilitres"]
POWER_UNITS = ["W", "kW", "watt", "watts", "Watt"]

WEIGHT_VALUES = [
    "50", "75", "100", "125", "150", "200", "250", "300", "400",
    "500", "750", "1000", "1.5", "2", "2.5", "5", "10",
    "0.5", "1", "3", "100", "200", "350", "450", "600", "800",
]

VOLUME_VALUES = [
    "100", "150", "200", "250", "330", "355", "500", "750",
    "1", "1.5", "2", "3", "5", "50", "175", "300", "600",
]

POWER_VALUES = ["5", "7", "10", "15", "25", "40", "60", "100", "500", "750", "1000", "1200", "1500", "2000"]

PACK_COUNTS = ["2", "3", "4", "5", "6", "8", "10", "12", "24"]

PROMO_TEXTS = [
    "20% Extra Free", "25% More", "Buy 1 Get 1 Free",
    "BOGO", "Extra 50g Free", "+100ml FREE",
    "New Improved Formula", "Family Pack", "Value Pack",
    "Limited Edition", "Save 30%", "Special Offer",
    "50% Extra", "Bonus Pack", "Jumbo Size",
]

BRAND_NAMES = [
    "Nestle", "Kelloggs", "PepsiCo", "Coca-Cola", "Unilever",
    "P&G", "Britannia", "Amul", "Haldiram's", "Parle",
    "ITC", "Dabur", "Tata", "Godrej", "Himalaya",
]

PRODUCT_NAMES = [
    "Corn Flakes", "Milk Chocolate", "Orange Juice", "Mineral Water",
    "Potato Chips", "Instant Noodles", "Hair Oil", "Face Cream",
    "Washing Powder", "Floor Cleaner", "Basmati Rice", "Whole Wheat Atta",
    "Green Tea", "Mixed Fruit Jam", "Peanut Butter", "Tomato Ketchup",
]

NUTRITION_NOISE = [
    "Per 100g: Energy 520 kcal, Protein 8g, Fat 25g",
    "Serving Size 30g (about 1 cup)",
    "Calories per serving: 150",
    "Per 100ml: Sugar 11g",
    "Daily Value: Vitamin C 45%",
    "Dietary Fibre 3.5g per serving",
]

APPROX_MODIFIERS = ["approx.", "approximately", "about", "~", "avg.", "average", "typical"]
MIN_MODIFIERS = ["min.", "minimum", "not less than"]


# ──────────────────────────────────────────────────────────────
# Data generation functions
# ──────────────────────────────────────────────────────────────

def _make_entity(text: str, full_text: str, label: str, start_offset: int = 0):
    """Find the entity span in the full text and return (start, end, label)."""
    start = full_text.find(text, start_offset)
    if start == -1:
        return None
    return (start, start + len(text), label)


def generate_simple_weight() -> Tuple[str, Dict]:
    """Generate: 'Net Wt. 500 g' style text with annotations."""
    qualifier = random.choice(QUALIFIERS[:8])  # net-type qualifiers
    value = random.choice(WEIGHT_VALUES)
    unit = random.choice(WEIGHT_UNITS[:4])  # common units
    
    weight_str = f"{value} {unit}"
    text = f"{qualifier} {weight_str}"
    
    entities = [
        _make_entity(qualifier, text, "QUALIFIER"),
        _make_entity(weight_str, text, "WEIGHT"),
    ]
    entities = [e for e in entities if e is not None]
    return (text, {"entities": entities})


def generate_multipack() -> Tuple[str, Dict]:
    """Generate: '6 x 500 g' or 'Pack of 6 (500 g each)' style."""
    count = random.choice(PACK_COUNTS)
    value = random.choice(WEIGHT_VALUES[:12])
    unit = random.choice(WEIGHT_UNITS[:3])
    
    formats = [
        (f"{count} x {value} {unit}",),
        (f"Pack of {count} ({value} {unit} each)",),
        (f"{value} {unit} x {count}",),
    ]
    text = random.choice(formats)[0]
    
    count_str = count
    weight_str = f"{value} {unit}"
    
    entities = []
    # Find count entity
    e = _make_entity(count_str, text, "COUNT")
    if e:
        entities.append(e)
    # Find weight entity
    e = _make_entity(weight_str, text, "WEIGHT")
    if e:
        entities.append(e)
    
    return (text, {"entities": entities})


def generate_volume() -> Tuple[str, Dict]:
    """Generate volume declarations."""
    qualifier = random.choice(["Net Vol.", "Net Content", "Contents", "NET"])
    value = random.choice(VOLUME_VALUES)
    unit = random.choice(VOLUME_UNITS[:4])
    
    vol_str = f"{value} {unit}"
    text = f"{qualifier} {vol_str}"
    
    entities = [
        _make_entity(qualifier, text, "QUALIFIER"),
        _make_entity(vol_str, text, "VOLUME"),
    ]
    entities = [e for e in entities if e is not None]
    return (text, {"entities": entities})


def generate_dual_unit() -> Tuple[str, Dict]:
    """Generate: 'NET 12 FL OZ (355 ml)' style dual-unit labels."""
    imperial_val = random.choice(["8", "12", "16", "20", "32", "64"])
    metric_val = random.choice(VOLUME_VALUES)
    
    vol1 = f"{imperial_val} fl oz"
    vol2 = f"{metric_val} ml"
    text = f"NET {vol1} ({vol2})"
    
    entities = [
        _make_entity("NET", text, "QUALIFIER"),
        _make_entity(vol1, text, "VOLUME"),
        _make_entity(vol2, text, "VOLUME", start_offset=text.find("(")),
    ]
    entities = [e for e in entities if e is not None]
    return (text, {"entities": entities})


def generate_with_promo_noise() -> Tuple[str, Dict]:
    """Generate text with promotional noise that the model should NOT extract."""
    # Real attribute
    qualifier = random.choice(QUALIFIERS[:4])
    value = random.choice(WEIGHT_VALUES[:10])
    unit = random.choice(WEIGHT_UNITS[:2])
    weight_str = f"{value} {unit}"
    
    # Promo noise
    promo = random.choice(PROMO_TEXTS)
    brand = random.choice(BRAND_NAMES)
    product = random.choice(PRODUCT_NAMES)
    
    order = random.choice([
        f"{brand} {product} {promo} {qualifier} {weight_str}",
        f"{qualifier} {weight_str} {brand} {product} {promo}",
        f"{brand} {product} {qualifier} {weight_str} {promo}",
    ])
    text = order
    
    entities = [
        _make_entity(qualifier, text, "QUALIFIER"),
        _make_entity(weight_str, text, "WEIGHT"),
        _make_entity(promo, text, "PROMO"),
    ]
    entities = [e for e in entities if e is not None]
    return (text, {"entities": entities})


def generate_with_nutrition_noise() -> Tuple[str, Dict]:
    """
    Generate text containing BOTH a net quantity AND nutrition info.
    The model must learn that nutrition info is NOT the net quantity.
    """
    qualifier = random.choice(QUALIFIERS[:4])
    value = random.choice(WEIGHT_VALUES[:10])
    unit = random.choice(WEIGHT_UNITS[:2])
    weight_str = f"{value} {unit}"
    
    nutrition = random.choice(NUTRITION_NOISE)
    
    text = f"{qualifier} {weight_str} {nutrition}"
    
    entities = [
        _make_entity(qualifier, text, "QUALIFIER"),
        _make_entity(weight_str, text, "WEIGHT"),
        # Nutrition text is deliberately NOT annotated — the model learns to skip it
    ]
    entities = [e for e in entities if e is not None]
    return (text, {"entities": entities})


def generate_power() -> Tuple[str, Dict]:
    """Generate power declarations for electronics."""
    value = random.choice(POWER_VALUES)
    unit = random.choice(POWER_UNITS[:2])
    power_str = f"{value} {unit}"
    
    prefixes = ["Rated Power:", "Power:", "Max Power", "Power Consumption"]
    prefix = random.choice(prefixes)
    
    text = f"{prefix} {power_str}"
    entities = [
        _make_entity(power_str, text, "POWER"),
    ]
    entities = [e for e in entities if e is not None]
    return (text, {"entities": entities})


def generate_approximate() -> Tuple[str, Dict]:
    """Generate: 'Net Wt. approx. 500 g' style."""
    qualifier = random.choice(QUALIFIERS[:4])
    modifier = random.choice(APPROX_MODIFIERS)
    value = random.choice(WEIGHT_VALUES[:10])
    unit = random.choice(WEIGHT_UNITS[:2])
    weight_str = f"{value} {unit}"
    
    text = f"{qualifier} {modifier} {weight_str}"
    
    entities = [
        _make_entity(qualifier, text, "QUALIFIER"),
        _make_entity(weight_str, text, "WEIGHT"),
    ]
    entities = [e for e in entities if e is not None]
    return (text, {"entities": entities})


# ──────────────────────────────────────────────────────────────
# Main generator
# ──────────────────────────────────────────────────────────────

GENERATORS = [
    (generate_simple_weight, 0.25),
    (generate_multipack, 0.15),
    (generate_volume, 0.15),
    (generate_dual_unit, 0.08),
    (generate_with_promo_noise, 0.15),
    (generate_with_nutrition_noise, 0.10),
    (generate_power, 0.05),
    (generate_approximate, 0.07),
]


def generate_dataset(n_samples: int = 500, seed: int = 42) -> List[Tuple[str, Dict]]:
    """
    Generate a synthetic training dataset.

    Args:
        n_samples: Number of training examples to generate.
        seed:      Random seed for reproducibility.

    Returns:
        List of (text, {"entities": [(start, end, label), ...]}) tuples.
    """
    random.seed(seed)
    
    generators = [g for g, _ in GENERATORS]
    weights = [w for _, w in GENERATORS]
    
    dataset = []
    for _ in range(n_samples):
        gen = random.choices(generators, weights=weights, k=1)[0]
        try:
            sample = gen()
            # Validate: no overlapping entities
            entities = sample[1]["entities"]
            valid = True
            sorted_ents = sorted(entities, key=lambda e: e[0])
            for i in range(len(sorted_ents) - 1):
                if sorted_ents[i][1] > sorted_ents[i + 1][0]:
                    valid = False
                    break
            if valid and entities:
                dataset.append(sample)
        except Exception:
            continue
    
    return dataset


def save_dataset(dataset: List[Tuple[str, Dict]], filepath: str):
    """Save dataset to a JSON file."""
    serializable = []
    for text, annotations in dataset:
        serializable.append({
            "text": text,
            "entities": annotations["entities"],
        })
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Saved {len(serializable)} samples to {filepath}")


def load_dataset(filepath: str) -> List[Tuple[str, Dict]]:
    """Load dataset from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["text"], {"entities": item["entities"]}) for item in data]


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Synthetic Training Data Generator")
    print("=" * 60)

    train_data = generate_dataset(n_samples=400, seed=42)
    test_data = generate_dataset(n_samples=100, seed=99)

    save_dataset(train_data, "training/data/train.json")
    save_dataset(test_data, "training/data/test.json")

    print(f"\n  Sample training examples:")
    for text, ann in train_data[:5]:
        print(f"    Text: {text!r}")
        print(f"    Entities: {ann['entities']}")
        print()
