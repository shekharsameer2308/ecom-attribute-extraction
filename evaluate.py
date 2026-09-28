#!/usr/bin/env python3
"""
Model Evaluation & Capability Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Tests your trained NER model against real-world scenarios and
generates a clear capability report showing:

  ✅ What it CAN do
  ❌ What it CANNOT do yet
  📊 Accuracy scores per category
  🎯 Overall capability percentage

Usage:
  python evaluate.py
  python evaluate.py --model models/ner_model
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ner_model import AttributeNERModel
from src.models import ExtractionResult, ComputedAttribute
from src.local_pipeline import LocalExtractionPipeline


# ──────────────────────────────────────────────────────────────
# Test cases organized by difficulty and category
# ──────────────────────────────────────────────────────────────

TEST_SUITE = {

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 1: Basic Weight Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Basic Weight": {
        "description": "Can the model find simple weight declarations?",
        "difficulty": "Easy",
        "tests": [
            {
                "name": "Simple grams",
                "input": "Net Wt. 500 g",
                "expect_label": "WEIGHT",
                "expect_text_contains": "500",
            },
            {
                "name": "Kilograms",
                "input": "Net Weight 1.5 kg",
                "expect_label": "WEIGHT",
                "expect_text_contains": "1.5",
            },
            {
                "name": "Milligrams",
                "input": "Net Wt 250 mg",
                "expect_label": "WEIGHT",
                "expect_text_contains": "250",
            },
            {
                "name": "Ounces",
                "input": "Net Wt. 12 oz",
                "expect_label": "WEIGHT",
                "expect_text_contains": "12",
            },
            {
                "name": "No qualifier, bare weight",
                "input": "500 g",
                "expect_label": "WEIGHT",
                "expect_text_contains": "500",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 2: Volume Extraction
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Basic Volume": {
        "description": "Can the model find volume declarations?",
        "difficulty": "Easy",
        "tests": [
            {
                "name": "Milliliters",
                "input": "Net Content 355 ml",
                "expect_label": "VOLUME",
                "expect_text_contains": "355",
            },
            {
                "name": "Liters",
                "input": "Contents 2 L",
                "expect_label": "VOLUME",
                "expect_text_contains": "2",
            },
            {
                "name": "Fluid ounces",
                "input": "NET 12 fl oz",
                "expect_label": "VOLUME",
                "expect_text_contains": "12",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 3: Qualifier Detection
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Qualifier Detection": {
        "description": "Can the model identify Net/Gross/Drained labels?",
        "difficulty": "Easy",
        "tests": [
            {
                "name": "Net Weight qualifier",
                "input": "Net Wt. 500 g",
                "expect_label": "QUALIFIER",
                "expect_text_contains": "Net",
            },
            {
                "name": "Drained Weight qualifier",
                "input": "Drained Weight 200 g",
                "expect_label": "QUALIFIER",
                "expect_text_contains": "Drained",
            },
            {
                "name": "Net Content qualifier",
                "input": "Net Content 750 ml",
                "expect_label": "QUALIFIER",
                "expect_text_contains": "Net",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 4: Noisy Text (Promos + Nutrition)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Noise Filtering": {
        "description": "Can the model find the real weight and IGNORE marketing text?",
        "difficulty": "Medium",
        "tests": [
            {
                "name": "Weight mixed with promo",
                "input": "Nestle Corn Flakes 20% Extra Free Net Wt. 875 g",
                "expect_label": "WEIGHT",
                "expect_text_contains": "875",
            },
            {
                "name": "Weight with brand noise",
                "input": "Coca-Cola Classic Refreshing Taste Since 1886 Net Content 330 ml",
                "expect_label": "VOLUME",
                "expect_text_contains": "330",
            },
            {
                "name": "Ignore nutrition per-100g",
                "input": "Per 100g: Energy 520 kcal Protein 8g Fat 25g Net Wt. 750 g",
                "expect_label": "WEIGHT",
                "expect_text_contains": "750",
                "must_not_contain": "100",
            },
            {
                "name": "Ignore serving size",
                "input": "Serving Size 30g Net Weight 450 g Calories 150",
                "expect_label": "WEIGHT",
                "expect_text_contains": "450",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 5: Multipacks
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Multipacks": {
        "description": "Can the model handle multipack formats?",
        "difficulty": "Medium",
        "tests": [
            {
                "name": "Standard 6x format",
                "input": "6 x 500 g",
                "expect_label": "WEIGHT",
                "expect_text_contains": "500",
            },
            {
                "name": "Pack of N format",
                "input": "Pack of 12 330 ml each",
                "expect_label": "VOLUME",
                "expect_text_contains": "330",
            },
            {
                "name": "Pieces count",
                "input": "24 pcs Net Wt. 500 g",
                "expect_label": "WEIGHT",
                "expect_text_contains": "500",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 6: Dual Units
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Dual Units": {
        "description": "Can the model handle labels with two unit systems?",
        "difficulty": "Hard",
        "tests": [
            {
                "name": "fl oz with ml in parens",
                "input": "NET 12 FL OZ (355 ml)",
                "expect_label": "VOLUME",
                "expect_text_contains": "ml",  # should find the ml one
            },
            {
                "name": "oz with grams",
                "input": "Net Wt 8 oz (226 g)",
                "expect_label": "WEIGHT",
                "expect_text_contains": "226",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 7: Power (Electronics)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Power": {
        "description": "Can the model extract wattage for electronics?",
        "difficulty": "Medium",
        "tests": [
            {
                "name": "Simple watts",
                "input": "Power: 1200 W",
                "expect_label": "POWER",
                "expect_text_contains": "1200",
            },
            {
                "name": "Kilowatts",
                "input": "Rated Power 2.5 kW",
                "expect_label": "POWER",
                "expect_text_contains": "2.5",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 8: Unit Conversion (Validator)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Unit Conversion": {
        "description": "Does the deterministic validator correctly convert units?",
        "difficulty": "Easy",
        "type": "validator",
        "tests": [
            {
                "name": "kg → grams",
                "original_value": 1.5,
                "original_unit": "kg",
                "expected_computed": 1500.0,
                "expected_unit": "g",
            },
            {
                "name": "L → ml",
                "original_value": 2.0,
                "original_unit": "l",
                "expected_computed": 2000.0,
                "expected_unit": "ml",
            },
            {
                "name": "fl oz → ml",
                "original_value": 12.0,
                "original_unit": "fl oz",
                "expected_computed": 354.882,
                "expected_unit": "ml",
            },
            {
                "name": "mg → g",
                "original_value": 500.0,
                "original_unit": "mg",
                "expected_computed": 0.5,
                "expected_unit": "g",
            },
            {
                "name": "Multipack math: 6 x 500g",
                "original_value": 500.0,
                "original_unit": "g",
                "pack_count": 6,
                "unit_value": 500.0,
                "expected_computed": 3000.0,
                "expected_unit": "g",
            },
        ],
    },

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CATEGORY 9: Full Pipeline (Text → NER → Validator)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    "Full Pipeline": {
        "description": "End-to-end: raw text → NER → compute → final answer",
        "difficulty": "Hard",
        "type": "pipeline",
        "tests": [
            {
                "name": "Simple product text",
                "input": "Net Wt. 500 g",
                "target_attr": "net_weight",
                "expected_value": 500.0,
                "expected_unit": "g",
            },
            {
                "name": "Volume product",
                "input": "Net Content 1.5 L",
                "target_attr": "net_volume",
                "expected_value": 1500.0,
                "expected_unit": "ml",
            },
            {
                "name": "Complex noisy text",
                "input": "Kelloggs Corn Flakes Breakfast Cereal Family Pack Net Wt. 875 g 20% Extra Free",
                "target_attr": "net_weight",
                "expected_value": 875.0,
                "expected_unit": "g",
            },
        ],
    },
}


# ──────────────────────────────────────────────────────────────
# Test runners
# ──────────────────────────────────────────────────────────────

def run_ner_test(model, test):
    """Run a single NER prediction test."""
    entities = model.predict(test["input"])
    target_label = test["expect_label"]
    target_text = test["expect_text_contains"]

    # Check if any entity matches the expected label and contains expected text
    for ent in entities:
        if ent["label"] == target_label and target_text.lower() in ent["text"].lower():
            # Also check must_not_contain
            if "must_not_contain" in test:
                bad = test["must_not_contain"]
                # Make sure the primary match doesn't contain the bad text
                if bad.lower() in ent["text"].lower():
                    return False, f"Found '{ent['text']}' but it contains '{bad}'"
            return True, f"Found '{ent['text']}' → {ent['label']} (score: {ent['score']:.2f})"

    # Didn't find it
    found_str = ", ".join(f"'{e['text']}'→{e['label']}" for e in entities) or "nothing"
    return False, f"Expected {target_label} with '{target_text}', got: {found_str}"


def run_validator_test(test):
    """Run a single validator/conversion test."""
    ext = ExtractionResult(
        attribute="test",
        found=True,
        original_value=test["original_value"],
        original_unit=test["original_unit"],
        unit_value=test.get("unit_value", test["original_value"]),
        pack_count=test.get("pack_count"),
        confidence=0.90,
    )
    c = ComputedAttribute.from_extraction(ext)

    if c.computed_value is None:
        return False, "Computed value is None"

    expected = test["expected_computed"]
    if abs(c.computed_value - expected) < 0.01 and c.canonical_unit == test["expected_unit"]:
        return True, f"{test['original_value']} {test['original_unit']} → {c.computed_value} {c.canonical_unit}"
    else:
        return False, f"Expected {expected} {test['expected_unit']}, got {c.computed_value} {c.canonical_unit}"


def run_pipeline_test(pipeline, test):
    """Run a full end-to-end pipeline test."""
    result = pipeline.process_product(
        product_id="eval-test",
        raw_text=test["input"],
        target_attributes=[test["target_attr"]],
    )

    attrs = result.get("computed_attributes", [])
    if not attrs:
        return False, "No attributes returned"

    attr = attrs[0]
    computed = attr.get("computed_value")
    unit = attr.get("canonical_unit")
    expected_val = test["expected_value"]
    expected_unit = test["expected_unit"]

    if computed is not None and abs(computed - expected_val) < 0.01 and unit == expected_unit:
        return True, f"'{test['input']}' → {computed} {unit}"
    else:
        return False, f"Expected {expected_val} {expected_unit}, got {computed} {unit}"


# ──────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────

def main():
    model_path = "models/ner_model"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print()
    print("╔" + "═" * 58 + "╗")
    print("║   📊  MODEL CAPABILITY REPORT                            ║")
    print("║   E-Commerce Attribute Extraction Pipeline                ║")
    print("╚" + "═" * 58 + "╝")

    # Load model
    model = AttributeNERModel(model_path=model_path)

    # Load local pipeline for E2E tests
    pipeline = LocalExtractionPipeline(ner_model_path=model_path)

    total_passed = 0
    total_failed = 0
    category_results = {}

    for category_name, category in TEST_SUITE.items():
        cat_type = category.get("type", "ner")
        difficulty = category.get("difficulty", "?")
        description = category.get("description", "")
        tests = category.get("tests", [])

        print(f"\n{'━' * 60}")
        print(f"  📂 {category_name}  [{difficulty}]")
        print(f"     {description}")
        print(f"{'━' * 60}")

        cat_passed = 0
        cat_failed = 0

        for test in tests:
            if cat_type == "ner":
                passed, detail = run_ner_test(model, test)
            elif cat_type == "validator":
                passed, detail = run_validator_test(test)
            elif cat_type == "pipeline":
                passed, detail = run_pipeline_test(pipeline, test)
            else:
                passed, detail = False, "Unknown test type"

            if passed:
                cat_passed += 1
                total_passed += 1
                print(f"     ✅ {test['name']}")
                print(f"        {detail}")
            else:
                cat_failed += 1
                total_failed += 1
                print(f"     ❌ {test['name']}")
                print(f"        {detail}")

        cat_total = cat_passed + cat_failed
        cat_pct = (cat_passed / cat_total * 100) if cat_total > 0 else 0
        category_results[category_name] = {
            "passed": cat_passed,
            "total": cat_total,
            "pct": cat_pct,
            "difficulty": difficulty,
        }

        bar_len = 20
        filled = int(bar_len * cat_pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\n     Result: {cat_passed}/{cat_total}  [{bar}] {cat_pct:.0f}%")

    # ──────────────────────────────────────────────────────────
    # Final Report
    # ──────────────────────────────────────────────────────────

    grand_total = total_passed + total_failed
    overall_pct = (total_passed / grand_total * 100) if grand_total > 0 else 0

    print(f"\n\n{'╔' + '═' * 58 + '╗'}")
    print(f"{'║'}   🎯  OVERALL CAPABILITY SCORE                            {'║'}")
    print(f"{'╠' + '═' * 58 + '╣'}")

    bar_len = 30
    filled = int(bar_len * overall_pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"{'║'}                                                          {'║'}")
    print(f"{'║'}   [{bar}]  {overall_pct:.1f}%       {'║'}")
    print(f"{'║'}   {total_passed} passed / {grand_total} total tests                         {'║'}")
    print(f"{'║'}                                                          {'║'}")
    print(f"{'╠' + '═' * 58 + '╣'}")

    # Per-category summary
    print(f"{'║'}   Category Breakdown:                                    {'║'}")
    print(f"{'║'}   {'─' * 54} {'║'}")

    for cat_name, res in category_results.items():
        grade = "🟢" if res["pct"] >= 80 else ("🟡" if res["pct"] >= 50 else "🔴")
        line = f"   {grade} {cat_name:<22} {res['passed']}/{res['total']:>2}  ({res['pct']:>5.1f}%)  [{res['difficulty']}]"
        print(f"{'║'}{line:<58}{'║'}")

    print(f"{'║'}                                                          {'║'}")
    print(f"{'╠' + '═' * 58 + '╣'}")

    # What it can / can't do
    can_do = [name for name, r in category_results.items() if r["pct"] >= 80]
    needs_work = [name for name, r in category_results.items() if 50 <= r["pct"] < 80]
    cant_do = [name for name, r in category_results.items() if r["pct"] < 50]

    print(f"{'║'}                                                          {'║'}")
    if can_do:
        print(f"{'║'}   ✅ STRONG at:                                          {'║'}")
        for c in can_do:
            print(f"{'║'}      • {c:<49}{'║'}")
    if needs_work:
        print(f"{'║'}   🟡 NEEDS WORK:                                         {'║'}")
        for c in needs_work:
            print(f"{'║'}      • {c:<49}{'║'}")
    if cant_do:
        print(f"{'║'}   🔴 WEAK at:                                            {'║'}")
        for c in cant_do:
            print(f"{'║'}      • {c:<49}{'║'}")

    print(f"{'║'}                                                          {'║'}")
    print(f"{'║'}   💡 To improve: add more training data for weak areas   {'║'}")
    print(f"{'║'}      python training/train.py --samples 2000 --epochs 50{'║'}")
    print(f"{'║'}                                                          {'║'}")
    print(f"{'╚' + '═' * 58 + '╝'}")


if __name__ == "__main__":
    main()
