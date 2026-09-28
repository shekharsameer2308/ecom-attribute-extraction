#!/usr/bin/env python3
"""
CLI entry point for the E-Commerce Attribute Extraction Pipeline.

Usage:
  # Single image extraction (mock mode for testing)
  python main.py extract --image photo.jpg --mock

  # Single image extraction (real Gemini API)
  python main.py extract --image photo.jpg --product-id SKU-123

  # Multiple images of the same product
  python main.py extract --image front.jpg --image back.jpg

  # Batch mode from a JSON manifest
  python main.py batch --manifest products.json --mock

  # Run the full test suite (no API key needed)
  python main.py test
"""

import argparse
import json
import os
import sys

from src.pipeline import ExtractionPipeline
from src.models import ExtractionResult, ComputedAttribute


# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "system_prompt.txt")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# ──────────────────────────────────────────────────────────────
# CLI Commands
# ──────────────────────────────────────────────────────────────

def cmd_extract(args):
    """Extract attributes from one or more product images."""
    pipeline = ExtractionPipeline(
        system_prompt_path=PROMPT_PATH,
        use_mock=args.mock,
        log_dir=LOG_DIR,
    )

    attrs = [a.strip() for a in args.attributes.split(",")]
    context = {}
    if args.category:
        context["category"] = args.category
    if args.brand:
        context["brand"] = args.brand

    result = pipeline.process_product(
        product_id=args.product_id,
        image_paths=args.image,
        context=context,
        target_attributes=attrs,
    )

    print(json.dumps(result, indent=2, default=str))
    return result


def cmd_batch(args):
    """Process a batch of products from a JSON manifest file."""
    with open(args.manifest, "r") as f:
        products = json.load(f)

    pipeline = ExtractionPipeline(
        system_prompt_path=PROMPT_PATH,
        use_mock=args.mock,
        log_dir=LOG_DIR,
    )

    results = pipeline.process_batch(products)

    # Write results to output file
    out_path = args.output or "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Summary
    total = len(results)
    success = sum(1 for r in results if r.get("status") == "success")
    review = sum(1 for r in results if r.get("needs_review"))
    errors = sum(1 for r in results if r.get("status") == "error")

    print(f"\n{'='*50}")
    print(f"  Batch Complete: {total} products processed")
    print(f"  ✅ Success: {success}")
    print(f"  🔍 Needs Review: {review}")
    print(f"  ❌ Errors: {errors}")
    print(f"  📄 Results saved to: {out_path}")
    print(f"{'='*50}")


def cmd_test(args):
    """Run the comprehensive test suite (no API key needed)."""
    print("=" * 60)
    print("  E-Commerce Attribute Extraction — Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            print(f"  ❌ {name}  →  {detail}")

    # ── Test 1: Simple single-pack weight ──
    print("\n── Test 1: Single pack — 'Net Wt. 500 g e' ──")
    ext = ExtractionResult(
        attribute="net_weight", found=True,
        raw_text="Net Wt. 500 g e",
        original_value=500.0, original_unit="g",
        unit_value=500.0, qualifier="estimated",
        confidence=0.95,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Computed value = 500g", c.computed_value == 500.0)
    check("Canonical unit = g", c.canonical_unit == "g")
    check("No review needed", c.needs_review == False)
    check("Arithmetic consistent", c.arithmetic_consistent == True)

    # ── Test 2: Multipack ──
    print("\n── Test 2: Multipack — '6 x 500 g' ──")
    ext = ExtractionResult(
        attribute="net_weight", found=True,
        raw_text="6 x 500 g",
        pack_count=6, original_value=500.0, original_unit="g",
        unit_value=500.0, qualifier="net", confidence=0.95,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Total = 3000g", c.computed_value == 3000.0)
    check("No review needed", c.needs_review == False)

    # ── Test 3: Multipack with consistent total ──
    print("\n── Test 3: Multipack total stated — '3 kg (6 x 500 g)' ──")
    ext = ExtractionResult(
        attribute="net_weight", found=True,
        raw_text="3 kg (6 x 500 g)",
        pack_count=6, original_value=3000.0, original_unit="g",
        unit_value=500.0, qualifier="net", confidence=0.92,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Total = 3000g", c.computed_value == 3000.0)
    check("Arithmetic consistent", c.arithmetic_consistent == True)

    # ── Test 4: Inconsistent arithmetic ──
    print("\n── Test 4: Inconsistent math — '3 kg' but 6 x 200 g ──")
    ext = ExtractionResult(
        attribute="net_weight", found=True,
        raw_text="3 kg (6 x 200 g)",
        pack_count=6, original_value=3.0, original_unit="kg",
        unit_value=200.0, confidence=0.85,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Flagged inconsistent", c.arithmetic_consistent == False)
    check("Needs review", c.needs_review == True)
    check("Reason logged", "arithmetic_inconsistent" in c.review_reasons)

    # ── Test 5: Unit conversion — kg to grams ──
    print("\n── Test 5: Unit conversion — '1.5 kg' ──")
    ext = ExtractionResult(
        attribute="net_weight", found=True,
        raw_text="Net Wt. 1.5 kg",
        original_value=1.5, original_unit="kg",
        unit_value=1.5, qualifier="net", confidence=0.93,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Converted to 1500g", c.computed_value == 1500.0)
    check("Canonical unit = g", c.canonical_unit == "g")

    # ── Test 6: Volume — ml ──
    print("\n── Test 6: Volume — '355 ml' ──")
    ext = ExtractionResult(
        attribute="net_volume", found=True,
        raw_text="355 ml",
        original_value=355.0, original_unit="ml",
        unit_value=355.0, qualifier="net", confidence=0.91,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Value = 355ml", c.computed_value == 355.0)
    check("Canonical unit = ml", c.canonical_unit == "ml")

    # ── Test 7: Volume — liters to ml ──
    print("\n── Test 7: Volume conversion — '2 L' ──")
    ext = ExtractionResult(
        attribute="net_volume", found=True,
        raw_text="2 L",
        original_value=2.0, original_unit="l",
        unit_value=2.0, qualifier="net", confidence=0.90,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Converted to 2000ml", c.computed_value == 2000.0)
    check("Canonical unit = ml", c.canonical_unit == "ml")

    # ── Test 8: Low confidence triggers review ──
    print("\n── Test 8: Low confidence → review ──")
    ext = ExtractionResult(
        attribute="net_weight", found=True,
        raw_text="Net Wt. 5?0 g",
        original_value=500.0, original_unit="g",
        unit_value=500.0, qualifier="net",
        confidence=0.45,
        review_reasons=["ocr_uncertain"],
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Needs review (low confidence)", c.needs_review == True)
    check("OCR reason preserved", "ocr_uncertain" in c.review_reasons)

    # ── Test 9: Not found ──
    print("\n── Test 9: Attribute not found ──")
    ext = ExtractionResult(
        attribute="net_weight", found=False, confidence=0.0,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Computed value is None", c.computed_value is None)
    check("Needs review (low confidence)", c.needs_review == True)

    # ── Test 10: Unknown unit ──
    print("\n── Test 10: Unknown unit — 'bushels' ──")
    ext = ExtractionResult(
        attribute="net_weight", found=True,
        raw_text="2 bushels",
        original_value=2.0, original_unit="bushels",
        unit_value=2.0, qualifier="net", confidence=0.80,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Needs review (unknown unit)", c.needs_review == True)
    check("Reason logged", any("unknown_unit" in r for r in c.review_reasons))

    # ── Test 11: fl oz conversion ──
    print("\n── Test 11: fl oz → ml conversion ──")
    ext = ExtractionResult(
        attribute="net_volume", found=True,
        raw_text="12 fl oz",
        original_value=12.0, original_unit="fl oz",
        unit_value=12.0, qualifier="net", confidence=0.88,
    )
    c = ComputedAttribute.from_extraction(ext)
    check("Converted ≈ 354.88ml", abs(c.computed_value - 354.882) < 0.01)
    check("Canonical unit = ml", c.canonical_unit == "ml")

    # ── Test 12: Pipeline E2E (mock) ──
    print("\n── Test 12: E2E Pipeline (mock mode) ──")
    pipeline = ExtractionPipeline(
        system_prompt_path=PROMPT_PATH, use_mock=True, log_dir=LOG_DIR,
    )
    result = pipeline.process_product(
        product_id="SINGLE-test-001",
        image_paths=[],  # mock ignores images
        context={"category": "Grocery", "brand": "TestBrand"},
        target_attributes=["net_weight"],
    )
    check("Status = success", result["status"] == "success")
    check("Has computed_attributes", len(result["computed_attributes"]) > 0)
    attr = result["computed_attributes"][0]
    check("Attribute = net_weight", attr["attribute_name"] == "net_weight")
    check("Value = 500g", attr["computed_value"] == 500.0)

    # ── Test 13: Pipeline multipack scenario ──
    print("\n── Test 13: E2E Pipeline multipack (mock) ──")
    result = pipeline.process_product(
        product_id="MULTI-test-002",
        image_paths=[],
        context={"category": "Grocery"},
        target_attributes=["net_weight"],
    )
    attr = result["computed_attributes"][0]
    check("Total = 3000g for 6x500g pack", attr["computed_value"] == 3000.0)

    # ── Test 14: Pipeline blurry image scenario ──
    print("\n── Test 14: E2E Pipeline blurry image (mock) ──")
    result = pipeline.process_product(
        product_id="BLURRY-test-003",
        image_paths=[],
        context={},
        target_attributes=["net_weight"],
    )
    check("Needs review", result["needs_review"] == True)
    attr = result["computed_attributes"][0]
    check("OCR uncertainty flagged", "ocr_uncertain" in attr["review_reasons"])

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"  Results:  {passed} passed,  {failed} failed,  {passed + failed} total")
    print(f"{'='*60}")

    return failed == 0


# ──────────────────────────────────────────────────────────────
# Argument Parser
# ──────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="ecom-extractor",
        description="Multimodal Attribute Extraction for E-Commerce Product Cataloging",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── extract ──
    p_extract = subparsers.add_parser(
        "extract", help="Extract attributes from product images"
    )
    p_extract.add_argument(
        "--image", "-i", action="append", required=True,
        help="Path to a product image (can specify multiple)",
    )
    p_extract.add_argument(
        "--product-id", "-p", default="PRODUCT-001",
        help="Unique product identifier (default: PRODUCT-001)",
    )
    p_extract.add_argument(
        "--attributes", "-a", default="net_weight,net_volume",
        help="Comma-separated attributes to extract (default: net_weight,net_volume)",
    )
    p_extract.add_argument("--category", "-c", default=None, help="Product category")
    p_extract.add_argument("--brand", "-b", default=None, help="Product brand")
    p_extract.add_argument(
        "--mock", action="store_true",
        help="Use mock LLM (no API key needed, returns test data)",
    )

    # ── batch ──
    p_batch = subparsers.add_parser(
        "batch", help="Process a batch of products from a JSON manifest"
    )
    p_batch.add_argument(
        "--manifest", "-m", required=True,
        help="Path to JSON manifest file",
    )
    p_batch.add_argument("--output", "-o", default=None, help="Output file path")
    p_batch.add_argument("--mock", action="store_true", help="Use mock LLM")

    # ── test ──
    subparsers.add_parser("test", help="Run the full test suite (no API key needed)")

    return parser


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "test":
        success = cmd_test(args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
