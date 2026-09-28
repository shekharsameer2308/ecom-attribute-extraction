#!/usr/bin/env python3
"""
Training Script — Train the NER model locally on your Mac.

This script:
  1. Generates synthetic training data (realistic product label text)
  2. Trains an Averaged Perceptron NER model on CPU
  3. Evaluates on a held-out test set
  4. Saves the trained model to models/ner_model/
  5. Runs interactive demo predictions

No GPU needed. No heavy frameworks. Trains in seconds.

Usage:
  python training/train.py
  python training/train.py --epochs 50 --samples 1000
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.generate_data import generate_dataset, save_dataset, load_dataset
from src.ner_model import AttributeNERModel


def main():
    parser = argparse.ArgumentParser(description="Train the NER model locally")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--samples", type=int, default=500, help="Training samples to generate (default: 500)")
    parser.add_argument("--test-samples", type=int, default=100, help="Test samples (default: 100)")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--output", default="models/ner_model", help="Model output directory")
    parser.add_argument("--data-dir", default="training/data", help="Training data directory")
    args = parser.parse_args()

    print("=" * 60)
    print("  🧠 Local NER Model Training Pipeline")
    print("  No API keys. No GPU. Trains on your Mac.")
    print("=" * 60)

    # ── Step 1: Generate or load training data ──
    train_path = os.path.join(args.data_dir, "train.json")
    test_path = os.path.join(args.data_dir, "test.json")

    if os.path.exists(train_path):
        print(f"\n📂 Loading existing training data from {train_path}")
        train_data = load_dataset(train_path)
    else:
        print(f"\n🔧 Generating {args.samples} synthetic training samples...")
        train_data = generate_dataset(n_samples=args.samples, seed=42)
        save_dataset(train_data, train_path)

    if os.path.exists(test_path):
        test_data = load_dataset(test_path)
    else:
        test_data = generate_dataset(args.test_samples, seed=99)
        save_dataset(test_data, test_path)

    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples:     {len(test_data)}")

    # ── Step 2: Show sample data ──
    print(f"\n📋 Sample training data:")
    for text, ann in train_data[:3]:
        print(f"   Text: {text!r}")
        ent_strs = [f"'{text[s:e]}'→{l}" for s, e, l in ann["entities"]]
        print(f"   Entities: {', '.join(ent_strs)}\n")

    # ── Step 3: Train ──
    model = AttributeNERModel()

    model.train(
        training_data=train_data,
        n_epochs=args.epochs,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    # ── Step 4: Evaluate ──
    print(f"\n📊 Evaluating on {len(test_data)} test samples...\n")
    results = model.evaluate(test_data)

    print(f"  {'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'─'*52}")
    for label, metrics in results.items():
        marker = "  " if label != "MICRO_AVG" else "→ "
        print(
            f"  {marker}{label:<10} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
            f"{metrics['f1']:>10.3f} {metrics['support']:>10}"
        )

    # ── Step 5: Interactive demo ──
    print(f"\n{'='*60}")
    print(f"  ✅ Training Complete!")
    print(f"  Model saved to: {args.output}/")
    print(f"  Micro F1: {results['MICRO_AVG']['f1']:.3f}")
    print(f"{'='*60}")

    print(f"\n🔍 Demo predictions on unseen text:")
    demo_texts = [
        "Net Wt. 500 g",
        "6 x 200 ml",
        "Buy 1 Get 1 Free Net Weight 250 grams",
        "Power: 1200 W",
        "Per 100g: Energy 520 kcal Net Content 750 ml",
        "Nestle Corn Flakes 20% Extra Free Net Wt. 875 g",
        "Pack of 12 (330 ml each)",
        "Drained Weight 200 g",
    ]

    # Reload saved model for clean predictions
    saved_model = AttributeNERModel(model_path=args.output)
    for text in demo_texts:
        entities = saved_model.predict(text)
        if entities:
            ent_str = ", ".join(f"'{e['text']}'→{e['label']}({e['score']:.2f})" for e in entities)
        else:
            ent_str = "(none detected)"
        print(f"   Input:    {text!r}")
        print(f"   Detected: {ent_str}\n")


if __name__ == "__main__":
    main()
