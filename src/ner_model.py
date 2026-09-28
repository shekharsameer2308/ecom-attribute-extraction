"""
NER Model — Trainable Named Entity Recognition for product attributes.

This is a lightweight, locally-trained model that works on ANY Python version.
No spaCy, no heavy ML frameworks required. Uses a trainable pattern-matching
approach with learned confidence weights.

HOW THE TRAINING WORKS:
━━━━━━━━━━━━━━━━━━━━━━
This model learns from labeled examples. Each training example is a piece of
text (like what OCR extracts from packaging) with human-annotated entity spans.

During training, the model:
  1. Extracts features from each labeled entity (word patterns, surrounding
     context, position in text, character patterns)
  2. Builds a "pattern memory" — a database of text patterns associated with
     each entity type
  3. Learns confidence weights — which features are most predictive for each
     entity type (e.g., "Net Wt." before a number → high weight for WEIGHT)

During prediction:
  1. Scans input text for candidate spans using regex
  2. Extracts the same features for each candidate
  3. Scores each candidate against the pattern memory
  4. Returns the highest-scoring entity classification

This is similar to how a simple Naive Bayes or logistic regression NER works,
but with domain-specific features designed for product labels.

Entity types:
  WEIGHT     → "500 g", "1.5 kg", "12 oz"
  VOLUME     → "355 ml", "2 L", "12 fl oz"
  POWER      → "1200 W", "2.5 kW"
  COUNT      → "6 pcs", "pack of 12", "6 x"
  QUALIFIER  → "Net Wt.", "Net", "Gross", "Drained", "approx."
  PROMO      → "20% Extra Free", "Buy 1 Get 1", "+50g free"
"""

import os
import re
import json
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# Entity labels
# ──────────────────────────────────────────────────────────────

ENTITY_LABELS = ["WEIGHT", "VOLUME", "POWER", "COUNT", "QUALIFIER", "PROMO"]


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

# Patterns that help identify entity types
WEIGHT_UNITS_RE = re.compile(r'\b(g|gm|gms|grams|kg|kilograms|mg|oz|lb|lbs)\b', re.I)
VOLUME_UNITS_RE = re.compile(r'\b(ml|mL|cl|l|L|litre|litres|millilitres|fl\s*oz)\b', re.I)
POWER_UNITS_RE = re.compile(r'\b(W|kW|watt|watts|Watt)\b')
COUNT_PATTERNS_RE = re.compile(r'\b(pcs|pieces|pack|count|units)\b|\b\d+\s*x\s', re.I)
QUALIFIER_RE = re.compile(r'\b(net\s*w|net\s*q|net\s*c|net\s*vol|contents?|gross|drained|approx|minimum|min\.)\b', re.I)
PROMO_RE = re.compile(r'(extra\s*free|buy\s*\d|get\s*\d|bogo|%\s*(more|extra|off|free)|bonus|save|offer|free)', re.I)
NUMBER_RE = re.compile(r'\d+(?:[.,]\d+)?')


def extract_features(text: str, full_context: str = "") -> Dict[str, float]:
    """
    Extract a feature vector from a text span.
    Features are designed to distinguish product attributes from noise.
    """
    features = {}
    text_lower = text.lower()

    # Unit presence features
    features["has_weight_unit"] = 1.0 if WEIGHT_UNITS_RE.search(text) else 0.0
    features["has_volume_unit"] = 1.0 if VOLUME_UNITS_RE.search(text) else 0.0
    features["has_power_unit"] = 1.0 if POWER_UNITS_RE.search(text) else 0.0
    features["has_count_pattern"] = 1.0 if COUNT_PATTERNS_RE.search(text) else 0.0
    features["has_qualifier"] = 1.0 if QUALIFIER_RE.search(text) else 0.0
    features["has_promo"] = 1.0 if PROMO_RE.search(text) else 0.0
    features["has_number"] = 1.0 if NUMBER_RE.search(text) else 0.0

    # Length features
    features["len_chars"] = min(len(text) / 50.0, 1.0)
    features["len_words"] = min(len(text.split()) / 10.0, 1.0)

    # Position in context
    if full_context:
        pos = full_context.find(text)
        if pos >= 0:
            features["position_ratio"] = pos / max(len(full_context), 1)
        else:
            features["position_ratio"] = 0.5
    else:
        features["position_ratio"] = 0.5

    # Character composition
    digits = sum(1 for c in text if c.isdigit())
    alphas = sum(1 for c in text if c.isalpha())
    total = max(len(text), 1)
    features["digit_ratio"] = digits / total
    features["alpha_ratio"] = alphas / total

    # Specific keyword features
    features["has_percent"] = 1.0 if "%" in text else 0.0
    features["has_x_separator"] = 1.0 if re.search(r'\d\s*x\s*\d', text, re.I) else 0.0
    features["starts_with_number"] = 1.0 if re.match(r'\d', text.strip()) else 0.0

    return features


# ──────────────────────────────────────────────────────────────
# Trainable NER Model
# ──────────────────────────────────────────────────────────────

class AttributeNERModel:
    """
    A trainable NER model for product attribute extraction.

    Training:
      model = AttributeNERModel()
      model.train(training_data, n_epochs=30)

    Prediction:
      entities = model.predict("Net Wt. 500 g")
      # [{"text": "Net Wt.", "label": "QUALIFIER"}, {"text": "500 g", "label": "WEIGHT"}]
    """

    def __init__(self, model_path: Optional[str] = None):
        # Per-label feature weights (this is what gets "trained")
        self.weights: Dict[str, Dict[str, float]] = {}
        # Per-label bias
        self.biases: Dict[str, float] = {}
        # Training stats
        self.training_stats: Dict = {}

        if model_path and os.path.exists(model_path):
            self._load(model_path)
        else:
            # Initialize with zeros
            for label in ENTITY_LABELS:
                self.weights[label] = {}
                self.biases[label] = 0.0

    # ──────────────────────────────────────────────────────────
    # Prediction
    # ──────────────────────────────────────────────────────────

    def predict(self, text: str) -> List[Dict]:
        """
        Run NER on text. Returns list of detected entities with labels.
        """
        # Step 1: Find candidate spans using regex
        candidates = self._find_candidates(text)

        # Step 2: Score each candidate against each label
        results = []
        for span_text, start, end in candidates:
            features = extract_features(span_text, text)
            best_label = None
            best_score = 0.0

            for label in ENTITY_LABELS:
                score = self._score(features, label)
                if score > best_score:
                    best_score = score
                    best_label = label

            if best_label and best_score > 0.3:  # threshold
                results.append({
                    "text": span_text,
                    "label": best_label,
                    "start": start,
                    "end": end,
                    "score": round(best_score, 4),
                })

        # Remove overlapping entities (keep highest scoring)
        results = self._remove_overlaps(results)
        return results

    def _score(self, features: Dict[str, float], label: str) -> float:
        """Compute score = sigmoid(sum(weight_i * feature_i) + bias)"""
        w = self.weights.get(label, {})
        total = self.biases.get(label, 0.0)
        for feat_name, feat_val in features.items():
            total += w.get(feat_name, 0.0) * feat_val
        # Sigmoid to get 0-1 score
        return 1.0 / (1.0 + math.exp(-max(min(total, 20), -20)))

    def _find_candidates(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Find candidate entity spans in the text using regex patterns.
        Returns list of (text, start, end) tuples.
        """
        candidates = []
        seen = set()

        # Pattern: number + unit (e.g., "500 g", "1.5 kg", "12 fl oz")
        for m in re.finditer(r'\d+(?:[.,]\d+)?\s*(?:g|gm|gms|grams|kg|kilograms|mg|oz|lb|lbs|ml|mL|cl|l|L|litre|litres|millilitres|fl\s*oz|W|kW|watt|watts|pcs|pieces)\b', text, re.I):
            key = (m.start(), m.end())
            if key not in seen:
                candidates.append((m.group(), m.start(), m.end()))
                seen.add(key)

        # Pattern: count expressions (e.g., "6 x", "pack of 12")
        for m in re.finditer(r'\d+\s*x\b|\bpack\s*of\s*\d+|\b\d+\s*pcs\b', text, re.I):
            key = (m.start(), m.end())
            if key not in seen:
                candidates.append((m.group(), m.start(), m.end()))
                seen.add(key)

        # Pattern: qualifiers (e.g., "Net Wt.", "Net Content")
        for m in re.finditer(r'(?:Net\s*(?:Wt\.?|Weight|Qty\.?|Quantity|Content|Vol\.?)|Contents?|Gross\s*(?:Wt\.?|Weight)|Drained\s*(?:Wt\.?|Weight)|approx\.?|about|minimum|min\.)', text, re.I):
            key = (m.start(), m.end())
            if key not in seen:
                candidates.append((m.group(), m.start(), m.end()))
                seen.add(key)

        # Pattern: promotional text
        for m in re.finditer(r'(?:\d+%\s*(?:Extra|More|Off)\s*Free?|Buy\s*\d+\s*Get\s*\d+\s*Free|BOGO|Extra\s*\d+\s*(?:g|ml)\s*Free|\+\d+\s*(?:g|ml)\s*(?:free|FREE))', text, re.I):
            key = (m.start(), m.end())
            if key not in seen:
                candidates.append((m.group(), m.start(), m.end()))
                seen.add(key)

        return candidates

    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        """Remove overlapping entities, keeping the highest-scoring one."""
        if not entities:
            return entities

        entities.sort(key=lambda e: e["score"], reverse=True)
        kept = []
        for ent in entities:
            overlap = False
            for k in kept:
                if ent["start"] < k["end"] and ent["end"] > k["start"]:
                    overlap = True
                    break
            if not overlap:
                kept.append(ent)

        kept.sort(key=lambda e: e["start"])
        return kept

    # ──────────────────────────────────────────────────────────
    # Training (Perceptron-style online learning)
    # ──────────────────────────────────────────────────────────

    def train(
        self,
        training_data: List[Tuple[str, Dict]],
        n_epochs: int = 30,
        learning_rate: float = 0.1,
        output_dir: str = "models/ner_model",
    ):
        """
        Train the model using labeled data.

        HOW IT WORKS:
        ─────────────
        For each training example:
          1. The model predicts entities
          2. We compare predictions to ground truth labels
          3. For MISSED entities (false negatives):
             → Increase weights for that label's features
          4. For WRONG entities (false positives):
             → Decrease weights for that label's features
          5. For CORRECT entities:
             → Reinforce the weights slightly

        This is a variant of the Perceptron algorithm, which is one of the
        simplest but most effective training methods for NER.

        Args:
            training_data: List of (text, {"entities": [(start, end, label)]})
            n_epochs:      Number of passes over the data
            learning_rate: How much to adjust weights per update
            output_dir:    Where to save the trained model
        """
        import random

        print(f"\n  Training NER model...")
        print(f"  {'─'*45}")
        print(f"  Training samples  : {len(training_data)}")
        print(f"  Epochs            : {n_epochs}")
        print(f"  Learning rate     : {learning_rate}")
        print(f"  Entity labels     : {ENTITY_LABELS}")
        print(f"  Algorithm         : Averaged Perceptron")
        print(f"  {'─'*45}\n")

        # Accumulated weights for averaging (reduces overfitting)
        acc_weights = {label: defaultdict(float) for label in ENTITY_LABELS}
        acc_biases = {label: 0.0 for label in ENTITY_LABELS}
        update_count = 0

        for epoch in range(n_epochs):
            random.shuffle(training_data)
            correct = 0
            total = 0
            errors = 0

            for text, annotations in training_data:
                gold_entities = set()
                for start, end, label in annotations.get("entities", []):
                    gold_entities.add((start, end, label))

                # Find candidates in this text
                candidates = self._find_candidates(text)

                for span_text, c_start, c_end in candidates:
                    features = extract_features(span_text, text)

                    # What did the model predict?
                    pred_label = None
                    pred_score = 0.0
                    for label in ENTITY_LABELS:
                        s = self._score(features, label)
                        if s > pred_score:
                            pred_score = s
                            pred_label = label

                    # What is the ground truth?
                    true_label = None
                    for gs, ge, gl in gold_entities:
                        # Allow some flexibility in span matching
                        if abs(c_start - gs) <= 2 and abs(c_end - ge) <= 2:
                            true_label = gl
                            break

                    total += 1
                    update_count += 1

                    if true_label and pred_label == true_label and pred_score > 0.3:
                        # Correct prediction — small reinforcement
                        correct += 1
                        for feat, val in features.items():
                            self.weights[true_label][feat] = self.weights[true_label].get(feat, 0) + learning_rate * 0.1 * val
                            acc_weights[true_label][feat] += self.weights[true_label][feat]

                    elif true_label and pred_label != true_label:
                        # Wrong label — decrease wrong, increase correct
                        errors += 1
                        for feat, val in features.items():
                            # Promote true label
                            self.weights[true_label][feat] = self.weights[true_label].get(feat, 0) + learning_rate * val
                            acc_weights[true_label][feat] += self.weights[true_label][feat]

                            # Demote predicted label
                            if pred_label:
                                self.weights[pred_label][feat] = self.weights[pred_label].get(feat, 0) - learning_rate * val
                                acc_weights[pred_label][feat] += self.weights[pred_label][feat]

                    elif true_label and pred_score <= 0.3:
                        # Missed (false negative) — boost the true label
                        errors += 1
                        for feat, val in features.items():
                            self.weights[true_label][feat] = self.weights[true_label].get(feat, 0) + learning_rate * val
                            acc_weights[true_label][feat] += self.weights[true_label][feat]
                        self.biases[true_label] = self.biases.get(true_label, 0) + learning_rate
                        acc_biases[true_label] += self.biases[true_label]

                    elif not true_label and pred_score > 0.3:
                        # False positive — demote
                        errors += 1
                        if pred_label:
                            for feat, val in features.items():
                                self.weights[pred_label][feat] = self.weights[pred_label].get(feat, 0) - learning_rate * val
                                acc_weights[pred_label][feat] += self.weights[pred_label][feat]

            accuracy = correct / max(total, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}  |  Accuracy: {accuracy:.3f}  |  Errors: {errors}  |  Total: {total}")

        # Use averaged weights (reduces overfitting)
        if update_count > 0:
            for label in ENTITY_LABELS:
                for feat in acc_weights[label]:
                    self.weights[label][feat] = acc_weights[label][feat] / update_count
                self.biases[label] = acc_biases[label] / update_count

        # Save the model
        self._save(output_dir)

        self.training_stats = {
            "epochs": n_epochs,
            "samples": len(training_data),
            "final_accuracy": round(accuracy, 3),
        }

        print(f"\n  ✅ Model saved to: {output_dir}/")
        print(f"  Final accuracy: {accuracy:.3f}")

    # ──────────────────────────────────────────────────────────
    # Evaluation
    # ──────────────────────────────────────────────────────────

    def evaluate(self, test_data: List[Tuple[str, Dict]]) -> Dict:
        """Evaluate and return per-label precision, recall, F1."""
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for text, annotations in test_data:
            predicted = self.predict(text)
            pred_set = {(e["start"], e["end"], e["label"]) for e in predicted}
            gold_set = {(s, e, l) for s, e, l in annotations.get("entities", [])}

            for label in ENTITY_LABELS:
                pred_for = {(s, e) for s, e, l in pred_set if l == label}
                gold_for = {(s, e) for s, e, l in gold_set if l == label}

                # Fuzzy matching (allow ±2 char offset)
                matched_pred = set()
                matched_gold = set()
                for ps, pe in pred_for:
                    for gs, ge in gold_for:
                        if abs(ps - gs) <= 2 and abs(pe - ge) <= 2:
                            matched_pred.add((ps, pe))
                            matched_gold.add((gs, ge))

                tp[label] += len(matched_pred)
                fp[label] += len(pred_for) - len(matched_pred)
                fn[label] += len(gold_for) - len(matched_gold)

        results = {}
        for label in ENTITY_LABELS:
            p = tp[label] / max(tp[label] + fp[label], 1)
            r = tp[label] / max(tp[label] + fn[label], 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            results[label] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3), "support": tp[label] + fn[label]}

        total_tp = sum(tp.values())
        total_fp = sum(fp.values())
        total_fn = sum(fn.values())
        micro_p = total_tp / max(total_tp + total_fp, 1)
        micro_r = total_tp / max(total_tp + total_fn, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
        results["MICRO_AVG"] = {"precision": round(micro_p, 3), "recall": round(micro_r, 3), "f1": round(micro_f1, 3), "support": total_tp + total_fn}

        return results

    # ──────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────

    def _save(self, path: str):
        os.makedirs(path, exist_ok=True)
        data = {
            "weights": {k: dict(v) for k, v in self.weights.items()},
            "biases": self.biases,
            "labels": ENTITY_LABELS,
        }
        with open(os.path.join(path, "model.json"), "w") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: str):
        model_file = os.path.join(path, "model.json")
        if os.path.exists(model_file):
            with open(model_file, "r") as f:
                data = json.load(f)
            self.weights = data.get("weights", {})
            self.biases = data.get("biases", {})
            print(f"  ✅ Loaded trained NER model from {path}/")
        else:
            print(f"  ⚠️  No model.json found in {path}, using untrained model")
