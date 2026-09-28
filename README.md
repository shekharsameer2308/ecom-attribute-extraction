<p align="center">
  <h1 align="center">🏷️ E-Commerce Attribute Extraction</h1>
  <p align="center">
    <strong>Extract structured product attributes from packaging images using AI</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-how-it-works">How It Works</a> •
    <a href="#-local-model-training">Train Locally</a> •
    <a href="#-evaluation">Evaluation</a> •
    <a href="#-api-reference">API Reference</a>
  </p>
</p>

---

> **The Problem:** Millions of e-commerce products lack structured metadata (weight, volume, dimensions). Sellers embed this data on packaging images, but manual extraction doesn't scale, and basic OCR can't distinguish `500g net weight` from `20% Extra Free` promotional noise.
>
> **This Solution:** A dual-mode AI pipeline that reads packaging images and outputs clean, validated, structured data — ready for search filters, logistics, and catalog databases.

## ✨ Key Features

| Feature | Description |
|---|---|
| **Dual Pipelines** | ☁️ Cloud mode (Google Gemini API) or 🖥️ fully local mode (trained NER model) |
| **Perception ≠ Computation** | LLM only *reads* text. All math is deterministic Python — zero arithmetic hallucinations |
| **Noise Filtering** | Ignores promos ("20% Extra Free"), nutrition tables ("Per 100g"), and marketing text |
| **Multipack Handling** | Parses `6 x 500g`, `Pack of 12`, and validates `pack_count × unit_value = total` |
| **Unit Standardization** | Converts kg→g, L→ml, fl oz→ml, oz→g automatically |
| **Human-in-the-Loop** | Routes low-confidence items to review queues + random 3% audit sampling |
| **Trainable** | Train and improve the local NER model on your own data — no GPU required |
| **93.3% Accuracy** | Out-of-the-box on 30 benchmark tests across 9 categories |

## 🏗️ Architecture

```
                        ┌─────────── Cloud Mode ───────────┐
                        │                                   │
┌──────────┐   ┌────────────────┐   ┌───────────────┐   ┌──────────────────┐
│          │   │                │   │  Gemini API   │   │                  │
│  Product ├──▶│  Preprocessor  ├──▶│  (multimodal) │──▶│    Validator     │
│  Image   │   │  quality/resize│   │       OR      │   │  unit conversion │
│          │   │                │   │  Local NER    │   │  arithmetic check│
└──────────┘   └────────────────┘   │  (trainable)  │   │  review routing  │
                                    └───────────────┘   └────────┬─────────┘
                        │                                   │    │
                        └─────────── Local Mode ────────────┘    ▼
                                                          ┌──────────────┐
                                                          │ Structured   │
                                                          │ JSON Output  │
                                                          └──────────────┘
```

**Core design principle:** The AI model is responsible only for **perception** (reading what's on the label). All **computation** (unit conversion, multipack totals, validation) is handled by deterministic Python code. This separation eliminates an entire class of LLM hallucination errors.

## 📁 Project Structure

```
├── main.py                      # CLI: extract / batch / test
├── evaluate.py                  # Capability benchmarking (30 tests, 9 categories)
├── requirements.txt
│
├── src/
│   ├── models.py                # Pydantic schemas + deterministic math engine
│   ├── pipeline.py              # Cloud pipeline orchestrator (Gemini API)
│   ├── local_pipeline.py        # Local pipeline orchestrator (OCR + NER)
│   ├── llm_client.py            # Gemini API client + mock client
│   ├── ner_model.py             # Trainable NER model (Averaged Perceptron)
│   ├── ocr_engine.py            # EasyOCR wrapper for text extraction
│   └── preprocessor.py          # Image quality, resize, EXIF, tiling
│
├── training/
│   ├── train.py                 # Training script (runs on CPU in seconds)
│   ├── generate_data.py         # Synthetic training data generator
│   └── data/                    # Train/test JSON datasets
│
├── models/
│   └── ner_model/model.json     # Trained model weights (portable JSON)
│
├── prompts/
│   └── system_prompt.txt        # LLM system prompt (version-controlled)
│
└── data/
    └── sample_batch.json        # Example batch manifest
```

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/shekharsameer2308/ecom-attribute-extraction.git
cd ecom-attribute-extraction
pip install -r requirements.txt
```

### 2. Run Tests (no API key needed)

```bash
python main.py test
```
```
✅ 32 checks passed across 14 test scenarios
```

### 3. Evaluate Model Capability

```bash
python evaluate.py
```
```
🎯 OVERALL CAPABILITY SCORE: 93.3%
   🟢 Basic Weight          5/5  (100%)
   🟢 Noise Filtering       4/4  (100%)
   🟢 Multipacks            3/3  (100%)
   🟢 Dual Units            2/2  (100%)
   🟢 Unit Conversion       5/5  (100%)
   🟢 Full Pipeline         3/3  (100%)
```

---

## ☁️ Cloud Mode (Gemini API)

Best accuracy. Requires internet and a free API key.

### Get Your API Key

1. Go to **[Google AI Studio](https://aistudio.google.com/app/apikey)**
2. Click **Create API key** → copy it

### Extract Attributes

```bash
export GEMINI_API_KEY="your-key-here"

# Single image
python main.py extract --image product.jpg

# Multiple views of the same product
python main.py extract --image front.jpg --image back.jpg

# With full context
python main.py extract \
  --image product.jpg \
  --product-id SKU-12345 \
  --category "Grocery > Snacks" \
  --brand "Lays" \
  --attributes "net_weight,net_volume"
```

### Batch Processing

```bash
python main.py batch --manifest data/sample_batch.json --output results.json
```

<details>
<summary>📄 Batch manifest format</summary>

```json
[
  {
    "product_id": "SKU-001",
    "image_paths": ["images/front.jpg", "images/back.jpg"],
    "context": {"category": "Grocery", "brand": "Nestle"},
    "target_attributes": ["net_weight", "net_volume"]
  }
]
```
</details>

---

## 🖥️ Local Mode (Trained NER)

Runs entirely offline. No API key. No internet. Trains on CPU in seconds.

### How the Local Pipeline Works

```
Image → EasyOCR → "Net Wt. 500 g 20% Extra Free" → NER Model → [500 g → WEIGHT] → Validator → 500.0 g
                    ↑ extracts ALL text                ↑ classifies entities          ↑ converts units
                                                       (ignores promos)               (deterministic math)
```

### Use the Trained Model

```python
from src.ner_model import AttributeNERModel

model = AttributeNERModel(model_path="models/ner_model")

entities = model.predict("Net Wt. 500 g 20% Extra Free")
# [{'text': 'Net Wt.', 'label': 'QUALIFIER', 'score': 0.88},
#  {'text': '500 g',   'label': 'WEIGHT',    'score': 1.00}]
# ↑ Correctly ignores "20% Extra Free"
```

---

## 🧠 Local Model Training

### How Training Works

The NER model uses an **Averaged Perceptron** algorithm — a simple but effective method that learns from labeled examples:

```
For each training example:
  1. Extract features from text spans (unit patterns, digit ratios, context keywords)
  2. Model predicts an entity label
  3. Compare prediction to ground truth:
     • CORRECT → reinforce weights slightly
     • WRONG   → decrease wrong label's weights, increase correct label's weights
     • MISSED  → boost the missed label's weights
  4. Repeat for N epochs
  5. Average all weight updates (reduces overfitting)
```

No GPU needed. No heavy frameworks. Trains in under 10 seconds.

### Train the Model

```bash
# Default: 500 samples, 30 epochs
python training/train.py

# More data for better accuracy
python training/train.py --samples 2000 --epochs 50

# Custom output path
python training/train.py --output models/my_custom_model
```

### Training Output

```
🧠 Local NER Model Training Pipeline
─────────────────────────────────────────────
  Training samples  : 496
  Epochs            : 30
  Algorithm         : Averaged Perceptron
─────────────────────────────────────────────

  Epoch   1/30  |  Accuracy: 0.846  |  Errors: 146
  Epoch  10/30  |  Accuracy: 0.848  |  Errors: 109
  Epoch  30/30  |  Accuracy: 0.846  |  Errors: 111

  ✅ Model saved to: models/ner_model/

🔍 Demo predictions:
   'Net Wt. 500 g'  →  'Net Wt.'→QUALIFIER(0.88), '500 g'→WEIGHT(1.00)
   '6 x 200 ml'     →  '6 x'→COUNT(0.62), '200 ml'→VOLUME(1.00)
```

### Improve with Your Own Data

The **train → evaluate → improve** loop:

```bash
# 1. Train
python training/train.py --samples 2000 --epochs 50

# 2. Evaluate
python evaluate.py

# 3. Add real corrections to training/data/train.json

# 4. Retrain and re-evaluate
python training/train.py
python evaluate.py
```

---

## 📊 Evaluation

Run the full capability benchmark anytime:

```bash
python evaluate.py
```

### Test Categories

| Category | Tests | Difficulty | What It Checks |
|---|---|---|---|
| Basic Weight | 5 | Easy | `500 g`, `1.5 kg`, `12 oz` |
| Basic Volume | 3 | Easy | `355 ml`, `2 L`, `12 fl oz` |
| Qualifier Detection | 3 | Easy | `Net Wt.`, `Drained`, `Net Content` |
| Noise Filtering | 4 | Medium | Ignoring promos, nutrition tables, brand text |
| Multipacks | 3 | Medium | `6 x 500g`, `Pack of 12`, `24 pcs` |
| Dual Units | 2 | Hard | `12 FL OZ (355 ml)`, `8 oz (226 g)` |
| Power | 2 | Medium | `1200 W`, `2.5 kW` |
| Unit Conversion | 5 | Easy | kg→g, L→ml, fl oz→ml, mg→g, multipack math |
| Full Pipeline | 3 | Hard | End-to-end: text → NER → compute → answer |

### Current Scores

```
🎯 93.3%  ████████████████████████████░░  28/30 tests
```

---

## 📋 API Reference

### CLI Commands

| Command | Description | API Key? |
|---|---|---|
| `python main.py test` | Run 14 test scenarios | ❌ No |
| `python main.py extract --image img.jpg` | Extract from image | ✅ Yes |
| `python main.py extract --image img.jpg --mock` | Test with mock data | ❌ No |
| `python main.py batch --manifest file.json` | Batch processing | ✅ Yes |
| `python evaluate.py` | Full capability report | ❌ No |
| `python training/train.py` | Train local NER model | ❌ No |

### Output Schema

```json
{
  "product_id": "SKU-12345",
  "status": "success",
  "needs_review": false,
  "computed_attributes": [
    {
      "attribute_name": "net_weight",
      "computed_value": 3000.0,
      "canonical_unit": "g",
      "arithmetic_consistent": true,
      "needs_review": false,
      "raw_extraction": {
        "raw_text": "6 x 500 g",
        "pack_count": 6,
        "original_value": 500.0,
        "original_unit": "g",
        "qualifier": "net",
        "confidence": 0.95
      }
    }
  ]
}
```

### Supported Attributes

| Attribute | Entity Label | Example Inputs |
|---|---|---|
| `net_weight` | WEIGHT | `500 g`, `1.5 kg`, `12 oz` |
| `net_volume` | VOLUME | `330 ml`, `2 L`, `12 fl oz` |
| `power` | POWER | `1200 W`, `2.5 kW` |
| `item_count` | COUNT | `6 pcs`, `Pack of 12` |

### Unit Conversion Table

| Input Unit | Canonical Unit | Multiplier |
|---|---|---|
| `kg` | `g` | ×1000 |
| `mg` | `g` | ×0.001 |
| `L`, `l` | `ml` | ×1000 |
| `fl oz` | `ml` | ×29.5735 |
| `oz` | `g` | ×28.3495 |

---

## 🔒 Design Decisions

| Decision | Rationale |
|---|---|
| **LLM reads, code computes** | LLMs hallucinate arithmetic. Deterministic code doesn't. |
| **Pydantic schema enforcement** | Gemini's structured output mode guarantees valid JSON. No parsing failures. |
| **Averaged Perceptron for local NER** | Trains in seconds on CPU, no framework dependencies, portable JSON weights. |
| **Synthetic training data** | Bootstrap without labeled images. Fine-tune on real data from review queue. |
| **Random audit sampling** | 3% of auto-accepted items go to human review to catch systematic drift. |
| **Qualifier-first extraction** | The legally required "Net Quantity" statement is the primary target per GS1/GDSN standards. |

---

## 🗺️ Roadmap

- [ ] Web UI for upload + extraction visualization
- [ ] Bounding box overlay on source images
- [ ] Multi-language OCR support (Hindi, Arabic, Chinese)
- [ ] Fine-tuning pipeline for Gemini with collected review data
- [ ] REST API server for integration with catalog systems
- [ ] Confidence calibration using Platt scaling on labeled data

---

## 📄 License

MIT

---

<p align="center">
  Built with ❤️ for solving real e-commerce data quality problems
</p>
