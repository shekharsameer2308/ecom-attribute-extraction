# E-Commerce Attribute Extraction

An end-to-end pipeline for extracting structured product attributes (weight, volume, power) from packaging images. Runs in two modes: cloud (Google Gemini API) or fully local (trainable NER model, no GPU needed).

## The Problem

Millions of e-commerce products lack structured metadata. Sellers embed specs on packaging images, but manual extraction doesn't scale and basic OCR can't tell `500g net weight` apart from `20% Extra Free`.

This pipeline reads packaging images and outputs clean, validated, structured data ready for search filters, logistics systems, and catalog databases.

## Architecture

```
Image → Preprocessor → Perception (Gemini API or local NER) → Validator → Structured JSON
         resize/exif     reads text only, no math              unit conversion
         quality check   copies exactly as printed             arithmetic checks
                                                               review routing
```

The LLM (or NER model) is responsible only for **reading**. All math — unit conversion, multipack totals, consistency checks — is deterministic Python. This eliminates arithmetic hallucinations entirely.

## Project Structure

```
├── main.py                      CLI: extract / batch / test
├── evaluate.py                  Capability benchmark (30 tests, 9 categories)
├── requirements.txt
│
├── src/
│   ├── models.py                Pydantic schemas + deterministic math
│   ├── pipeline.py              Cloud pipeline (Gemini API)
│   ├── local_pipeline.py        Local pipeline (OCR + NER)
│   ├── llm_client.py            Gemini client + mock client
│   ├── ner_model.py             Trainable NER (Averaged Perceptron)
│   ├── ocr_engine.py            EasyOCR wrapper
│   └── preprocessor.py          Image quality, resize, EXIF, tiling
│
├── training/
│   ├── train.py                 Train the local NER model
│   ├── generate_data.py         Synthetic data generator
│   └── data/                    Train/test datasets
│
├── models/
│   └── ner_model/model.json     Trained weights (portable JSON)
│
├── prompts/
│   └── system_prompt.txt        LLM system prompt
│
└── data/
    └── sample_batch.json        Example batch manifest
```

## Quick Start

```bash
git clone https://github.com/shekharsameer2308/ecom-attribute-extraction.git
cd ecom-attribute-extraction
pip install -r requirements.txt
```

Run the test suite (no API key needed):

```bash
python main.py test        # 32 checks across 14 scenarios
python evaluate.py         # full capability report: 93.3% (28/30)
```

## Cloud Mode (Gemini API)

Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey), then:

```bash
export GEMINI_API_KEY="your-key"

# single image
python main.py extract --image product.jpg

# multiple views
python main.py extract --image front.jpg --image back.jpg

# with context
python main.py extract \
  --image product.jpg \
  --product-id SKU-12345 \
  --category "Grocery > Snacks" \
  --brand "Lays" \
  --attributes "net_weight,net_volume"

# batch
python main.py batch --manifest data/sample_batch.json --output results.json
```

## Local Mode (Trained NER)

Runs offline. No API key. No GPU. The pipeline works as:

```
Image → EasyOCR (extracts all text) → NER model (classifies spans) → Validator (converts units)
```

```python
from src.ner_model import AttributeNERModel

model = AttributeNERModel(model_path="models/ner_model")
model.predict("Net Wt. 500 g 20% Extra Free")
# [{'text': 'Net Wt.', 'label': 'QUALIFIER', 'score': 0.88},
#  {'text': '500 g',   'label': 'WEIGHT',    'score': 1.00}]
# ignores the promo text
```

## Training

The local NER model uses an Averaged Perceptron. It learns from labeled text examples — for each span, it extracts features (unit patterns, digit ratios, context words), predicts a label, and adjusts weights when wrong. Trains in seconds on CPU.

```bash
# default: 500 samples, 30 epochs
python training/train.py

# more data
python training/train.py --samples 2000 --epochs 50
```

The improvement loop: train → run `evaluate.py` → add corrections to `training/data/train.json` → retrain.

## Evaluation

```bash
python evaluate.py
```

Current scores:

| Category | Score | Difficulty |
|---|---|---|
| Basic Weight | 5/5 | Easy |
| Basic Volume | 3/3 | Easy |
| Qualifier Detection | 3/3 | Easy |
| Noise Filtering | 4/4 | Medium |
| Multipacks | 3/3 | Medium |
| Dual Units | 2/2 | Hard |
| Power | 0/2 | Medium |
| Unit Conversion | 5/5 | Easy |
| Full Pipeline (E2E) | 3/3 | Hard |
| **Overall** | **28/30 (93.3%)** | |

Power extraction is weak due to limited training samples. Retraining with `--samples 2000` fixes this.

## Output Format

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

## Unit Conversion

| From | To | Factor |
|---|---|---|
| kg | g | ×1000 |
| mg | g | ×0.001 |
| L | ml | ×1000 |
| fl oz | ml | ×29.5735 |
| oz | g | ×28.3495 |

## CLI Reference

| Command | What it does | Needs API key |
|---|---|---|
| `python main.py test` | Run test suite | No |
| `python main.py extract --image img.jpg` | Extract from image | Yes |
| `python main.py extract --image img.jpg --mock` | Extract with mock data | No |
| `python main.py batch --manifest file.json` | Batch process | Yes |
| `python evaluate.py` | Capability report | No |
| `python training/train.py` | Train local model | No |

## Design Decisions

- **LLM reads, code computes.** LLMs hallucinate math. Deterministic code doesn't.
- **Pydantic schema enforcement.** Gemini's structured output mode guarantees valid JSON.
- **Averaged Perceptron for local NER.** Trains in seconds, no dependencies, portable JSON weights.
- **Synthetic data bootstrap.** No labeled images needed to start. Fine-tune on real data from the review queue.
- **3% random audit.** Auto-accepted items are sampled for human review to catch systematic drift.
- **Qualifier-first extraction.** The legally required net quantity statement is the primary target, per GS1/GDSN standards.

## License

MIT
