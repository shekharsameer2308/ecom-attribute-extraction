# Multimodal Attribute Extraction for E-Commerce

An end-to-end pipeline that extracts structured product attributes (weight, volume, dimensions) from packaging images using Google Gemini's multimodal AI.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Raw Image  │────▶│ Preprocessor │────▶│  Gemini LLM │────▶│  Validator   │
│  (jpg/png)  │     │  resize/exif │     │  perception  │     │  math/units  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                     │
                                                          ┌──────────▼──────────┐
                                                          │   Review Router     │
                                                          │ auto-accept / queue │
                                                          └─────────────────────┘
```

**Key design principle:** The LLM only *reads* — it copies text exactly as printed. All math (unit conversion, multipack totals) is done by deterministic Python code. This eliminates LLM arithmetic hallucinations.

## Project Structure

```
ecom project/
├── main.py                  # CLI entry point (extract / batch / test)
├── requirements.txt         # Python dependencies
├── prompts/
│   └── system_prompt.txt    # LLM system prompt (versioned separately)
├── src/
│   ├── __init__.py
│   ├── models.py            # Pydantic schemas + deterministic math
│   ├── llm_client.py        # Gemini API client + mock client
│   ├── pipeline.py          # Orchestrator (preprocess → LLM → validate → route)
│   └── preprocessor.py      # Image quality checks, resize, tiling
├── data/
│   └── sample_batch.json    # Sample batch manifest for testing
└── logs/                    # Auto-generated extraction logs
```

## Quick Start

### 1. Install Dependencies

```bash
cd "ecom project"
pip install -r requirements.txt
```

### 2. Run Tests (No API Key Needed)

```bash
python main.py test
```

This runs 14 test cases covering:
- Single pack, multipack, dual units
- Unit conversion (kg→g, L→ml, fl oz→ml)
- Arithmetic consistency checks
- Low confidence / OCR uncertainty
- Not-found attributes
- Unknown units
- Full E2E pipeline with mock LLM

### 3. Extract from a Real Image

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Single image
python main.py extract --image product_front.jpg

# Multiple images of same product
python main.py extract --image front.jpg --image back.jpg

# With product context
python main.py extract \
  --image product.jpg \
  --product-id SKU-12345 \
  --category "Grocery > Snacks" \
  --brand "Lays" \
  --attributes "net_weight,net_volume"
```

### 4. Batch Processing

Create a JSON manifest (see `data/sample_batch.json` for format):

```json
[
  {
    "product_id": "SKU-001",
    "image_paths": ["images/sku001_front.jpg", "images/sku001_back.jpg"],
    "context": {"category": "Grocery", "brand": "Nestle"},
    "target_attributes": ["net_weight", "net_volume"]
  }
]
```

Run it:

```bash
# With real API
python main.py batch --manifest data/sample_batch.json --output results.json

# With mock (for testing)
python main.py batch --manifest data/sample_batch.json --mock
```

## How It Works

### Step 1: Image Preprocessing (`src/preprocessor.py`)
- Auto-rotates using EXIF data
- Resizes to max 2048px (Gemini optimal)
- Converts RGBA → RGB
- Checks quality: blur, glare, resolution, cropping
- Tiles panoramic images for curved packaging

### Step 2: LLM Perception (`src/llm_client.py`)
- Sends images + system prompt to Gemini
- Forces structured JSON output via Pydantic schema
- Low temperature (0.1) for deterministic extraction
- LLM copies text exactly — no math, no guessing

### Step 3: Deterministic Computation (`src/models.py`)
- Converts units: kg→g, L→ml, fl oz→ml, oz→g
- Computes multipack totals: `pack_count × unit_value × multiplier`
- Validates arithmetic consistency
- Flags unknown units for review

### Step 4: Review Routing (`src/pipeline.py`)
- Auto-flags items with confidence < 0.70
- Flags arithmetic inconsistencies
- Flags unknown units
- Random 3% audit of auto-accepted items
- Logs raw LLM output for evaluation

## Getting Your API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with Google
3. Click **Create API key**
4. Copy and export it:
   ```bash
   export GEMINI_API_KEY="your-key"
   ```

## Example Output

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
      "review_reasons": [],
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
