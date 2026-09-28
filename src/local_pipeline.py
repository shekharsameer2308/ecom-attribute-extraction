"""
Local Pipeline — Runs entirely on your machine, no API calls.

Combines:
  1. EasyOCR (pre-trained) → extracts ALL text from the image
  2. Your trained spaCy NER model → classifies which text is weight/volume/etc.
  3. Regex post-processor → parses the numeric value and unit
  4. Existing Validator (models.py) → deterministic math and review routing
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple

from src.models import ExtractionResult, ComputedAttribute, LLMOutputSchema
from src.preprocessor import preprocess_image, analyze_image_quality
from PIL import Image


class LocalExtractionPipeline:
    """
    Fully local pipeline — no API keys, no internet required.
    Uses OCR + your trained NER model instead of Gemini.
    """

    def __init__(
        self,
        ner_model_path: str = "models/ner_model",
        ocr_languages: List[str] = None,
        use_gpu: bool = False,
    ):
        # Load OCR engine
        try:
            from src.ocr_engine import OCREngine
            self.ocr = OCREngine(languages=ocr_languages or ["en"], gpu=use_gpu)
            self._has_ocr = True
            print("  ✅ EasyOCR loaded")
        except ImportError:
            print("  ⚠️  EasyOCR not installed. Install with: pip install easyocr")
            print("     Falling back to text-only mode (pass text directly).")
            self._has_ocr = False

        # Load trained NER model
        from src.ner_model import AttributeNERModel
        self.ner = AttributeNERModel(model_path=ner_model_path)

    # ------------------------------------------------------------------
    # Main extraction
    # ------------------------------------------------------------------

    def process_product(
        self,
        product_id: str,
        image_paths: List[str] = None,
        raw_text: str = None,
        target_attributes: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract product attributes locally.

        You can provide:
          - image_paths: list of image files → OCR extracts text → NER classifies
          - raw_text: pre-extracted text → NER classifies directly
        """
        target_attributes = target_attributes or ["net_weight", "net_volume"]

        # Step 1: Get text from images or use provided text
        all_text = ""
        image_quality = []

        if image_paths:
            for idx, path in enumerate(image_paths):
                if not os.path.isfile(path):
                    image_quality.append({"image_index": idx, "flags": ["file_not_found"], "usable": False})
                    continue

                img = Image.open(path)
                qr = analyze_image_quality(img)
                qr["image_index"] = idx
                image_quality.append(qr)

                if qr["usable"] and self._has_ocr:
                    processed = preprocess_image(path)
                    text = self.ocr.extract_full_text(processed)
                    all_text += " " + text
        elif raw_text:
            all_text = raw_text
        else:
            return {
                "product_id": product_id,
                "status": "error",
                "error": "No images or text provided",
                "needs_review": True,
            }

        if not all_text.strip():
            return {
                "product_id": product_id,
                "status": "image_unreadable",
                "needs_review": True,
                "image_quality": image_quality,
                "computed_attributes": [],
            }

        # Step 2: Run NER on extracted text
        entities = self.ner.predict(all_text)

        # Step 3: Post-process NER entities into structured ExtractionResults
        results = []
        for attr in target_attributes:
            extraction = self._build_extraction(attr, entities, all_text)
            results.append(extraction)

        # Step 4: Deterministic computation (reuse existing models.py)
        computed_results = []
        for result in results:
            computed = ComputedAttribute.from_extraction(result)
            computed_results.append(computed)

        needs_review = any(c.needs_review for c in computed_results)

        return {
            "product_id": product_id,
            "status": "success" if any(r.found for r in results) else "not_found",
            "needs_review": needs_review,
            "ocr_text": all_text.strip(),
            "ner_entities": entities,
            "image_quality": image_quality,
            "computed_attributes": [cr.model_dump() for cr in computed_results],
        }

    # ------------------------------------------------------------------
    # Post-processing: NER entities → structured ExtractionResult
    # ------------------------------------------------------------------

    # Regex to parse a numeric value and unit from a text span
    _VALUE_UNIT_RE = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*(g|gm|gms|grams|kg|kilograms|mg|oz|lb|"
        r"ml|mL|cl|l|L|litre|litres|millilitres|fl\s*oz|"
        r"W|kW|watt|watts|pcs|pieces)\b",
        re.IGNORECASE,
    )

    _UNIT_NORMALIZE = {
        "g": "g", "gm": "g", "gms": "g", "grams": "g",
        "kg": "kg", "kilograms": "kg",
        "mg": "mg",
        "oz": "oz", "lb": "lb",
        "ml": "ml", "ml": "ml",
        "cl": "cl", "l": "l", "litre": "l", "litres": "l",
        "millilitres": "ml",
        "fl oz": "fl oz",
        "w": "W", "watt": "W", "watts": "W",
        "kw": "kW",
        "pcs": "pcs", "pieces": "pcs",
    }

    # Map target_attributes → NER entity labels
    _ATTR_TO_LABEL = {
        "net_weight": "WEIGHT",
        "net_volume": "VOLUME",
        "power": "POWER",
        "item_count": "COUNT",
    }

    def _build_extraction(
        self,
        target_attr: str,
        entities: List[Dict],
        full_text: str,
    ) -> ExtractionResult:
        """Convert NER entities into a structured ExtractionResult."""

        target_label = self._ATTR_TO_LABEL.get(target_attr)
        if not target_label:
            return ExtractionResult(attribute=target_attr, found=False, confidence=0.0)

        # Find matching entities
        matches = [e for e in entities if e["label"] == target_label]

        if not matches:
            return ExtractionResult(attribute=target_attr, found=False, confidence=0.0)

        # Use the first (usually most prominent) match
        best = matches[0]
        raw_text = best["text"]

        # Parse value and unit from the entity text
        m = self._VALUE_UNIT_RE.search(raw_text)
        if not m:
            return ExtractionResult(
                attribute=target_attr, found=True,
                raw_text=raw_text, confidence=0.4,
                review_reasons=["ocr_uncertain"],
            )

        value_str = m.group(1).replace(",", ".")
        unit_raw = m.group(2).strip().lower()
        unit_norm = self._UNIT_NORMALIZE.get(unit_raw, unit_raw)

        try:
            value = float(value_str)
        except ValueError:
            return ExtractionResult(
                attribute=target_attr, found=True,
                raw_text=raw_text, confidence=0.3,
                review_reasons=["ocr_uncertain"],
            )

        # Check for pack count
        pack_count = None
        count_entities = [e for e in entities if e["label"] == "COUNT"]
        if count_entities:
            count_match = re.search(r"(\d+)", count_entities[0]["text"])
            if count_match:
                pack_count = int(count_match.group(1))

        # Determine qualifier from QUALIFIER entities
        qualifier = "unknown"
        qual_entities = [e for e in entities if e["label"] == "QUALIFIER"]
        if qual_entities:
            q_text = qual_entities[0]["text"].lower()
            if any(w in q_text for w in ["net", "contents", "content", "qty"]):
                qualifier = "net"
            elif "gross" in q_text:
                qualifier = "gross"
            elif "drained" in q_text:
                qualifier = "drained"

        # Confidence: higher if we have a qualifier, lower if not
        confidence = 0.85 if qualifier != "unknown" else 0.70

        return ExtractionResult(
            attribute=target_attr,
            found=True,
            raw_text=raw_text,
            original_value=value,
            original_unit=unit_norm,
            unit_value=value,
            pack_count=pack_count,
            qualifier=qualifier,
            confidence=confidence,
        )
