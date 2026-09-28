import json
import os
from typing import List, Dict, Any
from src.models import LLMOutputSchema


class MultimodalLLMClient:
    """Client wrapper for Google Gemini API for structured multimodal extraction."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not set. "
                "Run: export GEMINI_API_KEY='your-key-here'"
            )

        from google import genai
        self.genai = genai
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def extract_attributes(
        self,
        system_prompt: str,
        image_paths: List[str],
        request_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calls the LLM with images, the system prompt, and request context.
        Forces the output to match LLMOutputSchema via structured output.
        """
        from google.genai import types
        from PIL import Image

        # Build the content parts
        contents = []

        # Add images first (Gemini performs best with images before text)
        for idx, path in enumerate(image_paths):
            img = Image.open(path)
            contents.append(img)

        # Add the request context as text
        contents.append(
            types.Part.from_text(
                f"Request Context:\n{json.dumps(request_json, indent=2)}"
            )
        )

        # Enforce structured output matching our Pydantic schema
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=LLMOutputSchema.model_json_schema(),
            temperature=0.1,  # Low temp for deterministic extraction
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        return json.loads(response.text)


class MockLLMClient:
    """
    A mock client for testing the pipeline without API costs.
    Returns realistic structured outputs for various test scenarios.
    """

    # Pre-built mock scenarios keyed by product_id prefix
    SCENARIOS = {
        "SINGLE": {
            "status": "success",
            "image_quality": [{"image_index": 0, "flags": [], "usable": True}],
            "results": [
                {
                    "attribute": "net_weight",
                    "found": True,
                    "raw_text": "Net Wt. 500 g e",
                    "language_detected": "en",
                    "pack_count": None,
                    "original_value": 500,
                    "original_unit": "g",
                    "unit_value": 500,
                    "qualifier": "estimated",
                    "is_approximate": False,
                    "is_minimum": False,
                    "confidence": 0.95,
                    "evidence_quality": "high",
                    "review_reasons": [],
                    "notes": "",
                }
            ],
        },
        "MULTI": {
            "status": "success",
            "image_quality": [{"image_index": 0, "flags": [], "usable": True}],
            "results": [
                {
                    "attribute": "net_weight",
                    "found": True,
                    "raw_text": "6 x 500 g",
                    "language_detected": "en",
                    "pack_count": 6,
                    "original_value": 500,
                    "original_unit": "g",
                    "unit_value": 500,
                    "qualifier": "net",
                    "confidence": 0.95,
                    "evidence_quality": "high",
                    "review_reasons": [],
                }
            ],
        },
        "DUAL": {
            "status": "success",
            "image_quality": [{"image_index": 0, "flags": [], "usable": True}],
            "results": [
                {
                    "attribute": "net_volume",
                    "found": True,
                    "raw_text": "NET 12 FL OZ (355 ml)",
                    "language_detected": "en",
                    "pack_count": None,
                    "original_value": 355,
                    "original_unit": "ml",
                    "unit_value": 355,
                    "qualifier": "net",
                    "confidence": 0.92,
                    "evidence_quality": "high",
                    "secondary_measure": {
                        "raw_text": "12 FL OZ",
                        "original_value": 12,
                        "original_unit": "fl oz",
                    },
                    "review_reasons": [],
                }
            ],
        },
        "BLURRY": {
            "status": "partial",
            "image_quality": [
                {"image_index": 0, "flags": ["blurry", "low_resolution"], "usable": False}
            ],
            "results": [
                {
                    "attribute": "net_weight",
                    "found": True,
                    "raw_text": "Net Wt. 5?0 g",
                    "language_detected": "en",
                    "pack_count": None,
                    "original_value": 500,
                    "original_unit": "g",
                    "unit_value": 500,
                    "qualifier": "net",
                    "confidence": 0.45,
                    "evidence_quality": "low",
                    "review_reasons": ["low_confidence", "ocr_uncertain"],
                }
            ],
        },
        "NOTFOUND": {
            "status": "not_found",
            "image_quality": [{"image_index": 0, "flags": [], "usable": True}],
            "results": [
                {
                    "attribute": "net_weight",
                    "found": False,
                    "raw_text": None,
                    "language_detected": None,
                    "pack_count": None,
                    "original_value": None,
                    "original_unit": None,
                    "unit_value": None,
                    "qualifier": None,
                    "confidence": 0.0,
                    "evidence_quality": None,
                    "review_reasons": [],
                }
            ],
        },
    }

    def extract_attributes(
        self,
        system_prompt: str,
        image_paths: List[str],
        request_json: Dict[str, Any],
    ) -> Dict[str, Any]:
        product_id = request_json.get("product_id", "")
        print(f"  [MockLLM] Processing {len(image_paths)} image(s) for: {product_id}")

        # Pick the matching scenario, default to SINGLE
        scenario_key = "SINGLE"
        for key in self.SCENARIOS:
            if product_id.upper().startswith(key):
                scenario_key = key
                break

        scenario = self.SCENARIOS[scenario_key]
        return {
            "request_id": f"mock-{product_id}",
            "product_id": product_id,
            **scenario,
        }
