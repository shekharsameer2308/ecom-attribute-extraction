"""
Pipeline orchestrator.
Ties together: Preprocessing → LLM Perception → Deterministic Computation → Routing
"""

import os
import json
import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from src.models import LLMOutputSchema, ComputedAttribute, ExtractionResult
from src.llm_client import MultimodalLLMClient, MockLLMClient
from src.preprocessor import preprocess_image, analyze_image_quality

from PIL import Image


class ExtractionPipeline:
    """
    End-to-end pipeline:
    1. Preprocess images (resize, rotate, quality check)
    2. Call LLM for structured perception
    3. Parse & validate against Pydantic schema
    4. Apply deterministic arithmetic & unit conversion
    5. Route to review queue or auto-accept
    6. Log everything for evaluation
    """

    def __init__(
        self,
        system_prompt_path: str,
        use_mock: bool = False,
        review_confidence_threshold: float = 0.70,
        random_audit_rate: float = 0.03,     # 3% of accepted items audited
        log_dir: Optional[str] = None,
    ):
        if use_mock:
            self.llm_client = MockLLMClient()
        else:
            self.llm_client = MultimodalLLMClient()

        with open(system_prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read()

        self.review_threshold = review_confidence_threshold
        self.audit_rate = random_audit_rate
        self.log_dir = log_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_product(
        self,
        product_id: str,
        image_paths: List[str],
        context: Dict[str, str],
        target_attributes: List[str],
    ) -> Dict[str, Any]:
        """Main entry point. Takes raw images and returns computed attributes."""

        timestamp = datetime.now(timezone.utc).isoformat()

        # ---- Step 1: Preprocess images ----
        preprocessed_paths = []
        quality_reports = []
        for idx, path in enumerate(image_paths):
            if not os.path.isfile(path):
                quality_reports.append(
                    {"image_index": idx, "flags": ["file_not_found"], "usable": False}
                )
                continue

            img = Image.open(path)
            qr = analyze_image_quality(img)
            qr["image_index"] = idx
            quality_reports.append(qr)

            # Only preprocess usable images
            if qr["usable"]:
                processed = preprocess_image(path)
                preprocessed_paths.append(processed)

        if not preprocessed_paths and image_paths:
            # All images were unusable
            return {
                "product_id": product_id,
                "status": "image_unreadable",
                "needs_review": True,
                "image_quality": quality_reports,
                "computed_attributes": [],
                "timestamp": timestamp,
            }

        # ---- Step 2: Call LLM ----
        request_json = {
            "product_id": product_id,
            "context": context,
            "target_attributes": target_attributes,
        }

        raw_llm_response = self.llm_client.extract_attributes(
            system_prompt=self.system_prompt,
            image_paths=preprocessed_paths if preprocessed_paths else [],
            request_json=request_json,
        )

        # ---- Step 3: Validate schema ----
        llm_output = LLMOutputSchema(**raw_llm_response)

        # ---- Step 4: Deterministic computation ----
        computed_results = []
        for result in llm_output.results:
            computed = ComputedAttribute.from_extraction(result)
            computed_results.append(computed)

        # ---- Step 5: Review routing ----
        any_needs_review = any(c.needs_review for c in computed_results)
        bad_status = llm_output.status not in ("success",)

        # Random audit sampling on otherwise-accepted items
        random_audit = (
            not any_needs_review
            and not bad_status
            and random.random() < self.audit_rate
        )

        product_needs_review = any_needs_review or bad_status or random_audit

        final_response = {
            "product_id": product_id,
            "status": llm_output.status,
            "needs_review": product_needs_review,
            "random_audit": random_audit,
            "image_quality": quality_reports if quality_reports else [
                iq.model_dump() for iq in llm_output.image_quality
            ],
            "computed_attributes": [cr.model_dump() for cr in computed_results],
            "timestamp": timestamp,
        }

        # ---- Step 6: Log for evaluation ----
        if self.log_dir:
            self._log_result(product_id, raw_llm_response, final_response, timestamp)

        return final_response

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def process_batch(
        self, products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a list of products. Each dict must have:
          product_id, image_paths, context, target_attributes
        Returns a list of results in the same order.
        """
        results = []
        total = len(products)
        for idx, product in enumerate(products, 1):
            print(f"  [{idx}/{total}] Processing {product['product_id']} ...")
            try:
                result = self.process_product(
                    product_id=product["product_id"],
                    image_paths=product["image_paths"],
                    context=product.get("context", {}),
                    target_attributes=product.get(
                        "target_attributes", ["net_weight"]
                    ),
                )
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "product_id": product["product_id"],
                        "status": "error",
                        "error": str(e),
                        "needs_review": True,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_result(
        self,
        product_id: str,
        raw_llm_response: Dict,
        final_response: Dict,
        timestamp: str,
    ):
        os.makedirs(self.log_dir, exist_ok=True)
        log_entry = {
            "product_id": product_id,
            "timestamp": timestamp,
            "raw_llm_response": raw_llm_response,
            "final_response": final_response,
        }
        log_path = os.path.join(self.log_dir, f"{product_id}_{timestamp}.json")
        # sanitize filename
        log_path = log_path.replace(":", "-")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, indent=2, default=str)
