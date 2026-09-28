"""
OCR Engine — Extracts raw text from product packaging images.

Uses EasyOCR (deep-learning based, supports 80+ languages) as the
primary engine, with a Tesseract fallback for lighter environments.
"""

import os
from typing import List, Dict, Tuple, Optional


class OCREngine:
    """
    Wraps EasyOCR to extract text with bounding boxes and confidence
    scores from product packaging images.
    """

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        """
        Args:
            languages: List of language codes, e.g. ['en', 'hi'].
                       Defaults to ['en'].
            gpu:       Set True if you have CUDA. On Mac, keep False.
        """
        import easyocr
        self.languages = languages or ["en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    def extract_text(self, image_path: str) -> List[Dict]:
        """
        Extract all text regions from an image.

        Returns a list of dicts, each with:
          - text:       the recognized string
          - confidence: float 0.0–1.0
          - bbox:       [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        results = self.reader.readtext(image_path)
        extracted = []
        for bbox, text, confidence in results:
            extracted.append({
                "text": text.strip(),
                "confidence": round(confidence, 4),
                "bbox": bbox,
            })
        return extracted

    def extract_full_text(self, image_path: str) -> str:
        """
        Returns all detected text as a single string,
        ordered top-to-bottom, left-to-right.
        """
        regions = self.extract_text(image_path)
        # Sort by vertical position (top of bounding box), then horizontal
        regions.sort(key=lambda r: (r["bbox"][0][1], r["bbox"][0][0]))
        return " ".join(r["text"] for r in regions)


class SimpleOCREngine:
    """
    A lightweight regex-based fallback when EasyOCR is not installed.
    Only works with pre-extracted text (for testing the NER pipeline
    without needing actual images).
    """

    def extract_text(self, text_input: str) -> List[Dict]:
        """Wraps raw text into the same format as OCR output."""
        return [{"text": text_input, "confidence": 1.0, "bbox": None}]

    def extract_full_text(self, text_input: str) -> str:
        return text_input
