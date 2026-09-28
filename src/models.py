from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, model_validator

# --- Enums ---
class Qualifier(str, Enum):
    net = "net"
    gross = "gross"
    drained = "drained"
    estimated = "estimated"
    per_serving = "per_serving"
    per_100g = "per_100g"
    unknown = "unknown"

class EvidenceQuality(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"

# --- Base LLM Perception Models ---

class BoundingBox(BaseModel):
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

class ImageSource(BaseModel):
    image_index: int
    bbox_norm: Optional[BoundingBox] = None

class SecondaryMeasure(BaseModel):
    raw_text: str
    original_value: float
    original_unit: str

class ExtractionResult(BaseModel):
    attribute: str
    found: bool
    raw_text: Optional[str] = None
    language_detected: Optional[str] = None
    pack_count: Optional[int] = None
    original_value: Optional[float] = None
    original_unit: Optional[str] = None
    unit_value: Optional[float] = None
    qualifier: Optional[Qualifier] = None
    is_approximate: bool = False
    is_minimum: bool = False
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    secondary_measure: Optional[SecondaryMeasure] = None
    variants: List[Any] = Field(default_factory=list)
    source: Optional[ImageSource] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_quality: Optional[EvidenceQuality] = None
    review_reasons: List[str] = Field(default_factory=list)
    candidates: List[Any] = Field(default_factory=list)
    notes: str = ""

class ImageQuality(BaseModel):
    image_index: int
    flags: List[str] = Field(default_factory=list)
    usable: bool = True

class LLMOutputSchema(BaseModel):
    """The exact schema the LLM is instructed to output."""
    request_id: str
    product_id: str
    status: str
    error: Optional[str] = None
    image_quality: List[ImageQuality] = Field(default_factory=list)
    results: List[ExtractionResult] = Field(default_factory=list)


# --- Deterministic Computation & Validation Models ---

# Standard conversion factors to base units (e.g., base weight = grams)
UNIT_CONVERSIONS = {
    "kg": {"base": "g", "multiplier": 1000.0},
    "g": {"base": "g", "multiplier": 1.0},
    "mg": {"base": "g", "multiplier": 0.001},
    "l": {"base": "ml", "multiplier": 1000.0},
    "ml": {"base": "ml", "multiplier": 1.0},
    "oz": {"base": "g", "multiplier": 28.3495},
    "fl oz": {"base": "ml", "multiplier": 29.5735},
}

class ComputedAttribute(BaseModel):
    """The final, application-ready attribute after rules and math are applied."""
    attribute_name: str
    computed_value: Optional[float] = None
    canonical_unit: Optional[str] = None
    arithmetic_consistent: bool = True
    needs_review: bool = False
    review_reasons: List[str] = Field(default_factory=list)
    raw_extraction: ExtractionResult

    @classmethod
    def from_extraction(cls, ext: ExtractionResult) -> "ComputedAttribute":
        needs_review = len(ext.review_reasons) > 0 or ext.confidence < 0.70
        reasons = list(ext.review_reasons)
        
        computed_value = None
        canonical_unit = None
        arithmetic_consistent = True

        if ext.found and ext.original_value is not None and ext.original_unit is not None:
            unit_norm = ext.original_unit.strip().lower()
            
            # 1. Unit Normalization
            if unit_norm in UNIT_CONVERSIONS:
                conv = UNIT_CONVERSIONS[unit_norm]
                canonical_unit = conv["base"]
                
                # 2. Arithmetic computation
                # Total = pack_count * unit_value * multiplier
                p_count = ext.pack_count if ext.pack_count is not None else 1
                u_val = ext.unit_value if ext.unit_value is not None else ext.original_value
                
                computed_value = p_count * u_val * conv["multiplier"]
                
                # 3. Consistency check: did the LLM report a total that contradicts the math?
                if p_count > 1:
                    # If label says "6 x 500g", original_value is often 500.
                    # If label says "3000g (6x500g)", original_value is 3000.
                    # Both are valid. We only flag if original_value matches NEITHER.
                    expected_total = p_count * u_val
                    if abs(ext.original_value - u_val) > 0.01 and abs(ext.original_value - expected_total) > 0.01:
                        arithmetic_consistent = False
                        needs_review = True
                        reasons.append("arithmetic_inconsistent")
            else:
                needs_review = True
                reasons.append(f"unknown_unit_{unit_norm}")

        return cls(
            attribute_name=ext.attribute,
            computed_value=computed_value,
            canonical_unit=canonical_unit,
            arithmetic_consistent=arithmetic_consistent,
            needs_review=needs_review,
            review_reasons=reasons,
            raw_extraction=ext
        )
