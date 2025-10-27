# hazard_schema.py  — Pydantic v2 compatible
from __future__ import annotations
from typing import List

from pydantic import BaseModel, Field, ValidationError, confloat, conlist

PROMPT_VERSION = "1.5"  # keep in sync with prompt.md

ALLOWED_TYPES = {
    "trafficcone","person","vehicle","bicycle","motorcycle","stroller","barrier",
    "fence","gatearm","construction","debris","pole","signpost","bollard","step",
    "curb","openhole","puddle","crack","uneven","ramp","trolley","door","furniture",
    "planter","vegetation","dog","leash","cart","ladder","pallet","scaffold","wire",
    "rope","rail","bench","trashcan","mailbox","hydrant","scooter","wheelchair",
    "crate","box","bag","suitcase"
}

CANON = {
    "cone": "trafficcone", "car": "vehicle", "truck": "vehicle", "van": "vehicle",
    "bike": "bicycle", "sign": "signpost", "bollards": "bollard"
}

class HazardOutput(BaseModel):
    hazard_detected: bool
    num_hazards: int = Field(ge=0)
    # v2: use min_length instead of min_items
    hazard_types: conlist(str, min_length=0)
    one_sentence: str = Field(min_length=1, max_length=140)
    evasive_suggestion: str = Field(min_length=1, max_length=160)
    bearing: str
    proximity: str
    confidence: confloat(ge=0.0, le=1.0)
    notes: str

    def normalized(self) -> "HazardOutput":
        d = self.model_dump()

        # guardrails
        if not d["hazard_detected"]:
            d["num_hazards"] = 0
            d["hazard_types"] = []
        if d["hazard_detected"] and d["num_hazards"] == 0:
            d["num_hazards"] = 1
        if d["hazard_detected"] and not d["hazard_types"]:
            d["hazard_types"] = ["debris"]

        # canonicalize + lowercase
        d["bearing"] = str(d["bearing"]).lower().strip()
        d["proximity"] = str(d["proximity"]).lower().strip()

        norm_types: List[str] = []
        for t in d["hazard_types"]:
            t = CANON.get(t.lower(), t.lower())
            norm_types.append(t)

        # dedupe + filter to allowed set
        d["hazard_types"] = [t for t in dict.fromkeys(norm_types) if t in ALLOWED_TYPES]

        # clamp confidence defensively
        c = float(d["confidence"])
        d["confidence"] = max(0.0, min(1.0, c))

        return HazardOutput(**d)

class OutputEnvelope(BaseModel):
    _meta: dict
    result: HazardOutput
