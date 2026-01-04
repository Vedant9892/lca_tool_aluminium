# Path: src/inference/stage_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import pandas as pd

PIPE_STAGES = [
    "Casting primary aluminium",
    "Extrusion",
    "Quench",
    "Hot shear",
    "Finish saw: cut to final lengths for downstream processing",
    "Aging/heat treat",
]

SHEET_STAGES = [
    "Casting primary aluminium",
    "Hot rolling",
    "Cold rolling",
    "Heat treatment and tempers",
    "Finishing and surface treatment",
]