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

PIPE_SPLITS = {
    "electricity_kwh": [0.18, 0.35, 0.08, 0.06, 0.08, 0.25],
    "carbon_kgco2e":   [0.20, 0.32, 0.07, 0.06, 0.08, 0.27],
    "naturalGas_nm3": [0.10, 0.20, 0.05, 0.05, 0.10, 0.50],
    "wastewater_l":    [0.15, 0.40, 0.05, 0.05, 0.15, 0.20],
    "manufacturing_cost_per_unit":        [0.22, 0.28, 0.06, 0.06, 0.10, 0.28],
    "transport_usd":   [0.35, 0.25, 0.05, 0.05, 0.10, 0.20],
}

SHEET_SPLITS = {
    "electricity_kwh": [0.25, 0.35, 0.20, 0.10, 0.10],
    "carbon_kgco2e":   [0.23, 0.37, 0.18, 0.12, 0.10],
    "naturalGas_nm3": [0.10, 0.30, 0.20, 0.25, 0.15],
    "wastewater_l":    [0.20, 0.30, 0.20, 0.15, 0.15],
    "manufacturing_cost_per_unit":        [0.22, 0.30, 0.18, 0.15, 0.15],
    "transport_usd":   [0.40, 0.25, 0.15, 0.10, 0.10],
}
