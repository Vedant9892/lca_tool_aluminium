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
DEFAULT_QUALITY_BY_GRADE = {"high": 0.92, "medium": 0.85, "low": 0.78, "na": 0.80}
EOL_CREDIT_FACTOR = {"recycle": -0.15, "reuse": -0.10, "landfill": 0.0}

@dataclass
class DashboardInput:
    product: str                    # "pipe" | "sheet"
    units: int
    route_type: str                 # "conventional" | "recycle"
    bauxite_grade: str              # "high" | "medium" | "low" | "na"
    energy_source: str              # "renewable" | "non_renewable"
    eol_option: str                 # "recycle" | "reuse" | "landfill"
    # pipe dims (m)
    outer_radius_m: float | None = None
    inner_radius_m: float | None = None
    length_m: float | None = None
    # sheet dims (m)
    thickness_m: float | None = None
    width_m: float | None = None
    sheet_length_m: float | None = None
