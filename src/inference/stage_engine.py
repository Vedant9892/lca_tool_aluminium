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

def compute_pipe_mass_per_unit(outer_radius_m: float, inner_radius_m: float | None, length_m: float) -> float:
    ri = inner_radius_m or 0.0
    vol_m3 = math.pi * (outer_radius_m**2 - ri**2) * length_m
    return vol_m3 * 2700.0  # kg

def compute_sheet_mass_per_unit(thickness_m: float, width_m: float, length_m: float) -> float:
    vol_m3 = thickness_m * width_m * length_m
    return vol_m3 * 2700.0  # kg

def _normalize(splits: list[float]) -> list[float]:
    s = sum(splits)
    return [x/s if s else 0 for x in splits]

def _split_table(stages: list[str], splits: dict[str, list[float]], totals: dict[str, float], quality_score: float) -> pd.DataFrame:
    keys = ["electricity_kwh","carbon_kgco2e","naturalGas_nm3","wastewater_l","manufacturing_cost_per_unit","transport_usd"]
    norm = {k: _normalize(splits[k]) for k in keys}
    rows = []
    for i, st in enumerate(stages):
        rows.append({
            "stage": st,
            "quality_score": quality_score,
            "electricity_kwh": totals["electricity_kwh"]*norm["electricity_kwh"][i],
            "carbon_kgco2e":   totals["carbon_kgco2e"]  *norm["carbon_kgco2e"][i],
            "naturalGas_nm3": totals["naturalGas_nm3"]*norm["naturalGas_nm3"][i],
            "wastewater_l":    totals["wastewater_l"]   *norm["wastewater_l"][i],
            "manufacturing_cost_per_unit_usd": totals["manufacturing_cost_per_unit"]*norm["manufacturing_cost_per_unit"][i],
            "transport_cost_usd": totals["transport_usd"]*norm["transport_usd"][i],
        })          
    return pd.DataFrame(rows)

def _adjust_factor(route_type: str, grade: str, energy_source: str) -> float:
    grade_factor = {"high": 0.95, "medium": 1.00, "low": 1.10, "na": 1.00}[grade or "na"]
    route_factor = 1.00 if route_type == "conventional" else 0.65
    energy_factor = 0.85 if energy_source == "renewable" else 1.00
    return grade_factor * route_factor * energy_factor


def _apply_eol(totals: dict[str,float], eol: str) -> dict[str,float]:
    credit = EOL_CREDIT_FACTOR.get(eol or "landfill", 0.0)
    out = totals.copy()
    out["carbon_kgco2e"] *= (1.0 + credit)
    out["manufacturing_cost_per_unit"]      *= (1.0 + 0.5*credit)
    return out
