# Path: src/inference/baselines.py
from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_baselines(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def median_baseline(df: pd.DataFrame, product: str, route_type: str, energy_source: str, grade: str) -> dict:
    def filt(dfa, **kw):
        out = dfa.copy()
        for k,v in kw.items():
            if k in out.columns:
                out = out[out[k]==v]
        return out

    candidates = [
        {"product_type": product, "route_type": route_type, "energy_source": energy_source, "grade": grade},
        {"product_type": product, "route_type": route_type, "energy_source": energy_source},
        {"product_type": product, "route_type": route_type},
        {"product_type": product},
        {}
    ]
    for keys in candidates:
        sub = filt(df, **keys)
        if len(sub) > 0:
            out = {}
            if "electricity_kwh" in sub.columns:
                out["electricity_kwh"] = float(sub["electricity_kwh"].median())
            elif "totalelectricitykwh" in sub.columns:
                out["electricity_kwh"] = float(sub["totalelectricitykwh"].median())
            else:
                out["electricity_kwh"] = 12.0

            if "carbon_kgco2e" in sub.columns:
                out["carbon_kgco2e"] = float(sub["carbon_kgco2e"].median())
            elif "totalcarbonkgco2e" in sub.columns:
                out["carbon_kgco2e"] = float(sub["totalcarbonkgco2e"].median())
            else:
                out["carbon_kgco2e"] = 8.0

            out["naturalGas_nm3"] = float(sub["totalnaturalgasnm3"].median()) if "totalnaturalgasnm3" in sub.columns else 0.05
            out["wastewater_l"] = float(sub["totalwastewaterl"].median()) if "totalwastewaterl" in sub.columns else 1.5
            out["manufacturing_cost_per_unit"] = float(sub["manufacturingcostperunit"].median()) if "manufacturingcostperunit" in sub.columns else 0.6
            out["transport_cost_usd"] = float(sub["transportcost"].median()) if "transportcost" in sub.columns else 0.1
            return out
    return {"electricity_kwh":12.0,"carbon_kgco2e":8.0,"naturalGas_nm3":0.05,"wastewater_l":1.5,"manufacturing_cost_per_unit":0.6,"transport_cost_usd":0.1}
