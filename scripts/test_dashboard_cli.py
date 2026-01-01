# Path: scripts/test_dashboard_cli.py
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd

from src.inference.stage_engine import DashboardInput, build_stage_breakdown
from src.inference.baselines import load_baselines, median_baseline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, choices=["pipe","sheet"])
    ap.add_argument("--units", type=int, default=1)
    ap.add_argument("--route", required=True, choices=["conventional","recycle"])
    ap.add_argument("--grade", default="na", choices=["high","medium","low","na"])
    ap.add_argument("--energy", required=True, choices=["renewable","non_renewable"])
    ap.add_argument("--eol", default="recycle", choices=["recycle","reuse","landfill"])
    # pipe dims (m)
    ap.add_argument("--outer_radius_m", type=float)
    ap.add_argument("--inner_radius_m", type=float)
    ap.add_argument("--length_m", type=float)
    # sheet dims (m)
    ap.add_argument("--thickness_m", type=float)
    ap.add_argument("--width_m", type=float)
    ap.add_argument("--sheet_length_m", type=float)
    ap.add_argument("--csv", default="data/processed/train.csv")
    args = ap.parse_args()

    df = load_baselines(Path(args.csv))
    grade = args.grade if args.route=="conventional" else "na"
    bl = median_baseline(df, args.product, args.route, args.energy, grade)

    d = DashboardInput(
        product=args.product, units=args.units, route_type=args.route,
        bauxite_grade=grade, energy_source=args.energy, eol_option=args.eol,
        outer_radius_m=args.outer_radius_m, inner_radius_m=args.inner_radius_m, length_m=args.length_m,
        thickness_m=args.thickness_m, width_m=args.width_m, sheet_length_m=args.sheet_length_m
    )
    stages = build_stage_breakdown(d, bl)

    # Print per-stage and totals
    print("\nPER-UNIT STAGES")
    print(stages[stages["scope"]=="per_unit"][["stage","quality_score","electricity_kwh","carbon_kgco2e","natural_gas_nm3","wastewater_l","manufacturing_cost_per_unit_usd","transport_cost_usd"]].to_string(index=False))

    print("\nTOTAL STAGES")
    print(stages[stages["scope"]=="total"][["stage","electricity_kwh","carbon_kgco2e","natural_gas_nm3","wastewater_l","manufacturing_cost_per_unit_usd","transport_cost_usd"]].to_string(index=False))

    # Totals row
    tot = stages[stages["scope"]=="total"][["electricity_kwh","carbon_kgco2e","natural_gas_nm3","wastewater_l","manufacturing_cost_per_unit_usd","transport_cost_usd"]].sum()
    print("\nTOTALS SUMMARY")
    print(tot.to_string())

if __name__ == "__main__":
    main()
