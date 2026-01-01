import sys
from pathlib import Path
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Local imports
from src.inference.stage_engine import DashboardInput, build_stage_breakdown
from src.inference.baselines import load_baselines, median_baseline

# ------------------------------------------------------------------------------
# App setup
# ------------------------------------------------------------------------------
app = Flask(__name__, template_folder='../web', static_folder='../web', static_url_path='/static')
CORS(app)

# Choose which processed CSV to use (update if needed)
DATA_CSV = Path("data/processed/train_1.csv")
if not DATA_CSV.exists():
    # fallback to default
    DATA_CSV = Path("data/processed/train.csv")

# ------------------------------------------------------------------------------
# Schemas
# ------------------------------------------------------------------------------
class DashboardRequest:
    def __init__(self, product: str, units: int, route_type: str, bauxite_grade: str, energy_source: str, eol_option: str, outer_radius_m: Optional[float], inner_radius_m: Optional[float], length_m: Optional[float], thickness_m: Optional[float], width_m: Optional[float], sheet_length_m: Optional[float]):
        self.product = product
        self.units = units
        self.route_type = route_type
        self.bauxite_grade = bauxite_grade
        self.energy_source = energy_source
        self.eol_option = eol_option
        self.outer_radius_m = outer_radius_m
        self.inner_radius_m = inner_radius_m
        self.length_m = length_m
        self.thickness_m = thickness_m
        self.width_m = width_m
        self.sheet_length_m = sheet_length_m


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    req = DashboardRequest(**data)

    # Load baselines table once per request (simple; can be cached in memory later)
    df = load_baselines(DATA_CSV)

    # Grade applies only for conventional route
    grade = req.bauxite_grade if req.route_type == "conventional" else "na"

    # Get median per-kg baselines for this context (product/route/energy[/grade])
    bl = median_baseline(
        df,
        product=req.product,
        route_type=req.route_type,
        energy_source=req.energy_source,
        grade=grade,
    )

    # Prepare input object for stage breakdown
    d = DashboardInput(
        product=req.product,
        units=req.units,
        route_type=req.route_type,
        bauxite_grade=grade,
        energy_source=req.energy_source,
        eol_option=req.eol_option,
        outer_radius_m=req.outer_radius_m,
        inner_radius_m=req.inner_radius_m,
        length_m=req.length_m,
        thickness_m=req.thickness_m,
        width_m=req.width_m,
        sheet_length_m=req.sheet_length_m,
    )

    # Compute per-stage dataframe (both per_unit and total scopes)
    stages_df = build_stage_breakdown(d, bl)

    # Build totals for both scopes
    def totals(scope: str):
        sub = stages_df[stages_df["scope"] == scope]
        if len(sub) == 0:
            return {
                "scope": scope,
                "units": req.units,
                "manufacturing_cost_per_unit": 0.0,
                "electricity_kwh": 0.0,
                "carbon_kgco2e": 0.0,
                "naturalGas_nm3": 0.0,
                "wastewater_l": 0.0,
                "transport_cost_usd": 0.0,
                "Quality_Score": None,
            }
        return {
            "scope": scope,
            "units": int(sub["units"].iloc[0]),
            "manufacturing_cost_per_unit": float(sub["manufacturing_cost_per_unit_usd"].sum()),
            "electricity_kwh": float(sub["electricity_kwh"].sum()),
            "carbon_kgco2e": float(sub["carbon_kgco2e"].sum()),
            "naturalGas_nm3": float(sub["naturalGas_nm3"].sum()),
            "wastewater_l": float(sub["wastewater_l"].sum()),
            "transport_cost_usd": float(sub["transport_cost_usd"].sum()),
            "Quality_Score": float(sub["quality_score"].mean()) if "quality_score" in sub.columns else None,
        }

    response = {
        "stages": stages_df.to_dict(orient="records"),
        "totals": {
            "per_unit": totals("per_unit"),
            "total": totals("total"),
        },
        "baselines_used": bl,
        "data_csv": str(DATA_CSV),
    }
    return jsonify(response)


@app.route("/health")
def health():
    return {"status": "ok", "data_csv": str(DATA_CSV)}


if __name__ == '__main__':
    app.run(debug=True)
