# Path: src/data/merge.py
from __future__ import annotations
from typing import Dict, List
import pandas as pd
from pathlib import Path
from pathlib import Path
import pandas as pd
import os
import time
import uuid


# Map YOUR columns → canonical names used by training
COLUMN_ALIASES = {
    # If your raw sets use these summary names, map them to canonical targets:
    "totalelectricitykwh": "electricity_kwh",
    "totalcarbonkgco2e": "carbon_kgco2e",
    # If you have cost column:
    #"total_cost": "manufacturing_cost_per_unit",
    # If your files explicitly use per-tonne names, prefer these instead:
    #"energy_kwh_per_tonne": "electricity_kwh",
    #"co2_kg_per_tonne": "carbon_kgco2e",
}

CANONICAL_NUMERICS = [
    "electricity_kwh",
    "carbon_kgco2e",
    "manufacturing_cost_per_unit",
    "yield_pct",
    "recovery_pct",
]

CONTEXT_COLS = ["route_type", "product_type", "energy_source", "grade"]

def _normalize_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # rename aliases to canonical names
    for src, tgt in COLUMN_ALIASES.items():
        if src in df.columns and tgt not in df.columns:
            df.rename(columns={src: tgt}, inplace=True)

    # heuristics: if the canonical values look like per-tonne magnitudes, downscale to per-kg
    if "electricity_kwh" in df.columns:
        med = pd.to_numeric(df["electricity_kwh"], errors="coerce").median()
        if pd.notna(med) and med > 100:  # e.g., 12,000 kWh/t → 12 kWh/kg
            df["electricity_kwh"] = pd.to_numeric(df["electricity_kwh"], errors="coerce") / 1000.0

    if "carbon_kgco2e" in df.columns:
        med = pd.to_numeric(df["carbon_kgco2e"], errors="coerce").median()
        if pd.notna(med) and med > 20:   # e.g., 10,000 kg/t → 10 kg/kg
            df["carbon_kgco2e"] = pd.to_numeric(df["carbon_kgco2e"], errors="coerce") / 1000.0

    if "manufacturing_cost_per_unit" in df.columns:
        med = pd.to_numeric(df["manufacturing_cost_per_unit"], errors="coerce").median()
        if pd.notna(med) and med > 100:
            df["manufacturing_cost_per_unit"] = pd.to_numeric(df["manufacturing_cost_per_unit"], errors="coerce") / 1000.0

    return df

def _select_columns(df: pd.DataFrame) -> pd.DataFrame:
    # keep context + all canonical targets + all numeric features as inputs
    keep = set(CONTEXT_COLS)
    keep.update([c for c in df.columns if c in CANONICAL_NUMERICS])
    numeric_extras = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in keep]
    keep.update(numeric_extras)
    cols = [c for c in df.columns if c in keep]
    return df[cols].copy()

def merge_datasets(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    bag: List[pd.DataFrame] = []
    for name, df in dfs.items():
        tdf = _normalize_units(df)
        # ensure context columns exist
        for c in CONTEXT_COLS:
            if c not in tdf.columns:
                tdf[c] = "na"
        tdf = _select_columns(tdf)
        tdf["source_name"] = name
        bag.append(tdf)

    merged = pd.concat(bag, ignore_index=True, sort=False)

    # drop rows where both energy and co2 are missing (keep rows if at least one target exists)
    targets = [c for c in ["electricity_kwh", "carbon_kgco2e"] if c in merged.columns]
    if targets:
        merged = merged.dropna(subset=targets, how="all")

    # clamp negatives to zero for targets and cost
    for c in ["electricity_kwh", "carbon_kgco2e", "manufacturing_cost_per_unit"]:
        if c in merged.columns:
            merged.loc[merged[c] < 0, c] = 0.0

    return merged


def save_training_table(df: pd.DataFrame, out_path: Path = Path("data/processed/train.csv")) -> Path:
    """
    Robust writer that handles Windows file locks:
    1) Try writing directly.
    2) If PermissionError, write a temp file and attempt to replace the target a few times.
    3) If still locked, keep the temp file and return its path with a warning.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt direct write first
    try:
        df.to_csv(out_path, index=False)
        return out_path
    except PermissionError:
        pass  # fall through to temp-write/replace

    # Temp-write then replace
    tmp = out_path.with_suffix(out_path.suffix + f".tmp.{uuid.uuid4().hex}")
    df.to_csv(tmp, index=False)

    for _ in range(8):  # try up to ~4 seconds
        try:
            if out_path.exists():
                os.remove(out_path)  # fails if still locked
            os.replace(tmp, out_path)
            return out_path
        except PermissionError:
            time.sleep(0.5)

    print(f"WARNING: Could not overwrite {out_path}. Wrote {tmp} instead. "
          f"Close any program using the file and rename it manually.")
    return tmp
