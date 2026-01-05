from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

RAW = Path("data/raw")

FILES = {
    # product-form
    "sheet_ren": "aluminum_sheet_renewable_energy_100_samples.csv",
    "sheet_non": "aluminum_sheet_non_renewable_energy_100_samples.csv",
    "pipe_ren": "aluminium_pipe_renewable_energy_200.csv",
    "pipe_non": "aluminium_pipe_non_renewable_energy_200.csv",
    # recycled route
    "recycle_a": "recycled_route_scrap_1kg_aluminium_100_samples.csv",
    "recycle_b": "recycled_route_scrap_1kg_aluminium_100_samples1.csv",
    # bauxite grades
    "baux_high": "high_grade_bauxite_1kg_samples.csv",
    "baux_med": "medium_grade_bauxite_1kg_samples.csv",
    "baux_low": "low_grade_bauxite_1kg_samples.csv",
}

def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {suf}")

def _std_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def _attach_context(df: pd.DataFrame, **context) -> pd.DataFrame:
    for k, v in context.items():
        df[k] = v
    return df

def load_all(raw_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Load all 9 CSVs into memory with normalized columns and context columns:
      - product_type: sheet|pipe|na
      - route_type: bauxite|recycle|na
      - energy_source: renewable|non_renewable|na
      - grade: high|medium|low|na
    """
    base = raw_dir or RAW
    out: Dict[str, pd.DataFrame] = {}

    # Sheet
    df = _std_columns(_read_any(base / FILES["sheet_ren"]))
    out["sheet_ren"] = _attach_context(df, product_type="sheet", route_type="na", energy_source="renewable", grade="na")
    df = _std_columns(_read_any(base / FILES["sheet_non"]))
    out["sheet_non"] = _attach_context(df, product_type="sheet", route_type="na", energy_source="non_renewable", grade="na")

    # Pipe
    df = _std_columns(_read_any(base / FILES["pipe_ren"]))
    out["pipe_ren"] = _attach_context(df, product_type="pipe", route_type="na", energy_source="renewable", grade="na")
    df = _std_columns(_read_any(base / FILES["pipe_non"]))
    out["pipe_non"] = _attach_context(df, product_type="pipe", route_type="na", energy_source="non_renewable", grade="na")

    # Recycle
    df = _std_columns(_read_any(base / FILES["recycle_a"]))
    out["recycle_a"] = _attach_context(df, product_type="na", route_type="recycle", energy_source="na", grade="na")
    df = _std_columns(_read_any(base / FILES["recycle_b"]))
    out["recycle_b"] = _attach_context(df, product_type="na", route_type="recycle", energy_source="na", grade="na")

    # Bauxite grades
    df = _std_columns(_read_any(base / FILES["baux_high"]))
    out["baux_high"] = _attach_context(df, product_type="na", route_type="bauxite", energy_source="na", grade="high")
    df = _std_columns(_read_any(base / FILES["baux_med"]))
    out["baux_med"] = _attach_context(df, product_type="na", route_type="bauxite", energy_source="na", grade="medium")
    df = _std_columns(_read_any(base / FILES["baux_low"]))
    out["baux_low"] = _attach_context(df, product_type="na", route_type="bauxite", energy_source="na", grade="low")


    return out
