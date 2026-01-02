from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.data_loader import load_all
from src.data.merge import merge_datasets, save_training_table

# Try optional boosters
HAS_XGB = False
HAS_LGBM = False

try:
    from xgboost import XGBRegressor  # type: ignore
    HAS_XGB = True
except Exception:
    pass

try:
    from lightgbm import LGBMRegressor  # type: ignore
    HAS_LGBM = True
except Exception:
    pass

MODELS_DIR = Path("models") / "trained"
EXP_DIR = Path("models") / "experiments"


def _build_preprocessor(df: pd.DataFrame, target: str) -> Tuple[Pipeline, List[str], List[str]]:
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop",
    )
    
    pipe = Pipeline([("pre", pre)])
    return pipe, num_cols, cat_cols


def _candidates(random_state: int = 42) -> Dict[str, object]:
    cand: Dict[str, object] = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_split=4, random_state=random_state, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
    }
    
    if HAS_XGB:
        cand["XGBRegressor"] = XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=random_state, n_jobs=-1, tree_method="hist"
        )
    
    if HAS_LGBM:
        cand["LGBMRegressor"] = LGBMRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=-1, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=random_state, n_jobs=-1
        )
    
    return cand


def _fit_and_score(
    name: str,
    estimator,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray
) -> Dict:
    estimator.fit(X_tr, y_tr)
    pred = estimator.predict(X_te)
    
    # FIXED: Use sqrt for RMSE calculation instead of squared=False parameter
    mse = mean_squared_error(y_te, pred)
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_te, pred))
    
    return {"model_name": name, "rmse": rmse, "r2": r2}


def _train_one_target(df: pd.DataFrame, target: str, out_name: str, random_state: int = 42) -> Dict:
    from time import time
    import joblib
    
    print(f"\n{'='*50}")
    print(f"Training models for target: {target}")
    print(f"{'='*50}")
    
    df_ = df.dropna(subset=[target]).copy()
    if df_.empty:
        print(f"No data available for target: {target}")
        return {"target": target, "winner": None, "rmse": None, "r2": None, "n_test": 0, "skipped": True}
    
    y = df_[target].values
    pre, num_cols, cat_cols = _build_preprocessor(df_, target)
    X = df_.drop(columns=[target])
    
    # Fit preprocessor and transform
    X_proc = pre.fit_transform(X)
    
    # Split once, shared by all candidates
    X_tr, X_te, y_tr, y_te = train_test_split(X_proc, y, test_size=0.2, random_state=random_state)
    
    print(f"Training set size: {len(X_tr)}")
    print(f"Test set size: {len(X_te)}")
    print("\nTraining models...")
    
    # Evaluate candidates
    leaderboard: List[Dict] = []
    best = {"model_name": None, "rmse": np.inf, "r2": -np.inf, "estimator": None}
    
    for mname, est in _candidates(random_state).items():
        print(f"\n  Training {mname}...", end=" ")
        start = time()
        metrics = _fit_and_score(mname, est, X_tr, y_tr, X_te, y_te)
        metrics["train_seconds"] = round(time() - start, 3)
        leaderboard.append(metrics)
        
        print(f"RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}, Time: {metrics['train_seconds']}s")
        
        # Primary: RMSE (lower is better), secondary: R2 (higher is better)
        if (metrics["rmse"] < best["rmse"]) or (
            np.isclose(metrics["rmse"], best["rmse"]) and metrics["r2"] > best["r2"]
        ):
            best = {"model_name": mname, "rmse": metrics["rmse"], "r2": metrics["r2"], "estimator": est}
    
    # Sort leaderboard by RMSE (ascending)
    leaderboard.sort(key=lambda x: x["rmse"])
    
    print(f"\n📊 LEADERBOARD for {target}:")
    print("-" * 70)
    print(f"{'Rank':<5} {'Model':<20} {'RMSE':<12} {'R²':<12} {'Time (s)':<10}")
    print("-" * 70)
    for i, model in enumerate(leaderboard, 1):
        winner_mark = "🏆" if model["model_name"] == best["model_name"] else "  "
        print(f"{winner_mark}{i:<3} {model['model_name']:<20} {model['rmse']:<12.4f} {model['r2']:<12.4f} {model['train_seconds']:<10}")
    
    print(f"\n🎯 WINNER: {best['model_name']} (RMSE: {best['rmse']:.4f}, R²: {best['r2']:.4f})")
    
    # Persist leaderboard
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXP_DIR / f"{out_name}_leaderboard.json", "w") as f:
        json.dump({"target": target, "leaderboard": leaderboard}, f, indent=2)
    
    # Save winning bundle (pre + best estimator)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": best["estimator"], "pre": pre, "target": target, "winner": best["model_name"]}
    joblib.dump(bundle, MODELS_DIR / f"{out_name}.pkl")
    print(f"✅ Model saved: {MODELS_DIR / f'{out_name}.pkl'}")
    
    # Also persist per-winner metrics for quick view
    metrics = {"target": target, "winner": best["model_name"], "rmse": best["rmse"], "r2": best["r2"], "n_test": int(len(y_te))}
    with open(EXP_DIR / f"{out_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train multi-model regressors and select best by RMSE.")
    parser.add_argument("--out", default="data/processed/train.csv", help="Output merged CSV path")
    args = parser.parse_args()
    
    out_path = Path(args.out)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting training pipeline...")
    print(f"Available models: RandomForest, GradientBoosting", end="")
    if HAS_XGB:
        print(", XGBRegressor", end="")
    if HAS_LGBM:
        print(", LGBMRegressor", end="")
    print("\n")
    
    # 1) Load and merge raw CSVs, then save processed table
    print("📂 Loading and merging datasets...")
    dfs = load_all(Path("data/raw"))
    merged = merge_datasets(dfs)
    save_training_table(merged, out_path)
    print(f"✅ Merged dataset saved: {out_path}")
    print(f"Dataset shape: {merged.shape}")
    
    # 2) Targets to train (only if present)
    targets = [
        c for c in ["electricity_kwh", "carbon_kgco2e", "manufacturing_cost_per_unit","naturalGas_nm3","wastewater_l","wastewater_l","flourine_g","so2_g","Quality_Score"] if c in merged.columns
    ]
    
    name_map = {
        "electricity_kwh": "energy_predictor",
        "carbon_kgco2e": "emissions_predictor",
        "manufacturing_cost_per_unit": "cost_predictor",
        "naturalGas_nm3" : "natural_gas",
        "wastewater_l" : "water_waste",
        "flourine_g" : "Flourine_used",
        "so2_g" : "so2_g",
        "Quality_Score":"Quality_predictor"

    }
    
    if not targets:
        print("❌ No target columns found in the dataset!")
        return
    
    print(f"📈 Training targets: {targets}")
    
    # 3) Train per target with a model zoo
    all_metrics: Dict[str, Dict] = {}
    
    for tgt in targets:
        out_name = name_map.get(tgt, tgt)
        metrics = _train_one_target(merged, tgt, out_name)
        all_metrics[out_name] = metrics
    
    # Save summary
    with open(EXP_DIR / "summary_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n{'='*70}")
    print("🎉 TRAINING COMPLETE - FINAL SUMMARY")
    print(f"{'='*70}")
    
    print(f"{'Target':<25} {'Winner Model':<20} {'RMSE':<12} {'R²':<12} {'Test Size':<10}")
    print("-" * 70)
    for name, metrics in all_metrics.items():
        if not metrics.get("skipped", False):
            print(f"{metrics['target']:<25} {metrics['winner']:<20} {metrics['rmse']:<12.4f} {metrics['r2']:<12.4f} {metrics['n_test']:<10}")
        else:
            print(f"{metrics['target']:<25} {'SKIPPED (no data)':<20} {'-':<12} {'-':<12} {'-':<10}")
    
    print(f"\n📁 Models saved in: {MODELS_DIR}")
    print(f"📊 Experiments saved in: {EXP_DIR}")


if __name__ == "__main__":
    main()