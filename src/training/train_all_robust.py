from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

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
    
    # Add imputers to handle any remaining NaN values
    num_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    pre = ColumnTransformer(
        transformers=[
            ("num", num_preprocessor, num_cols),
            ("cat", cat_preprocessor, cat_cols),
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
        # Use HistGradientBoosting instead of regular GradientBoosting for better NaN handling
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_depth=3, random_state=random_state
        ),
        # Keep regular GradientBoosting as backup (now with proper preprocessing)
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


