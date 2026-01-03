from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import pandas as pd

@dataclass(frozen=True)
class FeatureSchema:
    columns: List[str]
    dtypes: Dict[str, str]

def make_binary_target(series: pd.Series, rule: Dict[str, Any]) -> pd.Series:
    if rule.get("kind") != "threshold":
        raise ValueError(f"Unsupported target rule: {rule}")
    thr = float(rule["threshold"])
    pos_if = rule.get("positive_if", ">")
    if pos_if == ">":
        return (series > thr).astype(int)
    if pos_if == ">=":
        return (series >= thr).astype(int)
    if pos_if == "<":
        return (series < thr).astype(int)
    if pos_if == "<=":
        return (series <= thr).astype(int)
    raise ValueError(f"Unsupported positive_if: {pos_if}")

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def build_X_y(
    df: pd.DataFrame,
    target_column: str,
    target_rule: Dict[str, Any],
    predictor_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    y = make_binary_target(df[target_column], target_rule)

    if predictor_columns is None:
        X = df.drop(columns=[target_column])
    else:
        missing = [c for c in predictor_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing predictors in data: {missing}")
        X = df[predictor_columns].copy()

    # Coerce object -> numeric when possible
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.to_numeric(X[c], errors="ignore")

    return X, y

def infer_schema(X: pd.DataFrame) -> FeatureSchema:
    return FeatureSchema(columns=list(X.columns), dtypes={c: str(X[c].dtype) for c in X.columns})

def enforce_schema(X: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    missing = [c for c in schema.columns if c not in X.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    extra = [c for c in X.columns if c not in schema.columns]
    if extra:
        X = X.drop(columns=extra)

    X = X[schema.columns].copy()

    # dtype coercion best-effort
    for c, dt in schema.dtypes.items():
        try:
            X[c] = X[c].astype(dt)
        except Exception:
            pass

    return X
