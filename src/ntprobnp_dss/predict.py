from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import joblib
from .dataio import load_dataset, enforce_schema
from .export import load_schema

def load_threshold(threshold_json: str | Path) -> float:
    obj = json.loads(Path(threshold_json).read_text(encoding="utf-8"))
    return float(obj["threshold"])

def predict_csv(model_path: str, schema_path: str, threshold_path: str, input_csv: str, output_csv: str) -> None:
    model = joblib.load(model_path)
    schema = load_schema(schema_path)
    thr = load_threshold(threshold_path)

    df = load_dataset(input_csv)
    X = enforce_schema(df, schema)

    prob = model.predict_proba(X)[:,1]
    pred = (prob >= thr).astype(int)

    out = df.copy()
    out["pred_prob"] = prob
    out["pred_label"] = pred
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
