from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import joblib
from .dataio import FeatureSchema
from .threshold import ThresholdResult

def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def save_threshold(thr: ThresholdResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(thr), indent=2), encoding="utf-8")

def save_schema(schema: FeatureSchema, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"columns": schema.columns, "dtypes": schema.dtypes}, indent=2), encoding="utf-8")

def load_schema(path: str | Path) -> FeatureSchema:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return FeatureSchema(columns=obj["columns"], dtypes=obj["dtypes"])
