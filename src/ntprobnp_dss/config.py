from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

@dataclass(frozen=True)
class DataConfig:
    internal_csv: str
    external_csv: str
    target_column: str
    target_rule: Dict[str, Any]
    predictor_columns: Optional[List[str]]

@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int
    test_size: float
    stratify: bool

@dataclass(frozen=True)
class TreeConfig:
    max_depth: int
    min_samples_leaf: int
    class_weight: Optional[str]

@dataclass(frozen=True)
class ThresholdConfig:
    min_training_sensitivity: float

@dataclass(frozen=True)
class OutputConfig:
    model_dir: str
    docs_fig_dir: str

@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    training: TrainingConfig
    decision_tree: TreeConfig
    thresholding: ThresholdConfig
    outputs: OutputConfig

def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    data = raw["data"]
    training = raw["training"]
    tree = raw["decision_tree"]
    thr = raw["thresholding"]
    out = raw["outputs"]

    return AppConfig(
        data=DataConfig(
            internal_csv=data["internal_csv"],
            external_csv=data["external_csv"],
            target_column=data["target_column"],
            target_rule=data["target_rule"],
            predictor_columns=data.get("predictor_columns"),
        ),
        training=TrainingConfig(
            random_seed=int(training["random_seed"]),
            test_size=float(training["test_size"]),
            stratify=bool(training.get("stratify", True)),
        ),
        decision_tree=TreeConfig(
            max_depth=int(tree["max_depth"]),
            min_samples_leaf=int(tree["min_samples_leaf"]),
            class_weight=tree.get("class_weight"),
        ),
        thresholding=ThresholdConfig(
            min_training_sensitivity=float(thr["min_training_sensitivity"])
        ),
        outputs=OutputConfig(
            model_dir=out["model_dir"],
            docs_fig_dir=out["docs_fig_dir"],
        ),
    )
