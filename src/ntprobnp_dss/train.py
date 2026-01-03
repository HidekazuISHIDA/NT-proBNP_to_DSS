from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from .dataio import build_X_y, infer_schema
from .threshold import choose_threshold_for_sensitivity, ThresholdResult

@dataclass(frozen=True)
class TrainArtifacts:
    model: DecisionTreeClassifier
    schema: Any
    threshold: ThresholdResult
    internal_metrics: Dict[str, float]
    external_metrics: Dict[str, float]

def _eval_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_pred==1) & (y_true==1)).sum())
    fp = int(((y_pred==1) & (y_true==0)).sum())
    tn = int(((y_pred==0) & (y_true==0)).sum())
    fn = int(((y_pred==0) & (y_true==1)).sum())
    sens = tp/(tp+fn) if (tp+fn)>0 else float("nan")
    spec = tn/(tn+fp) if (tn+fp)>0 else float("nan")
    ppv  = tp/(tp+fp) if (tp+fp)>0 else float("nan")
    npv  = tn/(tn+fn) if (tn+fn)>0 else float("nan")
    return dict(sensitivity=float(sens), specificity=float(spec), ppv=float(ppv), npv=float(npv))

def train_and_validate(
    internal_df: pd.DataFrame,
    external_df: pd.DataFrame,
    target_column: str,
    target_rule: Dict[str, Any],
    predictor_columns: Optional[list[str]],
    random_seed: int,
    test_size: float,
    stratify: bool,
    tree_params: Dict[str, Any],
    min_training_sensitivity: float,
) -> TrainArtifacts:
    # Internal
    X, y = build_X_y(internal_df, target_column, target_rule, predictor_columns)
    schema = infer_schema(X)

    strat = y if stratify else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=strat
    )

    model = DecisionTreeClassifier(
        random_state=random_seed,
        max_depth=tree_params["max_depth"],
        min_samples_leaf=tree_params["min_samples_leaf"],
        class_weight=tree_params.get("class_weight"),
    )
    model.fit(X_tr, y_tr)

    # Threshold selection on TRAIN (matches revised manuscript intent)
    tr_prob = model.predict_proba(X_tr)[:,1]
    thr_res = choose_threshold_for_sensitivity(y_tr.to_numpy(), tr_prob, min_training_sensitivity)

    # Internal test metrics
    te_prob = model.predict_proba(X_te)[:,1]
    internal_auc = roc_auc_score(y_te, te_prob)
    internal_at_thr = _eval_at_threshold(y_te.to_numpy(), te_prob, thr_res.threshold)

    # External validation
    X_ext, y_ext = build_X_y(external_df, target_column, target_rule, predictor_columns or schema.columns)
    X_ext = X_ext[schema.columns]
    ext_prob = model.predict_proba(X_ext)[:,1]
    external_auc = roc_auc_score(y_ext, ext_prob)
    external_at_thr = _eval_at_threshold(y_ext.to_numpy(), ext_prob, thr_res.threshold)

    internal_metrics = {"auc": float(internal_auc), **internal_at_thr}
    external_metrics = {"auc": float(external_auc), **external_at_thr}

    return TrainArtifacts(
        model=model,
        schema=schema,
        threshold=thr_res,
        internal_metrics=internal_metrics,
        external_metrics=external_metrics,
    )
