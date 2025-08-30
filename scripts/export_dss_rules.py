#!/usr/bin/env python3
"""
Export a scikit-learn DecisionTreeClassifier to a DSS/CDS-friendly JSON rule tree.

Examples
--------
python scripts/export_dss_rules.py \
  --model model/model.pkl \
  --output dss_rules.json

python scripts/export_dss_rules.py \
  --model model/model.pkl \
  --output dss_rules.json \
  --threshold 0.50 \
  --feature-names "Sex,Age,Na,K,Cl,TP,Alb,CK,AST,ALT,LD,Cre,UN,WBC,MCV,MCHC,RDWCV,PLT,eGFR" \
  --comment-template "Predicted risk of NT-proBNP > 300 pg/mL is elevated based on routine laboratories. Consider NT-proBNP testing and clinical assessment for heart failure."

Notes
-----
- Supports bare DecisionTreeClassifier or a Pipeline/CalibratedClassifier wrapper containing one.
- If feature names are not available via `feature_names_in_`, you may pass them via `--feature-names`.
- The exported JSON includes: version, threshold, features, and a nested node/leaf structure.
- Leaf probabilities come from `tree_.value` proportions (class index controlled by `--positive-class-index`).

JSON schema (abridged)
----------------------
{
  "version": "1.0.0",
  "exported_at": "2025-08-30T00:00:00Z",
  "threshold": 0.50,
  "features": ["eGFR", "Alb", "UN", ...],
  "nodes": { ... nested rule tree ... },
  "leaf_payload": {
    "emit_pred_prob": true,
    "emit_pred_label": true,
    "comment_template": "Predicted risk of NT-proBNP > 300 pg/mL ..."
  }
}
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from joblib import load as joblib_load

# ---------- helpers ----------
def find_tree_estimator(model) -> Any:
    """Return an object with `.tree_` (DecisionTreeClassifier-like)."""
    # bare estimator
    if hasattr(model, "tree_"):
        return model
    # pipeline
    if hasattr(model, "steps"):
        for name, step in model.steps:
            if hasattr(step, "tree_"):
                return step
            # calibrated
            if step.__class__.__name__ == "CalibratedClassifierCV" and hasattr(step, "base_estimator_"):
                base = step.base_estimator_
                if hasattr(base, "tree_"):
                    return base
    # calibrated at top-level
    if model.__class__.__name__ == "CalibratedClassifierCV" and hasattr(model, "base_estimator_"):
        base = model.base_estimator_
        if hasattr(base, "tree_"):
            return base
    raise ValueError("Could not locate a DecisionTreeClassifier with attribute `.tree_` in the provided model.")

def resolve_feature_names(model, user_list: Optional[List[str]], n_features: int) -> List[str]:
    if user_list:
        if len(user_list) != n_features:
            raise ValueError(f"--feature-names length ({len(user_list)}) != n_features ({n_features})")
        return user_list
    if hasattr(model, "feature_names_in_"):
        arr = list(model.feature_names_in_)
        if len(arr) != n_features:
            # tolerate mismatch by truncation/pad
            arr = (arr + [f"x{i}" for i in range(len(arr), n_features)])[:n_features]
        return arr
    # fallback
    return [f"x{i}" for i in range(n_features)]

def build_rule_tree(tree, features: List[str], positive_class_index: int = 1) -> Dict[str, Any]:
    """Recursively convert sklearn tree_ into nested dict rules."""
    children_left = tree.children_left
    children_right = tree.children_right
    feature = tree.feature
    threshold = tree.threshold
    value = tree.value  # shape [n_nodes, n_classes]
    n_node_samples = tree.n_node_samples

    def node_dict(node_id: int) -> Dict[str, Any]:
        is_leaf = (children_left[node_id] == children_right[node_id])
        if is_leaf:
            counts = value[node_id][0]  # (n_classes,)
            total = counts.sum()
            probs = (counts / total).tolist() if total > 0 else [0.0 for _ in counts]
            pred_idx = int(np.argmax(counts))
            return {
                "leaf": True,
                "n": int(n_node_samples[node_id]),
                "class_counts": [int(c) for c in counts],
                "probs": probs,
                "predicted_class_index": pred_idx,
                "predicted_label": int(pred_idx == positive_class_index)
            }
        else:
            feat_idx = feature[node_id]
            return {
                "leaf": False,
                "feature": features[feat_idx],
                "op": "<=",
                "value": float(threshold[node_id]),
                "left": node_dict(children_left[node_id]),
                "right": node_dict(children_right[node_id])
            }
    return node_dict(0)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Export DecisionTree rules to DSS/CDS JSON")
    ap.add_argument("--model", required=True, help="Path to model .pkl (DecisionTree or Pipeline)")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--threshold", type=float, default=0.50, help="Fixed operating threshold (default: 0.50)")
    ap.add_argument("--feature-names", type=str, default=None,
                    help="Comma-separated feature names if model lacks feature_names_in_")
    ap.add_argument("--positive-class-index", type=int, default=1,
                    help="Index of the positive class in tree_.value (default: 1)")
    ap.add_argument("--comment-template", type=str, default=(
        "Predicted risk of NT-proBNP > 300 pg/mL is elevated based on routine laboratory parameters. "
        "Consider NT-proBNP testing and clinical evaluation for heart failure."
    ), help="Interpretive comment template for positive predictions.")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent (default: 2)")
    args = ap.parse_args()

    model = joblib_load(args.model)
    est = find_tree_estimator(model)
    tree = est.tree_
    n_features = tree.n_features

    user_feats = [s.strip() for s in args.feature_names.split(",")] if args.feature_names else None
    features = resolve_feature_names(est, user_feats, n_features)

    rules = build_rule_tree(tree, features, positive_class_index=args.positive_class_index)

    payload = {
        "version": "1.0.0",
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "threshold": float(args.threshold),
        "features": features,
        "nodes": rules,
        "leaf_payload": {
            "emit_pred_prob": True,
            "emit_pred_label": True,
            "comment_template": args.comment_template
        },
        "notes": {
            "index_test": "Decision-tree probability with fixed decision rule",
            "reference_standard": "NT-proBNP > 300 pg/mL",
            "disclaimer": "For research use; site validation required prior to clinical deployment."
        }
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=args.indent))
    print(f"Wrote {out} (features={len(features)}, threshold={args.threshold})")

if __name__ == "__main__":
    main()
