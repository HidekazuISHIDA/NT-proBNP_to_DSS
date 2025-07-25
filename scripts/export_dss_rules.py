```python
#!/usr/bin/env python3
"""
Export a trained sklearn DecisionTreeClassifier to a flat rule table (JSON/CSV)
suitable for DSS / CDS ingestion.

Usage:
  python scripts/export_dss_rules.py \
    --model models/model.pkl \
    --feature_order models/feature_order.json \
    --output dss_rules.json \
    --format json
"""
import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import _tree

def extract_rules(clf, feature_names, precision=4):
    """Traverse a sklearn DecisionTreeClassifier and return rule list."""
    tree_ = clf.tree_
    rules = []

    def recurse(node_id, conditions):
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feat = feature_names[tree_.feature[node_id]]
            thresh = round(tree_.threshold[node_id], precision)
            # left child: feature <= thresh
            recurse(tree_.children_left[node_id],
                    conditions + [f"{feat} <= {thresh}"])
            # right child: feature > thresh
            recurse(tree_.children_right[node_id],
                    conditions + [f"{feat} > {thresh}"])
        else:
            # leaf
            value = tree_.value[node_id][0]
            total = int(value.sum())
            pos = value[1] if len(value) > 1 else value[0]
            prob = float(pos / total) if total > 0 else 0.0
            label = int(prob >= 0.5)
            rules.append({
                "rule_id": len(rules) + 1,
                "if": " AND ".join(conditions) if conditions else "TRUE",
                "pred_label": label,
                "pred_prob": round(prob, 4),
                "n_samples": total
            })

    recurse(0, [])
    return rules

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model.pkl (DecisionTreeClassifier)")
    ap.add_argument("--feature_order", default="models/feature_order.json",
                    help="JSON listing training feature order")
    ap.add_argument("--output", required=True, help="Output file (json/csv)")
    ap.add_argument("--format", choices=["json", "csv"], default="json")
    args = ap.parse_args()

    clf = joblib.load(args.model)
    with open(args.feature_order, "r") as f:
        feature_names = json.load(f)

    rules = extract_rules(clf, feature_names)

    if args.format == "json":
        with open(args.output, "w") as f:
            json.dump(rules, f, indent=2)
    else:
        pd.DataFrame(rules).to_csv(args.output, index=False)

    print(f"Exported {len(rules)} rules to {args.output}")

if __name__ == "__main__":
    main()
