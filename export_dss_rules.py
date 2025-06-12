```python

#!/usr/bin/env python
"""
export_dss_rules.py
-------------------

Convert a trained scikit-learn DecisionTreeClassifier into a two-column CSV
compatible with Abbott’s Diagnostic Support System (DSS) or its global
Clinical Decision Support (CDS) platform.

Each leaf becomes:

    IF <conditions> THEN "Predict NT-proBNP ≥ 300 pg mL⁻¹"  (or < 300)

Usage
-----
python export_dss_rules.py --model model.pkl --out dss_rules.csv
(optional)                 --features feature_list.txt
"""

import argparse
import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.tree import _tree


def _load_feature_names(model, feature_file=None):
    """Return feature names in the exact order used by the model."""
    if feature_file:
        names = Path(feature_file).read_text().splitlines()
        if len(names) != model.n_features_in_:
            raise ValueError("Feature file length ≠ model input size.")
        return names
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    raise ValueError("Feature names missing; supply --features FILE")


def _traverse(node_id, tree, feat_names, path, out_rows):
    """Depth-first traversal accumulating IF conditions."""
    feature = tree.feature[node_id]
    threshold = tree.threshold[node_id]

    if feature != _tree.TREE_UNDEFINED:  # internal node
        name = feat_names[feature]
        _traverse(tree.children_left[node_id],  tree, feat_names,
                  path + [f"{name} ≤ {threshold:.3g}"], out_rows)
        _traverse(tree.children_right[node_id], tree, feat_names,
                  path + [f"{name} > {threshold:.3g}"],  out_rows)
    else:  # leaf
        prob_pos = tree.value[node_id][0, 1] / tree.value[node_id][0].sum()
        pred_cls = int(np.argmax(tree.value[node_id][0]))
        rule_if   = " AND ".join(path) if path else "TRUE"
        rule_then = ("Predict NT-proBNP ≥ 300 pg mL⁻¹"
                     if pred_cls == 1 else
                     "Predict NT-proBNP < 300 pg mL⁻¹")
        out_rows.append((rule_if, rule_then, f"{prob_pos:.2%}"))


def export_rules(model_path, csv_path, feature_file=None):
    clf = joblib.load(model_path)
    if clf.__class__.__name__ != "DecisionTreeClassifier":
        raise TypeError("Model is not a DecisionTreeClassifier.")
    feat_names = _load_feature_names(clf, feature_file)

    rows = []
    _traverse(0, clf.tree_, feat_names, [], rows)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "comment", "leaf_probability"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} DSS rules → {csv_path}")


def main():
    p = argparse.ArgumentParser(description="Export decision-tree rules for DSS/CDS")
    p.add_argument("--model", required=True, help="Trained model pickle (.pkl)")
    p.add_argument("--out",   default="dss_rules.csv", help="Output CSV filename")
    p.add_argument("--features", help="Text file listing feature names")
    args = p.parse_args()
    export_rules(args.model, args.out, args.features)


if __name__ == "__main__":
    main()

```
