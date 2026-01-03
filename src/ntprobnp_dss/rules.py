from __future__ import annotations
from typing import Dict, Any, List

def export_tree_structure(model, feature_names: List[str]) -> Dict[str, Any]:
    tree = model.tree_
    out = {"nodes": []}
    for i in range(tree.node_count):
        feat_idx = int(tree.feature[i])
        feature = None if feat_idx < 0 else feature_names[feat_idx]
        thr = None if feat_idx < 0 else float(tree.threshold[i])
        left = None if tree.children_left[i] == -1 else int(tree.children_left[i])
        right = None if tree.children_right[i] == -1 else int(tree.children_right[i])
        out["nodes"].append({
            "node_id": int(i),
            "feature": feature,
            "threshold": thr,
            "left": left,
            "right": right,
            "is_leaf": (left is None and right is None),
            "value": tree.value[i].tolist(),
        })
    out["n_classes"] = int(model.n_classes_)
    out["classes"] = [int(c) for c in model.classes_]
    return out
