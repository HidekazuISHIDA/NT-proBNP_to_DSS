from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.tree import plot_tree

def save_tree_top_levels(model, feature_names, out_png: str | Path, max_depth: int = 3) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["<=300", ">300"],
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def save_shap_summary(model, X: pd.DataFrame, out_png: str | Path, max_samples: int = 5000) -> None:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if len(X) > max_samples:
        X = X.sample(n=max_samples, random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
