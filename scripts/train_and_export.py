from __future__ import annotations
import argparse
from pathlib import Path
import json

from ntprobnp_dss.config import load_config
from ntprobnp_dss.dataio import load_dataset, build_X_y
from ntprobnp_dss.train import train_and_validate
from ntprobnp_dss.export import save_model, save_threshold, save_schema
from ntprobnp_dss.explain import save_tree_top_levels, save_shap_summary
from ntprobnp_dss.rules import export_tree_structure

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    internal_df = load_dataset(cfg.data.internal_csv)
    external_df = load_dataset(cfg.data.external_csv)

    artifacts = train_and_validate(
        internal_df=internal_df,
        external_df=external_df,
        target_column=cfg.data.target_column,
        target_rule=cfg.data.target_rule,
        predictor_columns=cfg.data.predictor_columns,
        random_seed=cfg.training.random_seed,
        test_size=cfg.training.test_size,
        stratify=cfg.training.stratify,
        tree_params=dict(
            max_depth=cfg.decision_tree.max_depth,
            min_samples_leaf=cfg.decision_tree.min_samples_leaf,
            class_weight=cfg.decision_tree.class_weight,
        ),
        min_training_sensitivity=cfg.thresholding.min_training_sensitivity,
    )

    model_dir = Path(cfg.outputs.model_dir)
    fig_dir = Path(cfg.outputs.docs_fig_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_model(artifacts.model, model_dir / "model.joblib")
    save_threshold(artifacts.threshold, model_dir / "threshold.json")
    save_schema(artifacts.schema, model_dir / "feature_schema.json")

    # Figures (use all internal rows; SHAP subsamples if huge)
    X_all, _ = build_X_y(
        internal_df,
        cfg.data.target_column,
        cfg.data.target_rule,
        cfg.data.predictor_columns or artifacts.schema.columns
    )
    X_all = X_all[artifacts.schema.columns]
    save_tree_top_levels(artifacts.model, artifacts.schema.columns, fig_dir / "decision_tree_top3.png", max_depth=3)
    save_shap_summary(artifacts.model, X_all, fig_dir / "shap_summary.png")

    # Export DSS rules (full depth)
    rules = export_tree_structure(artifacts.model, artifacts.schema.columns)
    (model_dir / "rules").mkdir(exist_ok=True)
    (model_dir / "rules" / "tree_rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")

    # Save metrics
    metrics = {"internal": artifacts.internal_metrics, "external": artifacts.external_metrics}
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("=== Training complete ===")
    print("Chosen threshold:", artifacts.threshold.threshold)
    print("Internal metrics:", artifacts.internal_metrics)
    print("External metrics:", artifacts.external_metrics)
    print("Artifacts saved to:", model_dir)

if __name__ == "__main__":
    main()
