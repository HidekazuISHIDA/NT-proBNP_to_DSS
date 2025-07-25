#!/usr/bin/env python3
"""
One-shot inference from a CSV using a saved scaler + decision tree.

Usage:
  python scripts/predict.py \
    --input data/sample_data.csv \
    --model models/model.pkl \
    --scaler models/standard_scaler.pkl \
    --output results/predictions.csv \
    --threshold 0.50
"""
import argparse
import json
import os
import pandas as pd
import joblib
import numpy as np

def load_feature_order(path_json):
    if path_json and os.path.exists(path_json):
        with open(path_json, "r") as f:
            return json.load(f)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV file (features only).")
    ap.add_argument("--model", required=True, help="Path to model.pkl")
    ap.add_argument("--scaler", required=True, help="Path to standard_scaler.pkl")
    ap.add_argument("--feature_order", default="models/feature_order.json",
                    help="Optional JSON listing feature order used in training.")
    ap.add_argument("--output", default="results/predictions.csv", help="Output CSV path.")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold.")
    ap.add_argument("--id_col", default=None, help="Optional ID column name to carry through.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load inputs
    df = pd.read_csv(args.input)
    feat_order = load_feature_order(args.feature_order)
    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)

    # If ID col provided, keep it aside
    id_series = df[args.id_col] if args.id_col and args.id_col in df.columns else None
    if args.id_col and args.id_col in df.columns:
        df = df.drop(columns=[args.id_col])

    # Reorder / subset / warn
    if feat_order:
        missing = [c for c in feat_order if c not in df.columns]
        extra   = [c for c in df.columns if c not in feat_order]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df = df[feat_order]
        if extra:
            # harmless: user gave more cols; we dropped them
            pass

    X_scaled = scaler.transform(df.values)
    prob = model.predict_proba(X_scaled)[:, 1]
    label = (prob >= args.threshold).astype(int)

    out = pd.DataFrame({
        "pred_prob": prob,
        "pred_label": label
    })
    if id_series is not None:
        out.insert(0, args.id_col, id_series.values)

    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
    print(f"Mean probability: {prob.mean():.3f}")
    print(f"Positive rate @ {args.threshold:.2f}: {label.mean():.3f}")

if __name__ == "__main__":
    main()
