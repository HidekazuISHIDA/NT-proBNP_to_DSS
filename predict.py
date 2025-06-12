## Quick look at `predict.py`

```python
#!/usr/bin/env python
"""
predict.py
-----------

Command-line script to assess a pretrained decision-tree model that predicts
whether NT-proBNP is ≥ 300 pg mL⁻¹.

Usage
-----
$ python predict.py --data sample_data.csv --model model.pkl
(optional)       --out  predictions.csv
"""

import argparse
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score


def load_data(csv_path: str):
    """Return features X and ground-truth labels y from a CSV file."""
    df = pd.read_csv(csv_path)
    if "NTproBNP_300" not in df.columns:
        raise ValueError("Input CSV must contain a 'NTproBNP_300' label column.")
    y = df.pop("NTproBNP_300").values
    return df, y


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the pretrained NT-proBNP decision-tree model."
    )
    parser.add_argument("--data", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--model", default="model.pkl", help="Path to pretrained model (joblib/pkl)"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save per-row probabilities as a CSV",
    )
    args = parser.parse_args()

    # Load model and data
    clf = joblib.load(args.model)
    X, y_true = load_data(args.data)

    # Predict
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)  # default threshold

    # Metrics
    auroc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    print(f"AUROC : {auroc:.2f}")
    print(f"F1    : {f1:.2f}")

    # Optional: save probabilities
    if args.out:
        X_out = X.copy()
        X_out["prob_≥300"] = y_prob
        X_out.to_csv(args.out, index=False)
        print(f"Predictions saved to '{args.out}'")


if __name__ == "__main__":
    main()

```
