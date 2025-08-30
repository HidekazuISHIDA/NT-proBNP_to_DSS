#!/usr/bin/env python3
"""
Reproduce Table 2 metrics at a fixed threshold = 0.50.

Outputs a CSV with rows:
  cohort, N, prevalence, AUROC, AUROC_lo, AUROC_hi,
  sensitivity, sens_lo, sens_hi,
  specificity, spec_lo, spec_hi,
  PPV, PPV_lo, PPV_hi,
  NPV, NPV_lo, NPV_hi,
  accuracy, acc_lo, acc_hi, threshold

Assumptions:
- CSVs contain the predictor columns expected by the scaler/model and one
  outcome column named (first match): 'label', 'y', 'target', 'outcome', 'NPB300'.
- Model is a scikit-learn estimator supporting predict_proba or decision_function.
- Scaler is a scikit-learn transformer (e.g., StandardScaler).

Usage:
python scripts/make_table2.py --model model/model.pkl --scaler model/standard_scaler.pkl \
  --train data/train.csv --test data/test.csv --external data/external.csv \
  --out results/table2.csv
"""
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from joblib import load as joblib_load


# --------------------- CI utilities --------------------- #
def wilson_ci(x, n, alpha=0.05):
    """Wilson score CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.959963984540054 if alpha == 0.05 else abs(__import__("scipy").stats.norm.ppf(alpha/2))
    phat = x / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n) / denom
    return (center - half, center + half)

def delong_roc_ci(y_true, y_score, alpha=0.05, n_boot=2000, seed=42):
    """DeLong 95% CI for AUROC with bootstrap fallback."""
    auc = roc_auc_score(y_true, y_score)
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score).astype(float)
    pos = scores[y == 1]
    neg = scores[y == 0]
    nx, ny = len(pos), len(neg)
    if nx == 0 or ny == 0:
        return float(auc), float("nan"), float("nan")
    try:
        X = pos[:, None]; Y = neg[None, :]
        V = (X > Y).astype(float) + 0.5*(X == Y).astype(float)
        v10 = V.mean(axis=1); v01 = V.mean(axis=0)
        s10 = np.var(v10, ddof=1); s01 = np.var(v01, ddof=1)
        var = s10/nx + s01/ny
        if var <= 0 or np.isnan(var):
            raise RuntimeError
        z = 1.959963984540054 if alpha == 0.05 else abs(__import__("scipy").stats.norm.ppf(alpha/2))
        se = math.sqrt(var)
        lo, hi = max(0.0, auc - z*se), min(1.0, auc + z*se)
        return float(auc), float(lo), float(hi)
    except Exception:
        rng = np.random.default_rng(seed)
        aucs = []
        idxs = np.arange(len(y))
        for _ in range(n_boot):
            b = rng.choice(idxs, size=len(y), replace=True)
            try:
                aucs.append(roc_auc_score(y[b], scores[b]))
            except Exception:
                pass
        if len(aucs) == 0:
            return float(auc), float("nan"), float("nan")
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        return float(auc), float(lo), float(hi)

# --------------------- helpers --------------------- #
LABEL_CANDIDATES = ("label", "y", "target", "outcome", "NPB300")

def find_label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"Label column not found; expected one of {LABEL_CANDIDATES}")

def to_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    if hasattr(model, "decision_function"):
        z = model.decision_function(X).astype(float)
        return 1.0 / (1.0 + np.exp(-z))
    # last resort (not recommended): treat predict as probability
    p = model.predict(X).astype(float)
    # best effort to force [0,1]
    p = np.clip(p, 0.0, 1.0)
    return p

def metrics_at_threshold(y_true, y_prob, thr=0.50):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = tp + tn + fp + fn

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) else float("nan")
    npv  = tn / (tn + fn) if (tn + fn) else float("nan")
    acc  = (tp + tn) / n if n else float("nan")

    sens_lo, sens_hi = wilson_ci(tp, tp + fn)
    spec_lo, spec_hi = wilson_ci(tn, tn + fp)
    ppv_lo,  ppv_hi  = wilson_ci(tp, tp + fp) if (tp + fp) else (float("nan"), float("nan"))
    npv_lo,  npv_hi  = wilson_ci(tn, tn + fn) if (tn + fn) else (float("nan"), float("nan"))
    acc_lo,  acc_hi  = wilson_ci(tp + tn, n)

    auc, auc_lo, auc_hi = delong_roc_ci(y_true, y_prob)

    prev = (y_true == 1).mean() if n else float("nan")

    return {
        "N": n,
        "prevalence": prev,
        "AUROC": auc, "AUROC_lo": auc_lo, "AUROC_hi": auc_hi,
        "sensitivity": sens, "sens_lo": sens_lo, "sens_hi": sens_hi,
        "specificity": spec, "spec_lo": spec_lo, "spec_hi": spec_hi,
        "PPV": ppv, "PPV_lo": ppv_lo, "PPV_hi": ppv_hi,
        "NPV": npv, "NPV_lo": npv_lo, "NPV_hi": npv_hi,
        "accuracy": acc, "acc_lo": acc_lo, "acc_hi": acc_hi,
        "threshold": thr,
    }

def evaluate_csv(path_csv: Path, model, scaler, cohort_name: str) -> dict:
    df = pd.read_csv(path_csv)
    label_col = find_label_column(df)
    y = df[label_col].astype(int).to_numpy()

    # Feature columns: prefer model.feature_names_in_ if present; else use all except label
    if hasattr(model, "feature_names_in_"):
        feat_cols = list(model.feature_names_in_)
    else:
        feat_cols = [c for c in df.columns if c != label_col]

    X = df[feat_cols]
    if scaler is not None:
        X = scaler.transform(X)
    # ensure DataFrame for downstream sklearn compatibility if needed
    X = np.asarray(X)
    p = to_proba(model, X)
    m = metrics_at_threshold(y, p, thr=0.50)
    m["cohort"] = cohort_name
    return m

# --------------------- main --------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model .pkl")
    ap.add_argument("--scaler", required=False, default=None, help="Path to scaler .pkl")
    ap.add_argument("--train", required=False, help="Path to train CSV")
    ap.add_argument("--test", required=False, help="Path to test CSV")
    ap.add_argument("--external", required=False, help="Path to external CSV")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    model = joblib_load(args.model)
    scaler = joblib_load(args.scaler) if args.scaler else None

    rows = []
    if args.train:
        rows.append(evaluate_csv(Path(args.train), model, scaler, "Train"))
    if args.test:
        rows.append(evaluate_csv(Path(args.test), model, scaler, "Test"))
    if args.external:
        rows.append(evaluate_csv(Path(args.external), model, scaler, "External"))

    if not rows:
        print("No datasets provided. Use --train/--test/--external.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows, columns=[
        "cohort","N","prevalence",
        "AUROC","AUROC_lo","AUROC_hi",
        "sensitivity","sens_lo","sens_hi",
        "specificity","spec_lo","spec_hi",
        "PPV","PPV_lo","PPV_hi",
        "NPV","NPV_lo","NPV_hi",
        "accuracy","acc_lo","acc_hi",
        "threshold"
    ])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Pretty print
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(df)

if __name__ == "__main__":
    main()
