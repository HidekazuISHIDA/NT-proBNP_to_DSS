# NT-proBNP to DSS (Rebuilt)

This repository contains a fully reproducible pipeline to develop, validate, and deploy a **decision-tree–based triage / risk-stratification DSS** that predicts **NT-proBNP > 300 pg/mL** from routine laboratory tests.

## What changed in this rebuilt repo
This repo is rebuilt to match the revised manuscript / response-to-reviewers:
- **No LASSO / no pre-feature selection**: the decision tree is trained using *all candidate predictors*.
- **Threshold optimized for sensitivity**: we choose a probability threshold that achieves **training sensitivity ≥ 0.90** (configurable).
- Clear separation of:
  - `src/` (library code),
  - `scripts/` (CLI wrappers),
  - `configs/` (single source of truth),
  - `model/` (exported artifacts),
  - `docs/` (figures and documentation).

> ⚠️ Data are NOT included. Place your internal/external CSVs under `data/` (see below).

---

## Repository structure

```
NT-proBNP_to_DSS_rebuilt/
  configs/                 # model + data configuration (single source of truth)
  data/                    # put your CSVs here (NOT tracked)
  docs/                    # manuscript figures / DSS documentation
  manuscript/              # response letter + revision notebook (optional)
  model/                   # trained artifacts (tree, threshold, metadata)
  scripts/                 # CLI entrypoints
  src/ntprobnp_dss/        # python package
  tests/                   # minimal tests
```

---

## Quickstart

### 1) Create environment
Python 3.9+ recommended.

```bash
pip install -r requirements.txt
```

### 2) Put data files
- Internal development dataset (CSV): `data/internal.csv`
- External validation dataset (CSV): `data/external.csv`

Both files must contain:
- A column named `NTproBNP` (numeric)
- Predictor columns (routine labs, demographics) with consistent column names between datasets

### 3) Edit configuration (recommended)
Open `configs/model_config.yaml` and set:
- `predictor_columns`: either an explicit list (recommended) or leave blank to auto-infer.
- `decision_tree.max_depth`, etc.
- `thresholding.min_training_sensitivity` (default: 0.90)

### 4) Train, validate, and export model artifacts
```bash
python scripts/train_and_export.py --config configs/model_config.yaml
```

Outputs:
- `model/model.joblib` (sklearn decision tree)
- `model/threshold.json` (probability threshold for DSS)
- `model/feature_schema.json` (columns + dtypes)
- `docs/figures/decision_tree_top3.png` (top 3 levels)
- `docs/figures/shap_summary.png` (TreeExplainer SHAP)

### 5) Run predictions (DSS behavior)
```bash
python scripts/predict.py --model model/model.joblib --threshold model/threshold.json --input data/external.csv --output outputs/predictions.csv
```

---

## Notes on column handling (important)
To prevent silent bugs in DSS deployment:
- By default, we **fail fast** if input columns do not match the trained feature schema.
- You can enable a *strict* column list in the config.

---

## Citation
If you publish with this code, please add your preferred citation in `CITATION.cff`.

## License
See `LICENSE`.
