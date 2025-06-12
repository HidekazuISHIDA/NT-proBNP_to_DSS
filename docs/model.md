# `model.pkl` — Pre-trained Decision-Tree Classifier  

This file contains the final **scikit-learn `DecisionTreeClassifier`** described in the NEJM AI manuscript  
“Development of an Interpretable Machine-Learning Model for Early Screening of Heart Failure and Its Application to DSS.”

| Detail | Value |
|--------|-------|
| File name | `models/decision_tree.pkl` |
| Size | < 1 MB |
| Framework | scikit-learn 1.6 |
| Python version | 3.9 |
| Input features | 26 routine laboratory variables (see Table 1 in the paper) |
| Target | Binary label: NT-proBNP ≥ 300 pg mL⁻¹ |
| Training set | 15 911 encounters (80 %) |
| Hyper-parameters | Depth & leaf size selected by 10-fold CV |
| Hash (SHA-256) | `e9bd…<fill in>` |

---

## 1  Loading the model

```python
import joblib, pandas as pd
from pathlib import Path

clf = joblib.load(Path("models") / "decision_tree.pkl")
X = pd.read_csv("data/synthetic_test.csv").drop(columns=["NTproBNP_300"])
y_pred = clf.predict(X)

```
---

## 2 Performance summary

| Metric      | Value (hold-out test, *n* = 3 978) |
| ----------- | ---------------------------------- |
| AUROC       | **0.80**                           |
| F1-score    | **0.70**                           |
| Sensitivity | 0.72                               |
| Specificity | 0.75                               |

These match Table 2 / Figure 4 in the manuscript.

---

## 3 Re-training

To reproduce the model from the (de-identified) synthetic training set:

```bash
python train_model.py --input data/synthetic_train.csv --out models/new_tree.pkl
```
The script repeats preprocessing, Lasso feature selection, and cross-validated depth tuning.

---

## 4 Clinical integration
* Use export_dss_rules.py to convert this pickle into a rule table (dss_rules.csv) for import into Abbott Diagnostic Support System (DSS) or AlinIQ CDS.

```bash
python export_dss_rules.py --model models/decision_tree.pkl --out dss_rules.csv
```

## 5 License & disclaimer

The model weights are released under the same MIT License as the code.
Research use only – not cleared for clinical decision-making without institution-specific validation and regulatory approval.


