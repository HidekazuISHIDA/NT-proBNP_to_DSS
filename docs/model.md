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
