# How to Export Decision-Tree Rules to DSS/CDS

This note explains how to convert the trained decision tree into a rule artifact consumable by Abbott’s Diagnostic Support System (DSS) or other middleware. Always validate locally before deployment.

## 1) Generate the rule file

```bash
python scripts/export_dss_rules.py \
  --model model/model.pkl \
  --output dss_rules.json
```

The output is a JSON rule tree. Example (abridged):

```json
{
  "version": "1.0.0",
  "threshold": 0.50,
  "nodes": [
    {
      "feature": "eGFR",
      "op": "<",
      "value": 60.0,
      "left": {"feature": "Alb", "op": "<", "value": 4.0, "...": "..."},
      "right": {"feature": "UN", "op": ">", "value": 20.0, "...": "..."}
    }
  ],
  "leaf_payload": {
    "pred_prob": true,
    "pred_label": true,
    "comment_template": "Predicted risk of NT-proBNP > 300 pg/mL is elevated based on routine laboratory parameters. Consider NT-proBNP testing and clinical evaluation for heart failure."
  }
}
```

## 2) Operating policy

Fixed decision rule: threshold = 0.50 (pre-specified).

Blinding: the index test does not use NT-proBNP.

Outputs: (i) binary label; (ii) predicted probability; (iii) interpretive comment.


## 3) Governance & safety

Versioned rule bundles and audit logs are recommended.

Validate turnaround time impact (should be negligible).

Monitor calibration and hit rates; review quarterly.

Site-specific validation is mandatory before clinical use.

```pgsql

---

### `data/README.md`

```md
# Synthetic Dataset

This repository includes `data/sample_data.csv`, a small synthetic dataset with the same schema as the study data. It is intended to let users run the full pipeline (prediction, evaluation, and rule export) without accessing clinical data.

- **Label column**: `NPB300` (1 if NT-proBNP > 300 pg/mL, else 0)  
- **Predictor columns**: routine laboratory parameters plus age and sex, matching the model’s expected features.  
- **Privacy**: no real patient-level data are included.
```
