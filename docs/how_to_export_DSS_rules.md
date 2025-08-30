# How to Export Decision-Tree Rules to DSS/CDS

This note explains how to convert the trained decision tree into a rule artifact consumable by Abbott’s Diagnostic Support System (DSS) or other middleware. Always validate locally before deployment.

## 1) Generate the rule file

```bash
python scripts/export_dss_rules.py \
  --model model/model.pkl \
  --output dss_rules.json
