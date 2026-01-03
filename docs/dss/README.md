# DSS integration notes

## Expected inputs
- Must match `model/feature_schema.json` columns (names + order).

## DSS output rule
- Use probability threshold stored in `model/threshold.json`.
- `pred_prob >= threshold` => positive (high risk for NT-proBNP > 300 pg/mL).

## Rule export
- `model/rules/tree_rules.json` contains the full tree structure for LIS / DSS embedding.
