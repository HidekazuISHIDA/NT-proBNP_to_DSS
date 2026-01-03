from __future__ import annotations
import argparse
from ntprobnp_dss.predict import predict_csv

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--schema", default="model/feature_schema.json")
    ap.add_argument("--threshold", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    predict_csv(args.model, args.schema, args.threshold, args.input, args.output)

if __name__ == "__main__":
    main()
