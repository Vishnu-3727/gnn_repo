# src/predict_batch.py
"""
Batch prediction: read a CSV with a column 'smiles' and write predictions CSV.
Usage:
    python -m src.predict_batch --in data/input_smiles.csv --out results.csv --model GNN
If --model omitted, script will auto-detect most recent best checkpoint across model folders.
"""
import os
import sys
import csv
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from src.predict import predict as predict_single  # re-use predict function (smiles, model_arg=None)

def predict_csv(in_csv, out_csv, model_name=None, batch_size=64):
    df = pd.read_csv(in_csv)
    if 'smiles' not in df.columns:
        raise RuntimeError("Input CSV must have a 'smiles' column")
    results = []
    for idx, row in df.iterrows():
        s = str(row['smiles'])
        try:
            model_used, preds = predict_single(s, model_arg=model_name)
            row_out = {"smiles": s}
            row_out.update(preds)
            results.append(row_out)
        except Exception as e:
            results.append({"smiles": s, **{t: None for t in []}})
            print(f"[ERROR] SMILES {s}: {e}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"[BATCH] Wrote predictions to {out_csv}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_csv", required=True, help="Input CSV file with 'smiles' column")
    p.add_argument("--out", dest="out_csv", required=True, help="Output CSV file path")
    p.add_argument("--model", type=str, default=None, help="Optional model to use")
    args = p.parse_args()
    predict_csv(args.in_csv, args.out_csv, model_name=args.model)
