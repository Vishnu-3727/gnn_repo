# src/eval.py
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

# PATH fix
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.dataset import Tox21Dataset, TOX21_TASKS
from src.models.model_registry import get_model_class


# -------------------------------------------------------------
#  Find newest best checkpoint among model folders (skip PNA)
# -------------------------------------------------------------
def find_latest_checkpoint(skip_unsupported=True):
    ckpt_root = os.path.join(ROOT, "checkpoints")
    if not os.path.exists(ckpt_root):
        return None

    best_files = []

    for model_dir in os.listdir(ckpt_root):
        full = os.path.join(ckpt_root, model_dir)
        if not os.path.isdir(full):
            continue

        if skip_unsupported and model_dir.upper() == "PNA":
            print(f"[AUTO-DETECT] Skipping unsupported model for eval: {model_dir}")
            continue

        best_path = os.path.join(full, "best.pt")
        if os.path.exists(best_path):
            mtime = os.path.getmtime(best_path)
            best_files.append((model_dir, best_path, mtime))

    if not best_files:
        return None

    best_files.sort(key=lambda x: x[2], reverse=True)
    return best_files[0]  # model_name, path, mtime


# -------------------------------------------------------------
#  Build model from checkpoint
# -------------------------------------------------------------
def build_model_from_ckpt(ckpt, device, model_name):
    ModelClass = get_model_class(model_name.upper())

    if model_name.upper() == "PNA":
        if "meta" not in ckpt or "deg" not in ckpt["meta"]:
            raise ValueError("PNA requires 'deg' metadata, not found in checkpoint.")

    data_dir = os.path.join(ROOT, "data")
    dataset = Tox21Dataset(data_dir, reprocess=False)

    sample = dataset[0]
    in_node = sample.x.size(1)
    in_edge = 0 if sample.edge_attr is None else sample.edge_attr.size(1)

    wants_edge = "in_edge" in ModelClass.__init__.__code__.co_varnames

    meta = ckpt.get("meta", {})
    hidden = meta.get("hidden", 256)
    num_layers = meta.get("num_layers", 4)
    dropout = meta.get("dropout", 0.15)

    model = ModelClass(
        in_node=in_node,
        in_edge=in_edge if wants_edge else 0,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout
    )

    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


# -------------------------------------------------------------
#  Evaluate a checkpoint
# -------------------------------------------------------------
def evaluate_checkpoint(model_name, batch_size=64):
    model_name = model_name.upper()

    ckpt_dir = os.path.join(ROOT, "checkpoints", model_name)
    best_path = os.path.join(ckpt_dir, "best.pt")
    latest_path = os.path.join(ckpt_dir, "latest.pt")

    if os.path.exists(best_path):
        ckpt_path = best_path
    elif os.path.exists(latest_path):
        ckpt_path = latest_path
    else:
        raise FileNotFoundError(f"No checkpoint found for model {model_name}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Using checkpoint: {ckpt_path} (model={model_name})")

    model = build_model_from_ckpt(ckpt, device, model_name)

    # Load dataset
    data_dir = os.path.join(ROOT, "data")
    dataset = Tox21Dataset(data_dir, reprocess=False)
    loader = DataLoader(dataset, batch_size=batch_size)

    preds = []
    labs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = torch.sigmoid(model(
                batch.x, batch.edge_index, batch.batch, batch.edge_attr
            ))

            y = batch.y.float()
            if y.dim() == 1:
                y = y.view(out.size(0), -1)

            preds.append(out.cpu().tolist())
            labs.append(y.cpu().tolist())

    preds = np.array(sum(preds, []), dtype=float)
    labs  = np.array(sum(labs,  []), dtype=float)

    aucs = []
    print("\n=== Per-task AUCs ===")

    for i, task in enumerate(TOX21_TASKS):
        mask = ~np.isnan(labs[:, i])
        if mask.sum() < 5:
            print(f"{task}: NA")
            aucs.append(np.nan)
            continue

        auc = roc_auc_score(labs[mask, i], preds[mask, i])
        aucs.append(auc)
        print(f"{task}: {auc:.4f}")

    mean_auc = float(np.nanmean(aucs))
    print(f"\nMean AUC: {mean_auc:.4f}")

    # Save results
    result = {
        "model": model_name,
        "mean_auc": mean_auc,
        "per_task_auc": {TOX21_TASKS[i]: (None if np.isnan(aucs[i]) else float(aucs[i])) for i in range(len(TOX21_TASKS))}
    }

    out_path = os.path.join(ckpt_dir, "eval_results.json")
    with open(out_path, "w") as f:
        import json
        json.dump(result, f, indent=2)

    print(f"[EVAL] Results saved to {out_path}")


# -------------------------------------------------------------
#  MAIN
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    # auto-detect if not provided
    if args.model is None:
        found = find_latest_checkpoint(skip_unsupported=True)
        if not found:
            print("[EVAL] No checkpoints found for evaluation.")
            return
        model_name, path, _ = found
        print(f"[AUTO-DETECT] Using most recent supported model: {model_name} ({path})")
        args.model = model_name

    evaluate_checkpoint(args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
