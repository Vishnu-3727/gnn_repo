# src/plot_metrics.py
"""
Loads checkpoints/<MODEL>/history.json and plots:
 - train_loss over epochs
 - mean_val_auc over epochs

Saves PNGs into checkpoints/<MODEL>/plots/
Usage:
    python -m src.plot_metrics --model GNN
"""
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import numpy as np

def load_history(model_name):
    path = os.path.join(ROOT, "checkpoints", model_name, "history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"history.json not found for model {model_name} at {path}. Run training first.")
    with open(path, "r") as f:
        history = json.load(f)
    return history

def plot_history(model_name, out_dir=None):
    history = load_history(model_name)
    epochs = history.get("epochs", list(range(1, len(history.get("train_loss", [])) + 1)))
    train_loss = history.get("train_loss", [])
    mean_val_auc = history.get("mean_val_auc", [])

    if out_dir is None:
        out_dir = os.path.join(ROOT, "checkpoints", model_name, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Plot train loss
    plt.figure()
    plt.plot(epochs, train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"{model_name} - Train Loss")
    plt.grid(True)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "train_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Plot mean AUC
    plt.figure()
    plt.plot(epochs, mean_val_auc)
    plt.xlabel("Epoch")
    plt.ylabel("Mean Val AUC")
    plt.title(f"{model_name} - Mean Val AUC")
    plt.grid(True)
    plt.tight_layout()
    auc_path = os.path.join(out_dir, "mean_val_auc.png")
    plt.savefig(auc_path)
    plt.close()

    print(f"[PLOTS] Saved: {loss_path}")
    print(f"[PLOTS] Saved: {auc_path}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="Model folder name (GNN, GAT, ...)")
    args = p.parse_args()
    plot_history(args.model.upper())
