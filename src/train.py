# src/train.py
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

# ---------------- PATH FIX ----------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------

from src.data.dataset import Tox21Dataset, TOX21_TASKS
from src.models.model_registry import get_model_class

import argparse
import random
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score


# ===========================
#       REPRODUCIBILITY
# ===========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
#    MASKED BINARY LOSS
# ===========================
def masked_bce_loss(logits, labels):
    if labels.dim() == 1:
        labels = labels.view(logits.size(0), -1)

    if labels.size(1) != logits.size(1):
        labels = labels[:, :logits.size(1)]

    mask = ~torch.isnan(labels)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    return F.binary_cross_entropy_with_logits(logits[mask], labels[mask])


# ===========================
#       VALIDATION METRIC
# ===========================
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = torch.sigmoid(model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.edge_attr
            ))

            y = batch.y
            if y.dim() == 1:
                y = y.view(out.size(0), -1)

            preds.append(out.cpu())
            labs.append(y.cpu())

    preds = torch.cat(preds, dim=0).cpu().tolist()
    labs = torch.cat(labs, dim=0).cpu().tolist()

    preds = np.array(preds, dtype=float)
    labs = np.array(labs, dtype=float)

    aucs = []
    for i in range(len(TOX21_TASKS)):
        mask = ~np.isnan(labs[:, i])
        if mask.sum() < 5:
            aucs.append(np.nan)
            continue
        try:
            a = roc_auc_score(labs[mask, i], preds[mask, i])
        except Exception:
            a = np.nan
        aucs.append(a)

    return float(np.nanmean(aucs))


# ===========================
#    DEGREE HISTOGRAM (PNA)
# ===========================
def compute_degree_stats(dataset):
    """
    Compute degree histogram over dataset nodes (used by PNAConv).
    Returns a 1D torch.LongTensor where index = degree and value = count.
    """
    from torch_geometric.utils import degree

    deg_list = []
    for data in dataset:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            continue
        # degree of source nodes
        d = degree(data.edge_index[0], num_nodes=data.x.size(0))
        # convert to long indices
        deg_list.append(d.to(torch.long))

    if not deg_list:
        return torch.tensor([0], dtype=torch.long)

    deg_cat = torch.cat(deg_list, dim=0)
    max_deg = int(deg_cat.max().item())
    hist = torch.bincount(deg_cat, minlength=max_deg + 1)
    return hist


# ===========================
#       TRAINING LOOP
# ===========================
def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[DEVICE]", device)

    # ===========================
    #   MODEL SELECTION MENU
    # ===========================
    if args.model is None:
        print("""
===========================================
      MODEL SELECTION MENU (Option B)
===========================================
1. GNN  - Fast, high accuracy (recommended)
2. GAT  - Very fast, lower accuracy
3. GCN  - Fast, stable, simple
4. GIN  - Strong accuracy, slower
5. PNA  - Best accuracy, slowest
""")
        choice = input("Enter choice [1-5]: ").strip()
        menu_map = {
            "1": "GNN",
            "2": "GAT",
            "3": "GCN",
            "4": "GIN",
            "5": "PNA"
        }
        model_name = menu_map.get(choice, "GNN")
    else:
        model_name = args.model.upper()

    print(f"\n[MODEL] Using: {model_name}\n")

    # Load model class
    ModelClass = get_model_class(model_name)

    # Dataset
    data_dir = os.path.join(ROOT, "data")
    dataset = Tox21Dataset(data_dir, reprocess=args.reprocess)

    # Split dataset
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    # Input sizes
    sample = dataset[0]
    in_node = sample.x.size(1)
    in_edge = 0 if sample.edge_attr is None else sample.edge_attr.size(1)

    wants_edge = "in_edge" in ModelClass.__init__.__code__.co_varnames

    # ===========================
    #    PNA requires 'deg' - compute if needed
    # ===========================
    deg = None
    if model_name == "PNA":
        print("[PNA] Computing degree histogram (this may take a few seconds)...")
        deg = compute_degree_stats(dataset)
        print(f"[PNA] deg histogram length {len(deg)}; sample: {deg.tolist()[:10]} (truncated)")

    # Construct model (pass deg for PNA; ignored by others)
    try:
        model = ModelClass(
            in_node=in_node,
            in_edge=in_edge if wants_edge else 0,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout,
            deg=deg
        ).to(device)
    except TypeError:
        # fallback if ModelClass doesn't accept deg kwarg
        model = ModelClass(
            in_node=in_node,
            in_edge=in_edge if wants_edge else 0,
            hidden=args.hidden,
            num_layers=args.num_layers,
            dropout=args.dropout
        ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Scheduler: ReduceLROnPlateau (mode='max' because we monitor AUC)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=False,
        min_lr=1e-8
    )

    ckpt_dir = os.path.join(ROOT, "checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_path = os.path.join(ckpt_dir, "latest.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")
    history_path = os.path.join(ckpt_dir, "history.json")

    # Resume bookkeeping
    start_epoch = 1
    best_auc = -np.inf  # use -inf so any real AUC will be larger
    epochs_since_best = 0
    lr_reductions = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    print("\n===========================================")
    print(f"      CHECKPOINT SUMMARY FOR {model_name}")
    print("===========================================\n")

    # AUTO-RESUME: load latest if exists (model-specific)
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model"])
            print(f"✓ Found LATEST checkpoint (next epoch: {ckpt.get('epoch', 'N/A')})")
        except Exception:
            print("⚠ LATEST checkpoint incompatible — ignored")
        try:
            optimizer.load_state_dict(ckpt.get("optimizer", optimizer.state_dict()))
        except Exception:
            # optimizer state might be incompatible across PyTorch versions; ignore safely
            pass
        start_epoch = ckpt.get("epoch", 1)
        best_auc = ckpt.get("best_auc", best_auc)
        print(f"✓ Loaded bookkeeping: best_auc={best_auc:.4f}, start_epoch={start_epoch}")

    # show best.pt info if exists
    if os.path.exists(best_path):
        ckpt_best = torch.load(best_path, map_location="cpu")
        print(f"✓ Found BEST checkpoint (AUC: {ckpt_best.get('best_auc', -1):.4f})")

    if not os.path.exists(latest_path):
        print(f"No checkpoints found for {model_name} — starting from scratch.")

    print("\n")  # spacing

    # Load history
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
    else:
        history = {"epochs": [], "train_loss": [], "val_loss": [], "mean_val_auc": []}

    # ===========================
    #   Training schedule (train for args.epochs more)
    # ===========================
    end_epoch = start_epoch + int(args.epochs)

    # Early stopping + combined logic:
    epochs_since_best = 0 if best_auc != -np.inf else 0
    lr_reductions = 0
    prev_lr = optimizer.param_groups[0]["lr"]

    print(f"[TRAINING] Will run epochs {start_epoch}..{end_epoch - 1} (total {end_epoch - start_epoch})\n")

    for epoch in range(start_epoch, end_epoch):
        model.train()
        losses = []

        for batch in train_loader:
            batch = batch.to(device)

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                batch.edge_attr
            )

            y = batch.y
            if y.dim() == 1:
                y = y.view(logits.size(0), -1)

            loss = masked_bce_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses)) if losses else 0.0
        val_auc = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch}]  TrainLoss={train_loss:.4f}  ValAUC={val_auc:.4f}  LR={optimizer.param_groups[0]['lr']:.3e}")

        # Scheduler step (ReduceLROnPlateau monitors val_auc)
        scheduler.step(val_auc)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr - 1e-12:
            lr_reductions += 1
            print(f"[SCHEDULER] LR reduced: {prev_lr:.6e} -> {new_lr:.6e}  (reductions={lr_reductions})")
            prev_lr = new_lr

        # Early-stopping bookkeeping
        improved = False
        if best_auc is None or np.isnan(best_auc) or val_auc > best_auc + 1e-12:
            best_auc = float(val_auc)
            epochs_since_best = 0
            improved = True
            print(f"[METRIC] New best AUC: {best_auc:.4f}")
        else:
            epochs_since_best += 1

        # Save checkpoint (state includes meta)
        state = {
            "epoch": epoch + 1,  # next epoch index (resume from this)
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_auc": best_auc,
            "model_name": model_name,
            "meta": {
                "hidden": args.hidden,
                "num_layers": args.num_layers,
                "dropout": args.dropout
            }
        }

        # Add deg to checkpoint for PNA so predict can reconstruct model
        if model_name == "PNA" and deg is not None:
            # Save as plain list so checkpoint is JSON-friendly if someone inspects meta
            try:
                state["deg"] = deg.tolist()
            except Exception:
                # fallback: convert to CPU first
                state["deg"] = deg.cpu().tolist()

        torch.save(state, latest_path)

        # Save best
        if improved:
            torch.save(state, best_path)
            print("→ New BEST model saved!")

        # Update history
        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(0.0)
        history["mean_val_auc"].append(val_auc)
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping CONDITION (Option 2 - combined)
        if epochs_since_best >= args.patience and lr_reductions >= args.lr_reductions_stop:
            print(f"\n[EARLY STOP] No improvement for {epochs_since_best} epochs and LR reduced {lr_reductions} times -> stopping early.")
            break

    print("\n[TRAINING COMPLETED]\n")


# ===========================
#            MAIN / ARGS
# ===========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to run in this invocation (training will continue from checkpoint epoch).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_factor", type=float, default=0.5,
                        help="Factor to multiply LR by on plateau (ReduceLROnPlateau).")
    parser.add_argument("--lr_patience", type=int, default=5,
                        help="Number of validation checks with no improvement before LR is reduced (scheduler patience).")
    parser.add_argument("--lr_reductions_stop", type=int, default=2,
                        help="Number of LR reductions required (combined with patience) to trigger early stop.")
    parser.add_argument("--patience", type=int, default=15,
                        help="Number of epochs with no improvement to consider for early stopping (combined logic).")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", type=str, default=None,
                        help="Optional: model to use (GNN, GAT, GCN, GIN, PNA). If omitted, a menu will appear.")
    parser.add_argument("--fresh", action="store_true",
                        help="If set, ignore and overwrite any existing checkpoints for the selected model (start truly fresh).")

    args = parser.parse_args()

    # If --fresh is provided, remove the model-specific checkpoint directory to truly start fresh
    if args.fresh and args.model is not None:
        ckpt_dir = os.path.join(ROOT, "checkpoints", args.model.upper())
        if os.path.exists(ckpt_dir):
            import shutil
            shutil.rmtree(ckpt_dir)
            print(f"[FRESH] Removed existing checkpoints at {ckpt_dir}")

    main(args)
