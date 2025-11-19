# src/predict.py
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

# PATH fix
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rdkit import Chem
import torch
from torch_geometric.data import Data
from src.data.dataset import Tox21Dataset, TOX21_TASKS
from src.models.model_registry import get_model_class


def find_most_recent_best_ckpt(ckpt_root):
    bests = []
    if not os.path.exists(ckpt_root):
        return None
    for name in sorted(os.listdir(ckpt_root)):
        model_dir = os.path.join(ckpt_root, name)
        if not os.path.isdir(model_dir):
            continue
        best_path = os.path.join(model_dir, "best.pt")
        if os.path.exists(best_path):
            try:
                mtime = os.path.getmtime(best_path)
                bests.append((name, best_path, mtime))
            except Exception:
                continue
    if not bests:
        return None
    bests.sort(key=lambda x: x[2], reverse=True)
    return bests[0]


def smiles_to_graph(smiles, edge_dim=1):
    """
    Convert SMILES -> torch_geometric.data.Data.
    If molecule has zero bonds, return a graph that has:
      - x: node features
      - edge_index: either real edges or a single self-loop (0->0)
      - edge_attr: shape (num_edges, edge_dim)
    edge_dim should match the model/dataset's edge feature width.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.float)

    edge_index_list = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = [bond.GetBondTypeAsDouble()]
        edge_index_list += [[i, j], [j, i]]
        edge_attr_list += [bf, bf]

    if len(edge_index_list) == 0:
        # No bonds: create a small safe self-loop so message-passing layers that expect
        # at least one edge do not crash. This is a practical engineering fix.
        # Create one self-loop on node 0 (single atom case).
        # Ensure edge_attr has correct width (edge_dim).
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        edge_attr = torch.zeros((1, edge_dim), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        # ensure edge_attr columns match edge_dim; if not, pad/truncate safely
        ea = torch.tensor(edge_attr_list, dtype=torch.float)
        if ea.size(1) < edge_dim:
            # pad columns with zeros
            pad = torch.zeros((ea.size(0), edge_dim - ea.size(1)), dtype=torch.float)
            edge_attr = torch.cat([ea, pad], dim=1)
        elif ea.size(1) > edge_dim:
            edge_attr = ea[:, :edge_dim]
        else:
            edge_attr = ea

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def compute_degree_stats_from_dataset(dataset):
    """Compute deg histogram (fallback) using same logic as train.py"""
    from torch_geometric.utils import degree
    deg_list = []
    for data in dataset:
        if data.edge_index is None or data.edge_index.size(1) == 0:
            continue
        d = degree(data.edge_index[0], num_nodes=data.x.size(0))
        deg_list.append(d.to(torch.long))
    if not deg_list:
        return torch.tensor([0], dtype=torch.long)
    deg_cat = torch.cat(deg_list, dim=0)
    max_deg = int(deg_cat.max().item())
    return torch.bincount(deg_cat, minlength=max_deg + 1)


def load_checkpoint_for_model(device, requested_model=None):
    ckpt_root = os.path.join(ROOT, "checkpoints")
    if requested_model:
        model_name = requested_model.upper()
        model_dir = os.path.join(ckpt_root, model_name)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"No checkpoint directory for model '{model_name}' (expected {model_dir})")
        best_path = os.path.join(model_dir, "best.pt")
        if not os.path.exists(best_path):
            latest_path = os.path.join(model_dir, "latest.pt")
            if os.path.exists(latest_path):
                best_path = latest_path
            else:
                raise FileNotFoundError(f"No checkpoint file found for model '{model_name}' in {model_dir}")
        ckpt = torch.load(best_path, map_location=device)
        return model_name, best_path, ckpt

    found = find_most_recent_best_ckpt(ckpt_root)
    if not found:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_root}. Train a model first.")
    model_name, path, _ = found
    ckpt = torch.load(path, map_location=device)
    ckpt_model_name = ckpt.get("model_name")
    if ckpt_model_name:
        model_name = ckpt_model_name.upper()
    return model_name, path, ckpt


def build_model_from_checkpoint(device, ckpt, model_name):
    """
    Returns: (model, in_edge)
    """
    data_dir = os.path.join(ROOT, "data")
    dataset = Tox21Dataset(data_dir, reprocess=False)
    sample = dataset[0]
    in_node = sample.x.size(1)
    in_edge = 0 if sample.edge_attr is None else sample.edge_attr.size(1)

    ModelClass = get_model_class(model_name)
    wants_edge = "in_edge" in ModelClass.__init__.__code__.co_varnames
    meta = ckpt.get("meta", {})

    if model_name == "PNA":
        deg = ckpt.get("deg", None)
        if deg is None:
            # fallback compute so predict doesn't force retrain
            print("[PNA] Warning: 'deg' missing in checkpoint — computing from dataset as fallback.")
            deg = compute_degree_stats_from_dataset(dataset)
        if not torch.is_tensor(deg):
            deg = torch.tensor(deg, dtype=torch.long)
        model = ModelClass(
            in_node=in_node,
            in_edge=in_edge,
            hidden=meta.get("hidden", 256),
            num_layers=meta.get("num_layers", 4),
            dropout=meta.get("dropout", 0.15),
            deg=deg
        )
    else:
        model = ModelClass(
            in_node=in_node,
            in_edge=in_edge if wants_edge else 0,
            hidden=meta.get("hidden", 256),
            num_layers=meta.get("num_layers", 4),
            dropout=meta.get("dropout", 0.15)
        )

    # robust weight loading
    try:
        model.load_state_dict(ckpt["model"])
    except Exception:
        try:
            model.load_state_dict(ckpt["model"], strict=False)
            print("[WARN] Loaded weights with strict=False.")
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

    model.to(device).eval()
    return model, in_edge


def predict(smiles, model_arg=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    model_name, ckpt_path, ckpt = load_checkpoint_for_model(device, requested_model=model_arg)
    print(f"[CHECKPOINT] Using checkpoint for model '{model_name}': {ckpt_path}")

    model, in_edge = build_model_from_checkpoint(device, ckpt, model_name)

    # build graph with edge_dim = in_edge (if 0, default to 1 for safe attr width)
    edge_dim_for_builder = in_edge if in_edge > 0 else 1
    graph = smiles_to_graph(smiles, edge_dim=edge_dim_for_builder)

    # If original molecule had zero bonds, smiles_to_graph created a self-loop.
    # But GIN/PNA are topology-sensitive; reject them explicitly for bondless molecules.
    # Detect original-bond-count by checking whether the returned graph has 1 edge and that edge is a self-loop.
    original_bondless = (graph.edge_index.size(1) == 1 and graph.edge_index[0, 0] == graph.edge_index[1, 0])
    if original_bondless and model_name in ["GIN", "PNA"]:
        raise ValueError(
            f"Model '{model_name}' requires real bonds (non-trivial topology). "
            "For single-atom molecules use GNN, GCN or GAT instead."
        )

    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)

    with torch.no_grad():
        out = torch.sigmoid(model(
            graph.x,
            graph.edge_index,
            graph.batch,
            graph.edge_attr
        )).cpu().tolist()[0]

    return model_name, {TOX21_TASKS[i]: float(out[i]) for i in range(len(TOX21_TASKS))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--model", type=str, default=None,
                        help="Optional: model to use (GNN, GAT, GCN, GIN, PNA). If omitted, auto-detect most recent checkpoint.")
    args = parser.parse_args()

    model_name, preds = predict(args.smiles, model_arg=args.model)
    print("\n=== Toxicity Predictions ===")
    print(f"(Model used: {model_name})\n")
    for k, v in preds.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
