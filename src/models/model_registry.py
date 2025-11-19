# src/models/model_registry.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINConv,
    PNAConv,
    global_add_pool
)

# Main GNN model
from src.models.gnn import GNN


# ============================================
#               GCN MODEL
# ============================================
class GCN(nn.Module):
    """Basic GCN  with global add pooling."""
    def __init__(self, in_node, in_edge=None, hidden=256, num_layers=3, dropout=0.15, **kwargs):
        super().__init__()
        layers = []
        last = in_node

        for _ in range(num_layers):
            layers.append(GCNConv(last, hidden))
            last = hidden

        self.convs = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 12)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        g = global_add_pool(x, batch)
        return self.head(g)


# ============================================
#                GAT MODEL
# ============================================
class GAT(nn.Module):
    """Graph Attention Network."""
    def __init__(self, in_node, in_edge=None, hidden=256, num_layers=3, dropout=0.15, heads=4, **kwargs):
        super().__init__()
        layers = []
        last = in_node

        for _ in range(num_layers):
            layers.append(GATConv(last, hidden // heads, heads=heads))
            last = hidden

        self.convs = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 12)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = self.dropout(x)
        g = global_add_pool(x, batch)
        return self.head(g)


# ============================================
#                GIN MODEL
# ============================================
class GIN(nn.Module):
    """GIN (Graph Isomorphism Network)."""
    def __init__(self, in_node, in_edge=None, hidden=256, num_layers=3, dropout=0.15, **kwargs):
        super().__init__()
        layers = []
        last = in_node

        for _ in range(num_layers):
            mlp = nn.Sequential(nn.Linear(last, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            layers.append(GINConv(mlp))
            last = hidden

        self.convs = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 12)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        g = global_add_pool(x, batch)
        return self.head(g)


# ============================================
#                PNA MODEL (FIXED)
# ============================================
class PNA(nn.Module):
    """Principal Neighbourhood Aggregation (requires degree histogram)."""
    def __init__(self, in_node, in_edge, hidden=256, num_layers=4, dropout=0.15, deg=None, **kwargs):
        super().__init__()

        if deg is None:
            raise ValueError("PNA requires 'deg' computed from dataset. Compute it in train.py.")

        self.node_emb = nn.Linear(in_node, hidden)
        self.edge_emb = nn.Linear(in_edge, hidden) if in_edge > 0 else None

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        towers = 1  # simpler + works very well

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            conv = PNAConv(
                in_channels=hidden,
                out_channels=hidden,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=hidden if self.edge_emb else None,
                towers=towers
            )
            self.layers.append(conv)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 12)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.node_emb(x)
        if self.edge_emb:
            edge_attr = self.edge_emb(edge_attr)

        for conv in self.layers:
            x = F.relu(conv(x, edge_index, edge_attr))
            x = self.dropout(x)

        g = global_add_pool(x, batch)
        return self.head(g)


# ============================================
#             MODEL REGISTRY
# ============================================
MODEL_REGISTRY = {
    "GNN": GNN,
    "GCN": GCN,
    "GAT": GAT,
    "GIN": GIN,
    "PNA": PNA,
}

def get_model_class(name: str):
    name = name.upper()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{name}'. Options: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
