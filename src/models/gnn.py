# src/models/gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class GNN(nn.Module):
    def __init__(self, in_node, in_edge, hidden=256, num_layers=4, dropout=0.15):
        super().__init__()

        self.node_emb = nn.Linear(in_node, hidden)
        self.edge_emb = nn.Linear(in_edge, hidden) if in_edge > 0 else None

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.dropout = nn.Dropout(dropout)
        self.readout = global_add_pool

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 12),
        )

    def forward(self, x, edge_index, batch, edge_attr):
        x = self.node_emb(x)

        # ensure edge_attr is valid
        if self.edge_emb is not None and edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_emb(edge_attr)
        else:
            edge_attr = None

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        g = self.readout(x, batch)
        return self.head(g)
