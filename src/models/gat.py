# src/models/gat.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool


class GATModel(nn.Module):
    def __init__(self, in_node, hidden=128, heads=4, num_layers=3, dropout=0.2, out_dim=12):
        super().__init__()

        self.node_emb = nn.Linear(in_node, hidden)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GATConv(hidden, hidden, heads=heads, concat=False))

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(num_layers)])

        self.dropout = nn.Dropout(dropout)
        self.readout = global_add_pool

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.node_emb(x)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        g = self.readout(x, batch)
        return self.head(g)
