# src/models/pna.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_add_pool
from torch_geometric.nn.norm import BatchNorm

class PNAModel(nn.Module):
    def __init__(
        self,
        in_node: int,
        in_edge: int = 1,
        hidden: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
        out_dim: int = 12
    ):
        super().__init__()

        # Node embedding
        self.node_emb = nn.Linear(in_node, hidden)

        # Edge embedding
        self.edge_emb = nn.Linear(in_edge, hidden)

        # Aggregators & scalers (best practice from PNA paper)
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            conv = PNAConv(
                in_channels=hidden,
                out_channels=hidden,
                aggregators=aggregators,
                scalers=scalers,
                deg=torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
                edge_dim=hidden
            )
            self.convs.append(conv)
            self.bns.append(BatchNorm(hidden))

        self.dropout = nn.Dropout(dropout)
        self.readout = global_add_pool

        # Final prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim)
        )

    def forward(self, x, edge_index, batch, edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        g = self.readout(x, batch)
        return self.head(g)
