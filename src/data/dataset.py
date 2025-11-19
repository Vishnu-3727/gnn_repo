import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem

# --- EXACT 12 TOX21 TASKS ---
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]


def smiles_to_graph(smiles, labels):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # node features = atom number only (simple)
    x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()],
                     dtype=torch.float)

    # edges
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        edge_attr += [[bond.GetBondTypeAsDouble()],
                      [bond.GetBondTypeAsDouble()]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    y = torch.tensor(labels, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


class Tox21Dataset(InMemoryDataset):
    def __init__(self, root, reprocess=False, transform=None, pre_transform=None):
        self.root = root
        super().__init__(root, transform, pre_transform)

        if reprocess or not os.path.exists(self.processed_paths[0]):
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return "tox21.csv"

    @property
    def processed_file_names(self):
        return "tox21.pt"

    def process(self):
        csv_path = os.path.join(self.root, "tox21.csv")
        df = pd.read_csv(csv_path)

        # force correct label selection
        labels = df[TOX21_TASKS].astype(float).values

        data_list = []
        for i, row in df.iterrows():
            g = smiles_to_graph(row["smiles"], labels[i])
            if g is not None:
                data_list.append(g)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
