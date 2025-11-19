# Clean GNN Repo — Graph Neural Networks for Drug Discovery (Tox21)

**Short summary:**  
This repository demonstrates a Graph Neural Network (GNN) workflow for molecular property prediction (Tox21 toxicity tasks). It includes data preprocessing, multiple GNN architectures, training with checkpointing and resume, evaluation, plotting, and batch prediction utilities. The project is designed to be beginner-friendly and production-ready for experimentation.

---

## Table of contents
1. Project overview
2. What we implemented
3. When to use this repo
4. Environment & dependencies
5. Directory layout
6. Getting data (Tox21)
7. Quickstart — commands (full manual)
8. Detailed user manual (step-by-step)
9. Scripts reference
10. Troubleshooting & tips
11. Next improvements & notes
12. License

---

## 1 — Project overview

This project trains Graph Neural Networks to predict molecular properties from SMILES. Key features:

- Multiple architectures (GNN based on GINEConv, and alternatives: GCN, GAT, GIN, PNA)
- Dataset handling (SMILES → PyG `Data`) using RDKit
- Per-model checkpoint directories (so models don't overwrite each other)
- Auto-resume training from latest checkpoint
- Save `meta` in checkpoints so prediction reconstructs model correctly
- Model-aware evaluation and saving of evaluation results
- Plotting of training history and AUC
- Batch prediction (CSV input → CSV output)

Goal: provide a usable, reproducible pipeline you can show on GitHub.

---

## 2 — What we implemented (concrete)

Files of interest:
- `src/train.py` — training script (menu-driven or CLI)
- `src/models/gnn.py` — primary model (GINEConv-based)
- `src/models/model_registry.py` — registry for model classes
- `src/data/dataset.py` — dataset preprocessing (SMILES → graphs)
- `src/predict.py` — single-SMILES prediction
- `src/predict_batch.py` — batch CSV prediction
- `src/evaluate.py` — model-aware evaluation (per-task AUC)
- `src/plot_metrics.py` — plotting script for results
- `checkpoints/<MODEL>/` — per-model checkpoint folders

---

## 3 — Where it can be used

- Academic experiments on toxicity datasets (e.g., Tox21)
- Rapid prototyping of graph models for molecular properties
- Baseline for model comparison (GNN vs GAT vs GIN)
- Small-scale inference pipelines (batch predictions)
- Teaching / demos for GNNs and cheminformatics

**NOT for production drug claims** — results are research-grade only.

---

## 4 — Environment & dependencies

We recommend a Conda environment. Example `environment.yml` (or pip `requirements.txt`):

**Conda (recommended):**
```yaml
name: chem
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pytorch
  - torchvision
  - pytorch-geometric
  - rdkit
  - scikit-learn
  - pandas
  - matplotlib
  - numpy
  - pip
  - pip:
      - torch-scatter
      - torch-sparse
      - torch-cluster
      - torch-spline-conv





## 5 - Directory layout

clean_gnn_repo/
│
├── data/
│   └── tox21.csv              # your CSV (SMILES + label columns)
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── predict_batch.py
│   ├── plot_metrics.py
│   ├── data/
│   │   └── dataset.py
│   └── models/
│       ├── gnn.py
│       ├── model_registry.py
│
├── checkpoints/
│   ├── GNN/
│   ├── GAT/
│   └── ...
└── README.md


## 6 — Getting the data (Tox21)

Place a tox21.csv in data/ with columns:

smiles (SMILES string)

the 12 Tox21 task columns (matching names used in TOX21_TASKS)

If you downloaded a different CSV, ensure column names match or adapt src/data/dataset.py mapping logic.


## 7 — Quickstart — commands (manual)

Run these in a terminal where the Conda env chem is activated and working.

7.1 Train (interactive menu shown if --model omitted)

Train for N epochs (N is number of epochs to run now; training resumes automatically from checkpoint):

python -m src.train --epochs 50


Or train a specific model (skip menu):

python -m src.train --model GAT --epochs 50 --batch_size 64


Start truly fresh (delete model-specific checkpoints first) with --fresh:

python -m src.train --model GNN --epochs 50 --fresh

7.2 Evaluate a model

Evaluate the best checkpoint for a specific model:

python -m src.evaluate --model GNN


Auto-detect the most recent best checkpoint:

python -m src.evaluate

7.3 Plot training history (creates PNGs)
python -m src.plot_metrics --model GNN


Outputs into checkpoints/GNN/plots/.

7.4 Single SMILES prediction
python -m src.predict --smiles "CCO" --model GNN


If --model omitted, src.predict auto-detects the most recent best checkpoint.

7.5 Batch prediction (CSV)

Input CSV must contain smiles column:

python -m src.predict_batch --in data/my_smiles.csv --out results.csv --model GNN


If --model omitted it auto-detects the most recent trained model.

7.6 Check checkpoints

List per-model checkpoints:

ls -la checkpoints/GNN



## 8 — Detailed user manual (step-by-step)

Step A — Prepare environment

Create conda env and install dependencies (see section 4).

Activate env: conda activate chem.

Step B — Put data in place

Place tox21.csv in data/.

Confirm src/data/dataset.py picks the correct label columns (the code tries to match tolerant names).

Step C — First training run

Run python -m src.train --epochs 50

If --model omitted you'll be shown a menu.

Training will create checkpoints/<MODEL>/latest.pt and best.pt.

If run is interrupted, re-run with same command — the script will resume automatically from the checkpoint.

Step D — Evaluate model

Run python -m src.evaluate --model <MODEL> — prints per-task AUC and writes checkpoints/<MODEL>/eval_results.json.

Step E — Plot results

Run python -m src.plot_metrics --model <MODEL> — PNGs saved to checkpoints/<MODEL>/plots/.

Step F — Predict

Single SMILES:

python -m src.predict --smiles "CCO" --model GNN


Batch:

python -m src.predict_batch --in some_smiles.csv --out preds.csv --model GNN

Step G — Clean / fresh start

To remove a model's checkpoints:

python -m src.train --model GNN --fresh


or manually:

rm -rf checkpoints/GNN


## — Scripts reference (short)

src/train.py — training (menu, per-model checkpoints, auto-resume, early stopping + scheduler)

src/evaluate.py — evaluate saved checkpoint

src/plot_metrics.py — plot training history

src/predict.py — single-SMILES prediction

src/predict_batch.py — batch predictions CSV → CSV


## 9 — Scripts reference (short)

src/train.py — training (menu, per-model checkpoints, auto-resume, early stopping + scheduler)

src/evaluate.py — evaluate saved checkpoint

src/plot_metrics.py — plot training history

src/predict.py — single-SMILES prediction

src/predict_batch.py — batch predictions CSV → CSV


## 10 — Troubleshooting & tips

NumPy DLL issue on Windows: you may see a warning Failed to initialize NumPy — ensure your Conda env installed numpy compatible with PyTorch and RDKit. Using Conda-forge solves many headaches.

Model mismatch when resuming: this means you changed model architecture or feature sizes — either train fresh or ensure meta saved in checkpoints matches model constructor params.

OOM on GPU: reduce --batch_size.

RDKit errors parsing SMILES: check your tox21.csv for clean SMILES; RDKit returns None for invalid strings.

Training appears to 'start from scratch': ensure you specified the same --model value, because checkpoints are saved in checkpoints/<MODEL>/.

You changed dataset/features: re-run with --fresh and reprocess.

## 11 — Next improvements (ideas)

TensorBoard logging

Hyperparameter sweep scripts

Unit tests for preprocessing

Docker container + CI for reproducibility

Small web UI for batch uploads




## 12 — License

Use your preferred license (MIT recommended for research demos).

Final notes & recommended next steps

Run a short experiment and evaluate: python -m src.train --epochs 5 then python -m src.evaluate then python -m src.plot_metrics --model GNN.

Save the repo to GitHub with README.md, requirements.txt or environment.yml.

If you want I can also create requirements.txt, .gitignore, and a short CONTRIBUTING.md.


