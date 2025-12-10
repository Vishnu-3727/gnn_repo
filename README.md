⚛️ GNN Toxicity Predictor

A Multi-Model Graph Neural Network System for Molecular Toxicity Prediction (Tox21)

<p align="center"> <img src="banner.svg" width="100%"> </p>
📘 Overview

GNN Toxicity Predictor is a complete end-to-end machine learning system for predicting 12 toxicity endpoints from molecular structures using Graph Neural Networks (GNNs).
It is built on:

PyTorch

PyTorch Geometric

RDKit

Tox21 dataset

This project supports five state-of-the-art GNN architectures:

GNN (recommended)

GCN

GAT

GIN

PNA (experimental — needs degree calculation)

The system includes:

✔ Automatic checkpointing
✔ Resume training seamlessly
✔ Fresh training mode
✔ Model selection menu
✔ NumPy-free compatible prediction
✔ Automatic model detection for prediction & evaluation
✔ Full plotting utilities
✔ Detailed evaluation (per-task AUC & mean AUC)

🧬 Supported Toxicity Endpoints (Tox21)
Category	Tasks
Nuclear Receptor Signaling	NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma
Stress Response	SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
🚀 Quick Start
1️⃣ Install Dependencies

Using conda (recommended):

conda env create -f environment.yml
conda activate chem


Or using pip:

pip install -r requirements.txt


Dataset is already included (data/tox21.csv).

🏋️ Training Models
▶ Start training
python -m src.train --epochs 50


You will see an interactive model menu:

1. GNN  - Fast, high accuracy (recommended)
2. GAT  - Very fast, lower accuracy
3. GCN  - Fast, stable, simple
4. GIN  - Strong accuracy, slower
5. PNA  - Best accuracy, slowest

✔ Features During Training

Auto-resume from:

checkpoints/<MODEL>/latest.pt


Best model saved to:

checkpoints/<MODEL>/best.pt


Training history stored in JSON

Early stopping (patience 15)

LR scheduler (ReduceLROnPlateau)

♻️ Fresh Training (start from scratch)
python -m src.train --model GNN --fresh --epochs 50


Only deletes checkpoints of the selected model.

🎛️ Useful Training Flags
Flag	Description
--epochs N	Number of epochs
--model NAME	GNN, GCN, GAT, GIN, PNA
--fresh	Delete checkpoints and retrain
--reprocess	Reprocess Tox21 dataset
--batch_size	Batch size
--seed	Training reproducibility
🔮 Predict Toxicity From SMILES
Basic prediction
python -m src.predict --smiles "CCO"


Auto-detects the newest trained model.

Predict with a specific model
python -m src.predict --smiles "CCO" --model GCN

Example Output
=== Toxicity Predictions ===
(Model used: GNN)

NR-AR: 0.1389
NR-ER: 0.2419
...
SR-p53: 0.1209

📊 Evaluation

Run AUC evaluation on full Tox21 dataset:

python -m src.eval --model GNN


Auto-detect latest model:

python -m src.eval


Outputs:

Per-task AUC

Mean AUC

Saves JSON to:

checkpoints/<MODEL>/eval_results.json

📈 Plot Training Metrics
python -m src.plot_metrics --model GCN


Plots saved in:

checkpoints/<MODEL>/plots/

📁 Project Structure

````
GNN Toxicity Predictor
│
├── src/
│ ├── train.py # Training engine
│ ├── predict.py # SMILES → Prediction
│ ├── eval.py # AUC evaluation
│ ├── plot_metrics.py # Loss/AUC plotting
│ │
│ ├── data/
│ │ └── dataset.py # Tox21 loader
│ │
│ ├── models/
│ │ ├── gnn.py
│ │ ├── gcn.py
│ │ ├── gat.py
│ │ ├── gin.py
│ │ ├── pna.py
│ │ └── model_registry.py
│ │
│ └── utils/
│
├── checkpoints/
│ └── <MODEL>/
│ ├── latest.pt
│ ├── best.pt
│ ├── history.json
│ └── plots/
│
├── data/
│ └── tox21.csv
│
├── environment.yml
├── requirements.txt
├── banner.svg
└── README.md
````

🧪 Model Performance

Latest results (example):

Model	Mean AUC
GNN	0.69
GCN	0.66
GAT	0.64
GIN	0.67
PNA	(not supported in eval)
🙋 FAQ
Q: I see “Numpy is not available” warnings. Is this a problem?

No — your system works perfectly without NumPy, and the project includes full NumPy-free fallback logic.

Q: Can I train multiple models?

Yes — each model has its own folder under checkpoints/.


📜 License

This project is released under the MIT License.

⭐ If you like this project, please star the repo!
