# 🕸️ ESM-Mamba Graph Transformer: PyG Graph Neural Networks for Antibody Prediction

[![Vercel Live Demo](https://img.shields.io/badge/Live_Demo-esm--mamba--graph.vercel.app-f97316?style=for-the-badge&logo=vercel)](https://esm-mamba-graph.vercel.app)
[![Framework](https://img.shields.io/badge/Framework-PyTorch_Geometric_(PyG)-3b82f6?style=for-the-badge&logo=pytorch)](https://github.com/Ar1es-XD/Esm-Mamba-Graph_Transformer)
[![Peak AUROC](https://img.shields.io/badge/Peak_AUROC-0.8625-f59e0b?style=for-the-badge)](https://github.com/Ar1es-XD/Esm-Mamba-Graph_Transformer)
[![License](https://img.shields.io/badge/License-MIT-000000?style=for-the-badge)](LICENSE)

An extension of the **ESM-Mamba architecture using Graph Neural Networks (PyTorch Geometric / PyG)** with dynamic top-k contact edges and Graph Attention Transformers to model 3D spatial contact topology and predict antibody-antigen binding neutralization.

---

## 🌟 Key Features

- **Dynamic Top-K Contact Edges**: Constructs 3D topological contact graphs dynamically using amino acid feature embeddings.
- **Graph Attention Transformers**: Utilizes PyTorch Geometric (`PyG`) Graph Transformer Conv layers to message-pass residue features across contact graph edges.
- **Out-of-Distribution Benchmark**: Tested across 4 partitioning splits including novel viruses and double-holdout splits.
- **Live Vercel Web Predictor**: Includes a zero-latency Vercel Python serverless API and interactive web interface with node-edge attention heatmaps.

---

## 📊 Experimental Benchmark Results

| Experiment | Split Partition | Train $N$ | Test $N$ | Best Epoch | AUROC 📈 | AUPRC 📊 | Accuracy | F1 Score |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp 1** | **Random Split** | 8,970 | 2,240 | 10 | **0.8372** | **0.8857** | 0.4112 | 0.0000 |
| **Exp 2** | **Novel Viruses** | 9,183 | 2,027 | 8 | **0.8625** | **0.9057** | 0.3981 | 0.0065 |
| **Exp 3** | **Novel Antibodies** | 8,686 | 2,524 | 6 | **0.8314** | **0.8862** | 0.4033 | 0.0000 |
| **Exp 4** | **Double Holdout** | 5,216 | 1,096 | 4 | **0.6961** | **0.7364** | 0.4325 | 0.0000 |

---

## 🚀 Live Demo & Setup

### 1. Vercel Web App (Production)
Visit the live Vercel web application:
👉 **[esm-mamba-graph.vercel.app](https://esm-mamba-graph.vercel.app)**

### 2. Local Training & Setup

```bash
git clone https://github.com/Ar1es-XD/Esm-Mamba-Graph_Transformer.git
cd Esm-Mamba-Graph_Transformer

# Install PyG dependencies
pip install -r requirements-train.txt

# Run training
python run_all_experiments.py
```

---

## 📄 Citation & Attribution

Created by **Chinmaya S** ([@Ar1es-XD](https://github.com/Ar1es-XD)).
