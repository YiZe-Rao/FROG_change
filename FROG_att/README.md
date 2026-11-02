# 🐸 FROG: Full-Resolution and Optimizable Graph Structure Learning Framework for RDL

Official implementation of the paper:

> **[ICML 2026]**
> *Is Fixing Schema Graphs Necessary? Full-Resolution Graph Structure Learning for Relational Deep Learning*

---

## 📖 Overview

**FROG** is a **Full-Resolution and Optimizable Graph Structure Learning Framework** for **Relational Deep Learning (RDL)**.

Instead of relying on fixed schema graphs, FROG enables **full-resolution graph structure learning** and jointly optimizes relational graph construction with downstream GNN training.

---

## 🚀 Installation

FROG uses the same environment as **RelBench**.

Install dependencies with:

```bash
pip install relbench[full]
```

---

## ⚡ Quick Start

### 1. Entity Classification / Regression

Run entity classification or regression tasks:

```bash
python -u gnn_entity.py \
    --dataset {DATASET} \
    --task {TASK} \
    --load_para
```

### 2. Entity Recommendation

Run recommendation tasks:

```bash
python -u gnn_recommendation.py \
    --dataset {DATASET} \
    --task {TASK} \
    --load_para
```

---

## 📂 Project Structure

```text
FROG/
├── log/                           # Training logs
│   └── rel-comp.py               # Statistical analysis of Table-as-Node/Edge behaviors
├── exp_model.py                  # REG structure + GNN optimization framework
├── gnn_entity.py                 # Entity classification & regression training
├── gnn_recommendation.py         # Entity recommendation training
├── graph.py                      # Graph construction utilities
├── hop_mode.py                   # Co-occurrence & completion implementation
├── load_train_args.py            # Hyperparameter configuration
├── model.py                      # Core FROG model
├── text_embedder.py              # Text embedding module
├── utils.py                      # Utility functions
└── README.md
```

---

## 🧪 Supported Tasks

* **Entity Classification**
* **Entity Regression**
* **Entity Recommendation**

---

## 🔗 Resources

* Paper: https://arxiv.org/pdf/2605.21475
* Code: https://github.com/RingBDStack/FROG
