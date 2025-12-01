# makeMoE: Mixture of Experts from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AviSoori1x/makeMoE/blob/main/makeMoE_from_Scratch.ipynb)

A PyTorch implementation of a **Sparse Mixture of Experts (MoE)** Language Model, built from the ground up. 

This repository demonstrates how to implement the core components of MoE architecture—including Top-k Gating, Noisy Gating, and Expert routing—and integrates them into a Transformer decoder-only architecture (GPT style) to train on the Tiny Shakespeare dataset.

## 🧠 Core Concepts Implemented

This project breaks down the MoE architecture into the following steps:

1.  **The Expert Module**: A simple Multi-Layer Perceptron (MLP) that acts as an individual "expert."
2.  **The Router**: A gating mechanism that determines which expert receives which token.
3.  **Top-k Gating**: Logic to select only the top $k$ experts per token to maintain sparsity and computational efficiency.
4.  **Noisy Top-k Gating**: Adding standard normal noise to logits to ensure load balancing across experts (balancing exploration vs. exploitation).
5.  **Sparse MoE Block**: The module that combines the Router and the Experts, performing the weighted sum of expert outputs.
6.  **The Transformer Block**: Integrating Multi-Head Self-Attention with the Sparse MoE block.

## 🛠️ Architecture Details

The model replaces the standard Feed-Forward Network (FFN) found in vanilla Transformers with a **Sparse MoE Layer**.

* **Embedding Dimension (`n_embed`)**: 128
* **Heads (`n_head`)**: 8
* **Experts (`num_experts`)**: 8
* **Active Experts per Token (`top_k`)**: 2
* **Layers (`n_layer`)**: 8
* **Context Length (`block_size`)**: 32

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/makeMoE-from-scratch.git](https://github.com/yourusername/makeMoE-from-scratch.git)
   cd makeMoE-from-scratch
