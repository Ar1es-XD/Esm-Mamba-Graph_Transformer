# -*- coding: utf-8 -*-
"""
fig5_fused_feature_pca.py
Generates Figure 5.1: Dimensionality Reduction (PCA & t-SNE) of Graph Transformer Node & Latent Representations.
"""
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("[VISUALIZATION] Figure 5.1 script ready.")
if __name__ == "__main__":
    main()
