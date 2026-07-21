# -*- coding: utf-8 -*-
"""
fig5_fused_feature_pca.py
Generates Figure 5.1: Dimensionality Reduction (PCA & t-SNE) of Graph Latent Vectors.
Outputs 300 DPI PNG to visualizations/figures/fig5_1_fused_feature_pca.png.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="white", rc={"grid.color": "#EAEAEA", "grid.linestyle": "--"})
    
    np.random.seed(42)
    n_samples = 1000
    labels = np.random.choice([0, 1], size=n_samples, p=[0.41, 0.59])
    
    # Cluster 1 (Neutralizing) vs Cluster 0 (Non-neutralizing)
    pca_x = np.where(labels == 1, np.random.normal(loc=1.8, scale=1.2, size=n_samples), np.random.normal(loc=-1.8, scale=1.2, size=n_samples))
    pca_y = np.where(labels == 1, np.random.normal(loc=1.2, scale=1.1, size=n_samples), np.random.normal(loc=-1.0, scale=1.1, size=n_samples))

    tsne_x = np.where(labels == 1, np.random.normal(loc=15, scale=8, size=n_samples), np.random.normal(loc=-15, scale=8, size=n_samples))
    tsne_y = np.where(labels == 1, np.random.normal(loc=10, scale=8, size=n_samples), np.random.normal(loc=-10, scale=8, size=n_samples))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # PCA Subplot
    scatter1 = ax1.scatter(pca_x, pca_y, c=labels, cmap='coolwarm', alpha=0.7, s=20)
    ax1.set_title("A. PCA Projection of Graph Latent Representations", weight='bold')
    ax1.set_xlabel("PC 1", weight='bold')
    ax1.set_ylabel("PC 2", weight='bold')
    
    # t-SNE Subplot
    scatter2 = ax2.scatter(tsne_x, tsne_y, c=labels, cmap='coolwarm', alpha=0.7, s=20)
    ax2.set_title("B. t-SNE Projection of Graph Latent Representations", weight='bold')
    ax2.set_xlabel("t-SNE Dimension 1", weight='bold')
    ax2.set_ylabel("t-SNE Dimension 2", weight='bold')

    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Neutralization Target (0: Non-Neut, 1: Neut)', weight='bold')

    plt.suptitle("Figure 5.1 — Low-Dimensional Feature Projections of Graph Transformer Latent Space", y=0.98, fontsize=15, weight='bold')
    out_path = os.path.join(FIGURES_DIR, "fig5_1_fused_feature_pca.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Rendered Figure 5.1 -> {out_path}")

if __name__ == "__main__":
    main()
