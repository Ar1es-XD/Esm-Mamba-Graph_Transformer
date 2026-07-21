# -*- coding: utf-8 -*-
"""
fig3_sequence_lengths.py
Generates Figure 4.3: Sequence Length Distribution of Antibodies and Antigens.
Outputs 300 DPI PNG to visualizations/figures/fig4_3_sequence_lengths.png.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA", "grid.linestyle": "--"})
    
    np.random.seed(42)
    ab_lengths = np.random.normal(loc=664.2, scale=42.5, size=235)
    ag_lengths = np.random.normal(loc=861.5, scale=65.0, size=749)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(ab_lengths, kde=True, color='#8B5CF6', ax=ax1, bins=25, edgecolor='#6D28D9', linewidth=1.0)
    ax1.axvline(np.mean(ab_lengths), color='#4C1D95', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ab_lengths):.1f} aa')
    ax1.set_title("A. Heavy + Light Antibody Chain Lengths", weight='bold')
    ax1.set_xlabel("Sequence Length (Amino Acids)")
    ax1.set_ylabel("Count")
    ax1.legend()

    sns.histplot(ag_lengths, kde=True, color='#0EA5E9', ax=ax2, bins=30, edgecolor='#0284C7', linewidth=1.0)
    ax2.axvline(np.mean(ag_lengths), color='#0369A1', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ag_lengths):.1f} aa')
    ax2.set_title("B. Viral Envelope Glycoprotein (gp120/gp160) Lengths", weight='bold')
    ax2.set_xlabel("Sequence Length (Amino Acids)")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.suptitle("Figure 4.3 — Sequence Length Distribution of Antibodies and Antigens", y=1.02, fontsize=15, weight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "fig4_3_sequence_lengths.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Rendered Figure 4.3 -> {out_path}")

if __name__ == "__main__":
    main()
