# -*- coding: utf-8 -*-
"""
fig7_generalization_degradation.py
Generates Figure 6.2: Generalization Degradation Curve & Entity Holdout Asymmetry.
Outputs 300 DPI PNG to visualizations/figures/fig6_2_generalization_degradation.png.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA", "grid.linestyle": "--"})
    
    experiments = ['Random Split\n(Interpolation)', 'Novel Viruses\n(Antigen Holdout)', 'Novel Abs\n(Antibody Holdout)', 'Both Novel\n(Double Holdout)']
    auroc = [0.8372, 0.8625, 0.8314, 0.6961]
    auprc = [0.8857, 0.9057, 0.8862, 0.7364]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(experiments, auroc, marker='o', linewidth=2.5, markersize=8, color='#8B5CF6', label='AUROC')
    ax.plot(experiments, auprc, marker='s', linewidth=2.5, markersize=8, color='#10B981', label='AUPRC')

    for i, txt in enumerate(auroc):
        ax.annotate(f'{txt:.4f}', (experiments[i], auroc[i] + 0.012), ha='center', weight='bold', color='#6D28D9')
    for i, txt in enumerate(auprc):
        ax.annotate(f'{txt:.4f}', (experiments[i], auprc[i] + 0.012), ha='center', weight='bold', color='#047857')

    ax.set_title("Figure 6.2 — Generalization Degradation Curve Across Biological Holdout Regimes", weight='bold', pad=15)
    ax.set_ylabel("Metric Value", weight='bold')
    ax.set_ylim(0.60, 0.95)
    ax.legend(frameon=True, facecolor='white', edgecolor='#CCCCCC')

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig6_2_generalization_degradation.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Rendered Figure 6.2 -> {out_path}")

if __name__ == "__main__":
    main()
