# -*- coding: utf-8 -*-
"""
fig2_partition_splits.py
Generates Figure 4.2: Generalization Partitioning & Data Split Breakdown for Graph Transformer.
Outputs 300 DPI PNG to visualizations/figures/fig4_2_partition_splits.png.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA", "grid.linestyle": "--"})
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'figure.titlesize': 15
    })

    # Data for the 4 experimental splits
    experiments = ['Exp 1: Random Split', 'Exp 2: Novel Viruses', 'Exp 3: Novel Antibodies', 'Exp 4: Both Novel']
    train_n = [8970, 9183, 8686, 5216]
    test_n = [2240, 2027, 2524, 1096]

    df = pd.DataFrame({
        'Experiment': experiments,
        'Train Pairs': train_n,
        'Test Pairs': test_n
    })

    fig, ax = plt.subplots(figsize=(11, 6))
    
    bar_width = 0.35
    x = np.arange(len(experiments))

    rects1 = ax.bar(x - bar_width/2, df['Train Pairs'], bar_width, label='Train Set ($n_{\text{train}}$)', color='#8B5CF6', edgecolor='#6D28D9', linewidth=1.2)
    rects2 = ax.bar(x + bar_width/2, df['Test Pairs'], bar_width, label='Test Set ($n_{\text{test}}$)', color='#C084FC', edgecolor='#7E22CE', linewidth=1.2)

    ax.set_ylabel('Number of Interaction Pairs', weight='bold')
    ax.set_title('Figure 4.2 — Generalization Partitioning & Pair Distribution Breakdown', weight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, weight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='#CCCCCC')

    # Annotate bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:,}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold', color='#4C1D95')
                    
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f'{height:,}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, weight='bold', color='#6B21A8')

    ax.set_ylim(0, 11000)
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "fig4_2_partition_splits.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Rendered Figure 4.2 -> {out_path}")

if __name__ == "__main__":
    main()
