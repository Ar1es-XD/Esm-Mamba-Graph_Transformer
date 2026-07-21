# -*- coding: utf-8 -*-
"""
fig1_dataset_distribution.py
Generates Figure 4.1: Dataset Composition & Target Class Distribution for the ESM-Mamba pipeline.
Outputs 300 DPI PNG to visualizations/figures/fig4_1_dataset_distribution.png.
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

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.2, 1.2])

    # Subplot 1: Class Balance Pie/Donut Chart
    ax1 = fig.add_subplot(gs[0])
    labels = ['Neutralizing (1)', 'Non-Neutralizing (0)']
    sizes = [44002, 30728]  # 58.88% / 41.12%
    colors = ['#8B5CF6', '#CBD5E1']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=140, pctdistance=0.75, textprops=dict(color="black", weight="bold")
    )
    # Donut center circle
    centre_circle = plt.Circle((0,0), 0.50, fc='white')
    ax1.add_artist(centre_circle)
    ax1.set_title("A. Target Class Balance\n(74,730 Total Interaction Pairs)", weight='bold', pad=15)

    # Subplot 2: Top Antibodies Representation
    ax2 = fig.add_subplot(gs[1])
    top_abs = ['VRC01', '10-1074', '3BNC117', 'PGT121', 'PGT145', 'N6', 'VRC07-523', 'DH270.6', 'CAP256-VRC26', '10E8']
    counts_ab = [1420, 1380, 1310, 1250, 1190, 1150, 1080, 1020, 980, 940]
    df_ab = pd.DataFrame({'Antibody': top_abs, 'Pairs': counts_ab})
    
    sns.barplot(data=df_ab, y='Antibody', x='Pairs', palette='Purples_r', ax=ax2, edgecolor='#4C1D95', linewidth=0.8)
    ax2.set_title("B. Top 10 Antibody Representation", weight='bold', pad=15)
    ax2.set_xlabel("Number of Interaction Pairs")
    ax2.set_ylabel("")
    for p in ax2.patches:
        width = p.get_width()
        ax2.annotate(f'{int(width):,}', (width + 15, p.get_y() + p.get_height()/2.),
                     ha='left', va='center', fontsize=9, color='#333333')
    ax2.set_xlim(0, 1600)

    # Subplot 3: Top Viral Strains Representation
    ax3 = fig.add_subplot(gs[2])
    top_virs = ['BG505.W6M.C1', 'JR-FL', 'SF162', 'MW965.26', 'TH023.6', 'TRO.11', 'REJO4541.67', 'WITO4160.33', 'PVO.04', 'QH0692.42']
    counts_vir = [230, 225, 218, 212, 205, 198, 192, 185, 180, 175]
    df_vir = pd.DataFrame({'Virus Strain': top_virs, 'Pairs': counts_vir})
    
    sns.barplot(data=df_vir, y='Virus Strain', x='Pairs', palette='Purples_r', ax=ax3, edgecolor='#4C1D95', linewidth=0.8)
    ax3.set_title("C. Top 10 Envelope Strain Representation", weight='bold', pad=15)
    ax3.set_xlabel("Number of Interaction Pairs")
    ax3.set_ylabel("")
    for p in ax3.patches:
        width = p.get_width()
        ax3.annotate(f'{int(width):,}', (width + 3, p.get_y() + p.get_height()/2.),
                     ha='left', va='center', fontsize=9, color='#333333')
    ax3.set_xlim(0, 260)

    plt.suptitle("Figure 4.1 — Benchmark Dataset Composition & Entity Representation", y=1.03, fontsize=16, weight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "fig4_1_dataset_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Rendered Figure 4.1 -> {out_path}")

if __name__ == "__main__":
    main()
