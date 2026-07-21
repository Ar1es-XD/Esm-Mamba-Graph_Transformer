# -*- coding: utf-8 -*-
"""
fig6_benchmark_performance.py
Generates Figure 6.1: Benchmark Performance Comparison (AUROC and AUPRC) across
the four experimental partitions for the ESM-Mamba Graph Transformer pipeline.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
SUMMARY_CSV = os.path.join(ROOT_DIR, "gt_summary_results.csv")

def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[PENDING] Summary CSV not found at {SUMMARY_CSV}")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    
    df['Experiment_Name'] = df['Experiment'].replace({
        'experiment_1_random': 'Random Split',
        'experiment_2_novel_viruses': 'Novel Viruses',
        'experiment_3_novel_antibodies': 'Novel Antibodies',
        'experiment_4_both_novel': 'Both Novel (Double Holdout)',
        'Experiment 1 – Random Split': 'Random Split',
        'Experiment 2 – Novel Viruses': 'Novel Viruses',
        'Experiment 3 – Novel Antibodies': 'Novel Antibodies',
        'Experiment 4 – Both Novel (Double Holdout)': 'Both Novel'
    })

    sns.set_theme(style="whitegrid", rc={"grid.color": "#EAEAEA", "grid.linestyle": "--"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # AUROC Barplot
    sns.barplot(
        data=df, x='Experiment_Name', y='AUROC',
        color='#8B5CF6', ax=ax1, edgecolor='#6D28D9', linewidth=1.2
    )
    ax1.set_title("ESM-Mamba (MambaCross-GT) AUROC")
    ax1.set_xlabel("")
    ax1.set_ylabel("AUROC")
    ax1.set_ylim(0.5, 1.0)
    for p in ax1.patches:
        height = p.get_height()
        ax1.annotate(f'{height:.4f}',
                     (p.get_x() + p.get_width() / 2., height + 0.005),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#4C1D95')

    # AUPRC Barplot
    sns.barplot(
        data=df, x='Experiment_Name', y='AUPRC',
        color='#7C3AED', ax=ax2, edgecolor='#5B21B6', linewidth=1.2
    )
    ax2.set_title("ESM-Mamba (MambaCross-GT) AUPRC")
    ax2.set_xlabel("")
    ax2.set_ylabel("AUPRC")
    ax2.set_ylim(0.5, 1.0)
    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(f'{height:.4f}',
                     (p.get_x() + p.get_width() / 2., height + 0.005),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color='#4C1D95')

    plt.suptitle("ESM-Mamba Graph Transformer Generalization Performance", y=0.98, weight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(FIGURES_DIR, "fig6_1_benchmark_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Exported Figure 6.1 -> {out_path}")

if __name__ == "__main__":
    main()
