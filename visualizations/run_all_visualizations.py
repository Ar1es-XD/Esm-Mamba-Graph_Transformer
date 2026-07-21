# -*- coding: utf-8 -*-
"""
run_all_visualizations.py
Master runner script to execute all thesis figure generation scripts sequentially.
Outputs 300 DPI PNGs and vector PDFs to visualizations/figures/.
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

FIGURE_SCRIPTS = [
    "fig1_dataset_distribution.py",
    "fig2_partition_splits.py",
    "fig3_sequence_lengths.py",
    "fig4_esm_embedding_pca.py",
    "fig5_fused_feature_pca.py",
    "fig6_benchmark_performance.py",
    "fig7_generalization_degradation.py",
    "fig8_model_diagnostics.py",
]

def main():
    print("=" * 80)
    print("  EXECUTING MAMBACROSS-GT THESIS VISUALIZATION ENGINE")
    print("=" * 80)
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    for script_name in FIGURE_SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        if os.path.exists(script_path):
            print(f"\n[RUNNING] Executing {script_name}...")
            res = subprocess.run([sys.executable, script_path], cwd=SCRIPT_DIR)
            if res.returncode == 0:
                print(f"[SUCCESS] Completed {script_name}")
            else:
                print(f"[WARNING] {script_name} returned code {res.returncode}")
        else:
            print(f"[PENDING] Script {script_name} not found.")

    print("\nVisualization run complete. Export directory: visualizations/figures/")

if __name__ == "__main__":
    main()
