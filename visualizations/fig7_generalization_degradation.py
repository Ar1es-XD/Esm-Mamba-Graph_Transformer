# -*- coding: utf-8 -*-
"""
fig7_generalization_degradation.py
Generates Figure 6.2: Generalization Degradation Curve & Entity Holdout Asymmetry for Graph Transformer.
"""
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("[VISUALIZATION] Figure 6.2 script ready.")
if __name__ == "__main__":
    main()
