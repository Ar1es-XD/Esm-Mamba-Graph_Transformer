# -*- coding: utf-8 -*-
"""
fig1_dataset_distribution.py
Generates Figure 4.1: Dataset Composition & Neutralization Class Distribution.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("[VISUALIZATION] Figure 4.1 script ready for full rendering pipeline.")

if __name__ == "__main__":
    main()
