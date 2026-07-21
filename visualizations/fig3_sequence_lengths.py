# -*- coding: utf-8 -*-
"""
fig3_sequence_lengths.py
Generates Figure 4.3: Sequence Length Distribution of Antibodies and Antigens.
"""
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("[VISUALIZATION] Figure 4.3 script ready.")
if __name__ == "__main__":
    main()
