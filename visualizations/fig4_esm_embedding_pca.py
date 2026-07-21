# -*- coding: utf-8 -*-
"""
fig4_esm_embedding_pca.py
Generates Figure 4.4: Principal Component Analysis (PCA) of ESM-2 Sequence Embeddings.
"""
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("[VISUALIZATION] Figure 4.4 script ready.")
if __name__ == "__main__":
    main()
