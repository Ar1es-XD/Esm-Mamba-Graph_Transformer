# -*- coding: utf-8 -*-
"""
verify_weights.py
Verification script to inspect trained model weights and ensure checkpoint integrity
for the ESM-Mamba Graph Transformer pipeline.
"""
import os
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.dirname(SCRIPT_DIR)
ROOT = os.path.dirname(DOCS_DIR)

EXPERIMENTS = [
    ("Experiment 1 – Random Split", os.path.join(ROOT, "experiment_1_random", "results", "best_model.pt")),
    ("Experiment 2 – Novel Viruses", os.path.join(ROOT, "experiment_2_novel_viruses", "results", "best_model.pt")),
    ("Experiment 3 – Novel Antibodies", os.path.join(ROOT, "experiment_3_novel_antibodies", "results", "best_model.pt")),
    ("Experiment 4 – Both Novel", os.path.join(ROOT, "experiment_4_both_novel", "results", "best_model.pt"))
]

def main():
    print("=" * 80)
    print("  VERIFYING MAMBACROSS-GT MODEL WEIGHT CHECKPOINTS")
    print("=" * 80)
    
    report_lines = [
        "================================================================================",
        "VERIFICATION REPORT: GRAPH TRANSFORMER CHECKPOINT AUDIT",
        "================================================================================\n"
    ]
    
    all_valid = True
    for exp_name, model_path in EXPERIMENTS:
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location='cpu')
                num_keys = len(state_dict) if isinstance(state_dict, dict) else len(state_dict.state_dict())
                print(f"[OK] {exp_name}: Valid weights found ({num_keys} parameter tensors) -> {model_path}")
                report_lines.append(f"[VERIFIED] {exp_name}: Loaded successfully ({num_keys} parameter tensors)")
            except Exception as e:
                print(f"[ERROR] {exp_name}: Failed to load weights ({e}) -> {model_path}")
                report_lines.append(f"[FAILED] {exp_name}: Error loading weights ({e})")
                all_valid = False
        else:
            print(f"[WARNING] {exp_name}: Model checkpoint not found at {model_path}")
            report_lines.append(f"[MISSING] {exp_name}: Checkpoint missing at {model_path}")
            all_valid = False
            
    report_lines.append("\n================================================================================")
    
    report_path = os.path.join(DOCS_DIR, "verification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
        
    print(f"\nVerification report updated at: {report_path}")
    return 0 if all_valid else 1

if __name__ == "__main__":
    main()
