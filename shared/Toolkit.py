# -*- coding: utf-8 -*-
import numpy as np
import random
import torch
import os
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, \
     auc, precision_recall_curve

# Resolve paths relative to the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Use the preloaded embeddings from Esm-Mamba-Neural to avoid duplicating 3.5GB of data
_EMBEDDINGS_DIR = os.path.join(os.path.dirname(_PROJECT_ROOT), 'Esm-Mamba-Neural', 'Outputs', 'Pretrained_HIV')

# Fallback in case Esm-Mamba-Neural isn't in parent directory
if not os.path.exists(_EMBEDDINGS_DIR):
    _EMBEDDINGS_DIR = os.path.join(_PROJECT_ROOT, 'Outputs', 'Pretrained_HIV')

def Metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, y_score=y_pred)
    aupr_score = auc(lr_recall, lr_precision)
    auc_score = roc_auc_score(y_true=y_true, y_score=y_pred)
    acc_score = accuracy_score(y_true=y_true, y_pred=(y_pred > 0.5).astype('int'))
    f1_value = f1_score(y_true=y_true, y_pred=(y_pred > 0.5).astype('int'))
    
    return auc_score, aupr_score, f1_value, acc_score

def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)

class AntibodyAntigenDataset(Dataset):
    def __init__(self, pair_list):
        self.pairs = pair_list  # [(ab_id, ag_id, label)]
        preload_embeddings()
    
    def __getitem__(self, idx):
        ab_id, ag_id, label = self.pairs[idx]
        ab_id_safe = str(ab_id).replace('/', '_')
        ag_id_str  = str(ag_id)
        
        ab_emb = _EMBEDDING_CACHE.get(('ab', ab_id_safe))
        if ab_emb is None:
            ab_emb = torch.from_numpy(np.load(os.path.join(_EMBEDDINGS_DIR, 'ab', f'{ab_id_safe}.npy'))).float()
            
        ag_emb = _EMBEDDING_CACHE.get(('ag', ag_id_str))
        if ag_emb is None:
            ag_emb = torch.from_numpy(np.load(os.path.join(_EMBEDDINGS_DIR, 'ag', f'{ag_id_str}.npy'))).float()
            
        return (
            ab_emb,
            ag_emb,
            torch.tensor([float(label)], dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.pairs)

_EMBEDDING_CACHE = {}

def preload_embeddings():
    global _EMBEDDING_CACHE
    if len(_EMBEDDING_CACHE) > 0:
        return
    ab_dir = os.path.join(_EMBEDDINGS_DIR, 'ab')
    ag_dir = os.path.join(_EMBEDDINGS_DIR, 'ag')
    if os.path.exists(ab_dir):
        print(f"[RAM Cache] Preloading antibody embeddings from {ab_dir}...")
        for f in os.listdir(ab_dir):
            if f.endswith('.npy') and not f.startswith('.'):
                key = ('ab', f[:-4])
                arr = np.load(os.path.join(ab_dir, f))
                _EMBEDDING_CACHE[key] = torch.from_numpy(arr).float()
    if os.path.exists(ag_dir):
        print(f"[RAM Cache] Preloading antigen embeddings from {ag_dir}...")
        for f in os.listdir(ag_dir):
            if f.endswith('.npy') and not f.startswith('.'):
                key = ('ag', f[:-4])
                arr = np.load(os.path.join(ag_dir, f))
                _EMBEDDING_CACHE[key] = torch.from_numpy(arr).float()
    print(f"[RAM Cache] Success! Cached {len(_EMBEDDING_CACHE)} tensors in memory.")
    
def custom_collate_fn(batch):
    ab_embs = torch.stack([item[0] for item in batch])
    ag_embs = torch.stack([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch])
    return ab_embs, ag_embs, torch.squeeze(labels)
