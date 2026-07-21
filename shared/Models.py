# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from mambapy.vim import VMamba, MambaConfig
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

torch.manual_seed(42)
   
class MambaCross(nn.Module):
    def __init__(self, hor_dim, ver_dim, feat_dim, seq_len, hidden_sizes,
                 mamba_layer, pooling='avg', activation='SiLU', drop_ratio=0.1,
                 hidden_dim=128, num_gt_layers=2, gt_heads=4, k_contacts=5):
        super(MambaCross, self).__init__()
        self.W = nn.Parameter(torch.randn(feat_dim, feat_dim))  
        self.hor_dim = hor_dim # L_ag (256)
        self.ver_dim = ver_dim # L_ab (256)
        self.hidden_dim = hidden_dim
        self.k_contacts = k_contacts
        
        # -------mamba_encoder
        self.config_hor = MambaConfig(d_model=hor_dim, expand_factor=1, n_layers=mamba_layer)
        self.config_ver = MambaConfig(d_model=ver_dim, expand_factor=1, n_layers=mamba_layer)
        self.mamba_hor = VMamba(self.config_hor)
        self.mamba_ver = VMamba(self.config_ver)
        
        # -------node feature projection
        # ESM-2 embeddings projection
        self.proj_esm = nn.Linear(feat_dim, hidden_dim)
        # Mamba sweeps projection
        self.proj_mamba_ab = nn.Linear(hor_dim, hidden_dim)
        self.proj_mamba_ag = nn.Linear(ver_dim, hidden_dim)
        
        # -------graph transformer layers
        self.gt_layers = nn.ModuleList([
            TransformerConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=gt_heads,
                concat=False,
                dropout=drop_ratio
            ) for _ in range(num_gt_layers)
        ])
        
        # ------predict_decoder (Classification MLP Head)
        if activation == 'SiLU':
            self.act = F.silu
        elif activation == 'Leaky':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'Tanh':
            self.act = F.tanh
        else:
            self.act = F.relu
 
        self.hidden_layers = nn.ModuleList()
        # Input to decoder is global pooled features: mean + max pooling = 2 * hidden_dim
        prev_size = 2 * hidden_dim
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            self.hidden_layers.append(nn.BatchNorm1d(hidden_size))
            prev_size = hidden_size
        self.output_layer = nn.Linear(prev_size, 1)
        self.r = drop_ratio
        self.reset_para()
     
    def reset_para(self):
        nn.init.xavier_uniform_(self.W) 
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return
 
    def forward(self, x_Ab, x_Ag):
        batch_size = x_Ab.size(0)
        
        # Downsample sequence dimension to hor_dim/ver_dim to optimize pscan memory and speed
        x_Ab_ds = F.adaptive_avg_pool1d(x_Ab.transpose(1, 2), self.ver_dim).transpose(1, 2)
        x_Ag_ds = F.adaptive_avg_pool1d(x_Ag.transpose(1, 2), self.hor_dim).transpose(1, 2)

        #-----Mamba fusion
        contacts = torch.matmul(torch.matmul(x_Ab_ds, self.W), x_Ag_ds.transpose(1, 2))
        x_Ab_mamba = self.mamba_hor(contacts)  # (batch, ver_dim, hor_dim)
        x_Ag_mamba = self.mamba_ver(contacts.transpose(1, 2))  # (batch, hor_dim, ver_dim)
        
        #-----Graph Construction & Graph Transformer
        # 1. Project node features
        x_esm = torch.cat([x_Ab_ds, x_Ag_ds], dim=1) # (batch, ver_dim + hor_dim, 320)
        x_esm_proj = self.proj_esm(x_esm) # (batch, 512, hidden_dim)
        
        x_ab_m_proj = self.proj_mamba_ab(x_Ab_mamba) # (batch, ver_dim, hidden_dim)
        x_ag_m_proj = self.proj_mamba_ag(x_Ag_mamba) # (batch, hor_dim, hidden_dim)
        x_mamba_proj = torch.cat([x_ab_m_proj, x_ag_m_proj], dim=1) # (batch, 512, hidden_dim)
        
        # Combine node representations
        h = x_esm_proj + x_mamba_proj # (batch, 512, hidden_dim)
        h_flat = h.view(-1, self.hidden_dim) # (batch * 512, hidden_dim)
        
        # 2. Build edges in a vectorised batch-parallel format
        device = x_Ab.device
        num_nodes_per_graph = self.ver_dim + self.hor_dim # 512
        
        # Sequence edges (backbone) for a single graph
        ab_static = torch.stack([
            torch.arange(self.ver_dim - 1, device=device),
            torch.arange(1, self.ver_dim, device=device)
        ], dim=0)
        ab_static_rev = torch.stack([ab_static[1], ab_static[0]], dim=0)
        
        ag_static = torch.stack([
            torch.arange(self.ver_dim, num_nodes_per_graph - 1, device=device),
            torch.arange(self.ver_dim + 1, num_nodes_per_graph, device=device)
        ], dim=0)
        ag_static_rev = torch.stack([ag_static[1], ag_static[0]], dim=0)
        
        single_static = torch.cat([ab_static, ab_static_rev, ag_static, ag_static_rev], dim=1) # (2, 1020)
        
        # Repeat static sequence edges for the batch and offset
        batch_offsets_static = (torch.arange(batch_size, device=device) * num_nodes_per_graph).view(-1, 1, 1)
        static_edges_batch = single_static.unsqueeze(0).repeat(batch_size, 1, 1)
        static_edges_batch = static_edges_batch + batch_offsets_static.transpose(1, 2)
        static_edges = static_edges_batch.transpose(0, 1).flatten(1) # (2, batch_size * 1020)
        
        # Dynamic contact edges from the contact maps
        # select top-k contacts per antibody residue
        _, topk_idx = torch.topk(contacts, k=self.k_contacts, dim=-1) # (batch, ver_dim, k)
        
        ver_indices = torch.arange(self.ver_dim, device=device).unsqueeze(1).expand(-1, self.k_contacts)
        ver_indices = ver_indices.unsqueeze(0).expand(batch_size, -1, -1) # (batch, ver_dim, k)
        
        batch_offsets_dynamic = (torch.arange(batch_size, device=device) * num_nodes_per_graph).view(-1, 1, 1)
        
        ab_nodes_global = ver_indices + batch_offsets_dynamic
        ag_nodes_global = topk_idx + self.ver_dim + batch_offsets_dynamic
        
        src_edges = ab_nodes_global.flatten()
        dst_edges = ag_nodes_global.flatten()
        
        # Bidirectional dynamic contact edges
        contact_edges = torch.stack([
            torch.cat([src_edges, dst_edges]),
            torch.cat([dst_edges, src_edges])
        ], dim=0) # (2, 2 * batch_size * ver_dim * k)
        
        # Combine static backbone and dynamic interaction edges
        edge_index = torch.cat([static_edges, contact_edges], dim=1)
        
        # 3. Apply Graph Transformer Conv layers
        for gt_layer in self.gt_layers:
            h_flat = gt_layer(h_flat, edge_index)
            h_flat = self.act(h_flat)
            h_flat = torch.dropout(h_flat, self.r, train=self.training)
            
        # 4. Global pooling to extract graph-level representations
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, num_nodes_per_graph).flatten()
        
        x_mean = global_mean_pool(h_flat, batch_indices) # (batch, hidden_dim)
        x_max = global_max_pool(h_flat, batch_indices) # (batch, hidden_dim)
        x = torch.cat([x_mean, x_max], dim=-1) # (batch, 2 * hidden_dim)
        
        #------MLP decoder classification head
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            else:
                x = self.act(layer(x))
                x = torch.dropout(x, self.r, train=self.training)
        x = torch.squeeze(self.output_layer(x))  # output logits
        
        return torch.sigmoid(x)
