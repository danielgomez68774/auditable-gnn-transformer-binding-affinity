import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, SiLU, LayerNorm, Dropout
from torch_geometric.nn import GINEConv, TransformerConv, global_mean_pool
from torch_scatter import scatter

def compute_local_angles(pos, edge_index):
    """
    Calcula el coseno del ángulo de coherencia local para cada arista de manera dinámica.
    """
    row, col = edge_index
    vecs = pos[col] - pos[row]
    vecs_norm = F.normalize(vecs, p=2, dim=-1, eps=1e-8)
    
    # Dirección promedio en cada nodo
    mean_vec = scatter(vecs_norm, row, dim=0, reduce='mean')
    
    # Coherencia (Coseno del ángulo): de -1 a 1
    local_coherence = torch.sum(vecs_norm * mean_vec[row], dim=-1)
    return local_coherence.unsqueeze(-1) # [E, 1]

class DistanceExpansionRBF(nn.Module):
    """
    🎯 NUEVO MÓDULO: Expansión Armónica Gaussiana calculada dentro del Forward.
    Garantiza que el atributo de arista reaccione instantáneamente a data.pos
    """
    def __init__(self, num_centers=64, cutoff=4.5):
        super().__init__()
        self.register_buffer('centers', torch.linspace(0.8, cutoff, num_centers))
        self.gamma = 2.4  # 🔥 Tu calibración campeona de alta resolución confinada

    def forward(self, distancias):
        # distancias: [E] -> output: [E, 64]
        return torch.exp(-self.gamma * (distancias.view(-1, 1) - self.centers.view(1, -1))**2)

class AngularExpansion(nn.Module):
    def __init__(self, num_bins=64, gamma=15.0):
        super().__init__()
        self.register_buffer('centers', torch.linspace(-1, 1, num_bins))
        self.gamma = gamma

    def forward(self, angles):
        return torch.exp(-self.gamma * (angles - self.centers)**2)

def build_gin_mlp(d_model):
    return Sequential(
        Linear(d_model, d_model * 2),
        SiLU(),
        Linear(d_model * 2, d_model),
        SiLU()
    )

class MolecularGNNBlock(nn.Module):
    def __init__(self, d_model, train_eps=True):
        super().__init__()
        self.conv = GINEConv(nn=build_gin_mlp(d_model), train_eps=train_eps)
        self.norm = LayerNorm(d_model)

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        return x + self.norm(out)

class MolecularTransformerBlock(nn.Module):
    def __init__(self, d_model, heads=4, dropout=0.2, angle_bins=16, use_gating=True):
        super().__init__()
        self.use_gating = use_gating # 🔥 Flag de Ablación para el Gating Angular
        
        self.conv = TransformerConv(
            in_channels=d_model,
            out_channels=d_model // heads,
            heads=heads,
            concat=True,
            edge_dim=d_model, 
            dropout=dropout
        )
        
        # Se instancian las capas para no romper las llaves del State Dict original
        self.angle_expansion = AngularExpansion(num_bins=angle_bins)
        # 🔧 Señal de interacción del par: proyección APRENDIDA de las
        # representaciones latentes de ambos extremos de la arista, en lugar de
        # tomar una única dimensión latente fija. Sigue siendo dinámica y
        # dependiente del modelo (la intención original), pero deriva de todo el
        # par (h_i, h_j) y no de un canal arbitrario.
        self.pair_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1)
        )
        self.feature_gate = nn.Sequential(
            nn.Linear(angle_bins + 1, d_model), 
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x, edge_index, edge_attr, pos, return_attn=False):
        
        # 📜 LÓGICA CON COMPUERTA SELECCIONABLE (Ablación del Gating)
        if self.use_gating:
            # 1. Obtener ángulos basados en el edge_index dinámico
            angles = compute_local_angles(pos, edge_index)
            angle_rbf = self.angle_expansion(angles)
            # 2. Señal de interacción del par derivada de una proyección aprendida
            #    de las representaciones latentes de ambos extremos de la arista.
            pair_repr = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)  # [E, 2*d_model]
            interaction_type = self.pair_proj(pair_repr)  # [E, 1]
            
            # 3. Gating Angular Activo
            gate_input = torch.cat([angle_rbf, interaction_type], dim=-1)
            gate = self.feature_gate(gate_input)
            refined_edge_attr = edge_attr * gate 
        else:
            # 🚫 Ablación activa: Bypass del gate. Las aristas pasan crudas sin filtrado angular
            refined_edge_attr = edge_attr
        
        # 4. Atención de Transformer
        if return_attn:
            out, attn_weights_data = self.conv(x, edge_index, refined_edge_attr, return_attention_weights=True)
            attn_weights = attn_weights_data[1] 
        else:
            out = self.conv(x, edge_index, refined_edge_attr)
            attn_weights = None
            
        x = self.norm(x + out)
        x = self.norm_ffn(x + self.ffn(x))
        
        return (x, attn_weights) if return_attn else x

class MolecularEncoder(nn.Module):
    def __init__(self, d_model=512, lpe_dim=15, use_lpe=True):
        super().__init__()
        self.use_lpe = use_lpe # 🔥 Flag de Ablación para los Embeddings Posicionales Laplacianos
        self.specs = [10, 128, 5, 10, 20, 5, 20, 20, 20, 5, 5, 5, 5]
        self.emb_dim = d_model // 32
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(num, self.emb_dim) for num in self.specs
        ])
        
        self.cont_dim = d_model // 32
        self.continuous_proj = nn.Sequential(
            nn.Linear(2, self.cont_dim),
            nn.SiLU()
        )
        
        self.lpe_out_dim = d_model // 4
        self.lpe_proj = nn.Linear(lpe_dim, self.lpe_out_dim)
        
        # El cálculo del tamaño se mantiene constante para blindar las dimensiones de fusión
        combined_dim = (self.emb_dim * len(self.specs)) + self.cont_dim + self.lpe_out_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x, lpe):
        embs = []
        for i in range(12):
            col_data = x[:, i].long()
            if i == 4: col_data = col_data + 5 
            col_data = torch.clamp(col_data, 0, self.specs[i]-1)
            embs.append(self.embeddings[i](col_data))
        
        molcode_data = (x[:, 14] + 1).long()
        molcode_data = torch.clamp(molcode_data, 0, self.specs[12]-1)
        embs.append(self.embeddings[12](molcode_data))
        
        h_cats = torch.cat(embs, dim=-1)
        h_cont = self.continuous_proj(x[:, 12:14])
        
        # 📜 LÓGICA CON FILTRADO RECEPTOR (Ablación LPE)
        if self.use_lpe and lpe is not None:
            h_lpe = self.lpe_proj(lpe)
        else:
            # 🚫 Ablación activa: Si se apaga el LPE, se inyecta un vector neutro de ceros equivalentes
            h_lpe = torch.zeros((x.size(0), self.lpe_out_dim), device=x.device, dtype=x.dtype)
        
        return self.fusion(torch.cat([h_cats, h_cont, h_lpe], dim=-1))

# =========================================================================
# 🏗️ MODELO MAESTRO MODULAR PARA ABLACIÓN
# =========================================================================
class BindingAffinityModel(nn.Module):
    def __init__(self, d_model=512, rbf_dim=64, lpe_dim=15, cutoff=4.5, 
                 use_rbf=True, use_lpe=True, use_gating=True):
        super().__init__()
        self.cutoff = cutoff
        self.use_rbf = use_rbf # 🔥 Flag de Ablación para las Funciones de Base Radial Geométricas
        
        # Traspaso coordinado de flags a los bloques atómicos internos
        self.mol_encoder = MolecularEncoder(d_model, lpe_dim, use_lpe=use_lpe)
        self.rbf_generator = DistanceExpansionRBF(num_centers=rbf_dim, cutoff=cutoff)
        
        self.edge_encoder = Sequential(
            Linear(rbf_dim, d_model),
            LayerNorm(d_model),
            SiLU(),
            Linear(d_model, d_model)
        )
        self.gine_block = MolecularGNNBlock(d_model)
        self.transformer_block = MolecularTransformerBlock(d_model, heads=4, angle_bins=16, use_gating=use_gating)
        
        self.readout = Sequential(
            Linear(d_model, d_model),
            LayerNorm(d_model),
            SiLU(),
            Dropout(0.3),
            Linear(d_model, d_model // 2),
            SiLU(),
            Linear(d_model // 2, 1)
        )

    def forward(self, data, return_attention=False, return_physics=False):
        # 1. Matriz de distancias euclidianas completa al vuelo [N, N]
        dist_matrix = torch.cdist(data.pos, data.pos, p=2)
        mask_cutoff = (dist_matrix <= self.cutoff)
        
        if data.batch is not None:
            batch_matrix = (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
            mask_cutoff = mask_cutoff & batch_matrix
            
        row, col = torch.where(mask_cutoff)
        edge_index_dinamico = torch.stack([row, col], dim=0)
        distancias_reales = dist_matrix[row, col]
        
        # 📜 LÓGICA CON FILTRADO RADIAL (Ablación RBF)
        if self.use_rbf:
            edge_attr_dinamico = self.rbf_generator(distancias_reales)
        else:
            # 🚫 Ablación activa: Si se apaga el RBF, la distancia geométrica real se vuelve invisible.
            # Rellenamos con ceros para neutralizar el peso métrico sin colapsar las dimensiones del tensor.
            edge_attr_dinamico = torch.zeros((edge_index_dinamico.size(1), self.rbf_generator.centers.size(0)), 
                                             device=distancias_reales.device, dtype=distancias_reales.dtype)
        
        h = self.mol_encoder(data.x, data.lpe)
        edge_attr = self.edge_encoder(edge_attr_dinamico)
        
        h = self.gine_block(h, edge_index_dinamico, edge_attr)
        
        # Cuando se pide la física, necesitamos la atención (depende de theta) junto
        # con el edge_index dinámico y las distancias, para construir una pérdida
        # física diferenciable que actúe sobre los coeficientes de atención.
        need_attn = return_attention or return_physics
        res = self.transformer_block(h, edge_index_dinamico, edge_attr, pos=data.pos, return_attn=need_attn)
        
        h = res[0] if need_attn else res
        h_graph = global_mean_pool(h, data.batch)
        out = self.readout(h_graph)
        
        if return_physics:
            # attn_weights: [E, heads] ; edge_index_dinamico: [2, E] ; distancias_reales: [E]
            physics = {
                "attn": res[1],
                "edge_index": edge_index_dinamico,
                "dist": distancias_reales,
            }
            return out, physics
        
        return (out, res[1]) if return_attention else out