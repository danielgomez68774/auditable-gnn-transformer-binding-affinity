import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Force PyTorch to ignore library duplication conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 🎯 IMPORTACIÓN DEL MODELO DINÁMICO CORREGIDO (CON GATING)
import sys

# =========================================================================
# 🎯 ANCLA DE RUTAS: Forzar a Python a encontrar la subcarpeta 'Modelo'
# =========================================================================
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config
RUTA_RAIZ_PROYECTO = str(config.PROJECT_ROOT)
if RUTA_RAIZ_PROYECTO not in sys.path:
    sys.path.insert(0, RUTA_RAIZ_PROYECTO)
from Model.model import BindingAffinityModel

# =========================================================================
# CONFIG  (same paths / threshold as analisis-dataset-enlaces.py)
# =========================================================================
SEED = 7
PATH_GRAFOS_TARGET = str(config.GRAFOS_CASF_CORE285)
PATH_MODELO_PT = str(config.MODELS_DIR / f"best_model_REFINEDsinCore_CASFtest_seed_{SEED}.pt")
RUTA_RAIZ_SALIDA = str(config.OUTPUTS_DIR / "RL")
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

TAU_OP = 0.1                          # operating threshold reported in the paper
TAU_GRID = np.linspace(0.0, 0.5, 51)  # sweep for the sensitivity analysis
RNG = np.random.default_rng(SEED)     # reproducible shuffling for the null baseline

# =========================================================================
# 🛠️ SAME DYNAMIC INTERCEPTOR AS THE MAIN ANALYSIS SCRIPT
# =========================================================================
def forward_interceptor_5angstrom(self, data, return_attention=False):
    from torch_geometric.nn import global_mean_pool
    self.cutoff = 4.5
    dist_matrix = torch.cdist(data.pos, data.pos, p=2)
    mask_cutoff = (dist_matrix <= self.cutoff)
    if data.batch is not None:
        mask_cutoff = mask_cutoff & (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))

    row, col = torch.where(mask_cutoff)
    edge_index_dinamico = torch.stack([row, col], dim=0)
    data.edge_index_vivas = edge_index_dinamico

    edge_attr_dinamico = self.rbf_generator(dist_matrix[row, col])
    h = self.mol_encoder(data.x, data.lpe)
    edge_attr = self.edge_encoder(edge_attr_dinamico)

    h = self.gine_block(h, edge_index_dinamico, edge_attr)
    res = self.transformer_block(h, edge_index_dinamico, edge_attr, pos=data.pos, return_attn=return_attention)

    h_graph = global_mean_pool(res[0] if return_attention else res, data.batch)
    out = self.readout(h_graph)
    return (out, res[1]) if return_attention else out


def accumulate_interfacial(attn_edges, src, dst, cross_mask, n_nodes):
    """A(v_i) = sum of edge attention over cross (protein<->ligand) edges incident to i.
    Identical accumulation rule to the RL metric in the main script."""
    acc = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(acc, src[cross_mask], attn_edges[cross_mask])
    np.add.at(acc, dst[cross_mask], attn_edges[cross_mask])
    return acc


def per_ligand_attention(attn_np, src, dst, molcodes):
    """Return per-ligand-atom interfacial attention A(v_i) for the three regimes:
    learned, uniform (flat map) and shuffled (learned weights permuted across edges)."""
    n_nodes = len(molcodes)
    lig = np.where(molcodes > 0)[0]
    cross = molcodes[src] * molcodes[dst] < 0

    # (1) Learned attention (the real model)
    a_learned = accumulate_interfacial(attn_np, src, dst, cross, n_nodes)[lig]

    # (2) Uniform null: every edge receives the same weight (global mean -> flat map)
    uni = np.full_like(attn_np, attn_np.mean())
    a_uniform = accumulate_interfacial(uni, src, dst, cross, n_nodes)[lig]

    # (3) Shuffled null: learned weights randomly permuted across all edges
    shuf = attn_np[RNG.permutation(len(attn_np))]
    a_shuffled = accumulate_interfacial(shuf, src, dst, cross, n_nodes)[lig]

    return a_learned, a_uniform, a_shuffled


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BindingAffinityModel.forward = forward_interceptor_5angstrom
    model = BindingAffinityModel(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(PATH_MODELO_PT, map_location=device))
    model.eval()

    file_list = [f for f in os.listdir(PATH_GRAFOS_TARGET) if f.endswith('.pt')]

    per_complex = []   # each entry: dict of arrays {'learned','uniform','shuffled'}
    skipped = 0

    with torch.no_grad():
        for file_name in tqdm(file_list, desc="RL sensitivity + null baselines"):
            data = torch.load(os.path.join(PATH_GRAFOS_TARGET, file_name),
                              map_location=device, weights_only=False)
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
            data = data.to(device)

            _, attn_weights = model(data, return_attention=True)
            edge_index = data.edge_index_vivas if hasattr(data, 'edge_index_vivas') else data.edge_index
            if attn_weights is None or edge_index is None:
                skipped += 1
                continue
            if attn_weights.dim() > 1:
                attn_weights = attn_weights.mean(dim=-1).view(-1)

            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
            attn_np = attn_weights.detach().cpu().numpy()
            molcodes = data.x[:, -1].cpu().numpy()

            # defensive: attention vector must align with the edge list
            if attn_np.shape[0] != src.shape[0]:
                skipped += 1
                continue

            lig = np.where(molcodes > 0)[0]
            prot = np.where(molcodes < 0)[0]
            if len(lig) == 0 or len(prot) == 0:
                skipped += 1
                continue

            a_l, a_u, a_s = per_ligand_attention(attn_np, src, dst, molcodes)
            per_complex.append({'learned': a_l, 'uniform': a_u, 'shuffled': a_s})

    n_cx = len(per_complex)

    # ---- Sweep tau: RL per complex, then macro-average across complexes ----
    # (matches how the paper reports RL = 89.70 +/- 4.54 %: mean/std over complexes)
    def rl_curve(key):
        means, stds = [], []
        for tau in TAU_GRID:
            rls = np.array([100.0 * np.mean(c[key] >= tau) for c in per_complex])
            means.append(rls.mean()); stds.append(rls.std())
        return np.array(means), np.array(stds)

    m_l, s_l = rl_curve('learned')
    m_u, s_u = rl_curve('uniform')
    m_s, s_s = rl_curve('shuffled')

    os.makedirs(RUTA_RAIZ_SALIDA, exist_ok=True)
    df = pd.DataFrame({
        'tau': TAU_GRID,
        'RL_learned_mean': m_l, 'RL_learned_std': s_l,
        'RL_uniform_mean': m_u, 'RL_uniform_std': s_u,
        'RL_shuffled_mean': m_s, 'RL_shuffled_std': s_s,
    })
    csv_path = os.path.join(RUTA_RAIZ_SALIDA, f"RL_tau_sensitivity_seed_{SEED}.csv")
    df.to_csv(csv_path, index=False)

    # ---- Numbers at the operating threshold (to fill the paper placeholders) ----
    def rl_at(tau, key):
        rls = np.array([100.0 * np.mean(c[key] >= tau) for c in per_complex])
        return rls.mean(), rls.std()

    l_mean, l_std = rl_at(TAU_OP, 'learned')
    u_mean, u_std = rl_at(TAU_OP, 'uniform')
    sh_mean, sh_std = rl_at(TAU_OP, 'shuffled')

    print("\n" + "=" * 72)
    print(f"RL at operating threshold tau = {TAU_OP}   (n = {n_cx} complexes, skipped = {skipped})")
    print("=" * 72)
    print(f"  Learned attention : {l_mean:6.2f} +/- {l_std:.2f} %   <-- should match mean(R_mol)*100")
    print(f"  Uniform  (null)   : {u_mean:6.2f} +/- {u_std:.2f} %")
    print(f"  Shuffled (null)   : {sh_mean:6.2f} +/- {sh_std:.2f} %")
    print("=" * 72)
    print(f"CSV saved:    {csv_path}")

    # ---- Figure: RL(tau) with the two null baselines ----
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(TAU_GRID, m_l, color='#1f77b4', lw=2.5, label='Learned attention')
    plt.fill_between(TAU_GRID, m_l - s_l, m_l + s_l, color='#1f77b4', alpha=0.15)
    plt.plot(TAU_GRID, m_u, color='#7f7f7f', lw=2.0, ls='--', label='Uniform (null)')
    plt.plot(TAU_GRID, m_s, color='#d62728', lw=2.0, ls=':', label='Shuffled (null)')
    plt.axvline(TAU_OP, color='#2ca02c', lw=1.5, ls='-.', label=fr'Operating $\tau = {TAU_OP}$')
    plt.xlabel(r'Attention threshold $\tau$', fontsize=11)
    plt.ylabel(r'Ligand Attentional Reconstruction  RL($\tau$) [%]', fontsize=11)
    plt.title('RL sensitivity to the attention threshold (CASF-2016 core)',
              fontsize=11, fontweight='bold')
    plt.ylim(-2, 102)
    plt.grid(True, alpha=0.15, ls=':')
    plt.legend(fontsize=9, frameon=False, loc='upper right')
    plt.tight_layout()
    fig_path = os.path.join(RUTA_RAIZ_SALIDA, f"RL_tau_sensitivity_seed_{SEED}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()