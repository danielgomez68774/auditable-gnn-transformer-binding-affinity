import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.nn import global_mean_pool
from scipy.interpolate import make_interp_spline
from pathlib import Path
# Safe import for the smooth LOWESS trend line
try:
    import statsmodels.api as sm
except ImportError:
    print("⚠️ 'statsmodels' is not installed. Run 'pip install statsmodels' in your terminal.")

# Avoid thread library deadlocks in local Windows environments
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
# Safe import from the local package
from Model.model import BindingAffinityModel

# =========================================================================
# ⚙️ SINGLE-GRAPH RESONANCE ANALYSIS CONFIGURATION
# =========================================================================
GRAFO_OBJETIVO = "1bcu.pt" 
PATH_CARPETA_GRAFOS = str(config.GRAFOS_CASF_DINAMICOS)
PATH_MODELO_PT = str(config.MODELS_DIR / "best_model_REFINEDsinCore_CASFtest_seed_7.pt")
RUTA_SALIDA = str(config.OUTPUTS_DIR / "LPE")

# Exact parameters of your trained architecture
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

# =========================================================================
# 🛠️ HOT INTERCEPTOR (MONKEY PATCHING) OF THE DYNAMIC FORWARD
# =========================================================================
def forward_interceptor_unico(self, data, return_attention=False):
    dist_matrix = torch.cdist(data.pos, data.pos, p=2)
    mask_cutoff = (dist_matrix <= self.cutoff)
    
    if data.batch is not None:
        batch_matrix = (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
        mask_cutoff = mask_cutoff & batch_matrix
        
    row, col = torch.where(mask_cutoff)
    edge_index_dinamico = torch.stack([row, col], dim=0)
    
    data.edge_index_vivas = edge_index_dinamico
    
    distancias_reales = dist_matrix[row, col]
    edge_attr_dinamico = self.rbf_generator(distancias_reales)
    
    h = self.mol_encoder(data.x, data.lpe)
    edge_attr = self.edge_encoder(edge_attr_dinamico)
    
    h = self.gine_block(h, edge_index_dinamico, edge_attr)
    res = self.transformer_block(h, edge_index_dinamico, edge_attr, pos=data.pos, return_attn=return_attention)
    
    h = res[0] if return_attention else res
    h_graph = global_mean_pool(h, data.batch)
    out = self.readout(h_graph)
    
    return (out, res[1]) if return_attention else out

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 Attentional Spectroscopy Probe Initialized on: {device}")
    
    # Injection of the dynamic patching
    BindingAffinityModel.forward = forward_interceptor_unico
    model = BindingAffinityModel(**MODEL_PARAMS).to(device)
    
    if not os.path.exists(PATH_MODELO_PT):
        print(f"❌ Error: Checkpoint not found at {PATH_MODELO_PT}")
        return
    model.load_state_dict(torch.load(PATH_MODELO_PT, map_location=device))
    model.eval()

    path_completo_grafo = os.path.join(PATH_CARPETA_GRAFOS, GRAFO_OBJETIVO)
    if not os.path.exists(path_completo_grafo):
        print(f"❌ Error: Target graph {GRAFO_OBJETIVO} not found.")
        return

    data = torch.load(path_completo_grafo, map_location='cpu', weights_only=False)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
    data = data.to(device)

    # Inference and capture of live attentions
    _, attn_weights = model(data, return_attention=True)
    edge_index_dinamico = data.edge_index_vivas if hasattr(data, 'edge_index_vivas') else None
    
    if attn_weights is None or edge_index_dinamico is None:
        print("❌ Critical error in attention maps.")
        return
        
    if attn_weights.dim() > 1:
        attn_weights = attn_weights.mean(dim=-1).view(-1)

    src, dst = edge_index_dinamico[0], edge_index_dinamico[1]
    molcodes = data.x[:, 14]
    
    # STRICT INTERFACIAL FILTERING (Protein <---> Ligand)
    mask_interfaz = (molcodes[src] * molcodes[dst] < 0)
    
    if mask_interfaz.sum() == 0:
        print("❌ Error: No intermolecular contacts detected within the cutoff radius.")
        return

    # Native tensor indexing to avoid misalignment on CUDA
    src_inter_t = src[mask_interfaz]
    dst_inter_t = dst[mask_interfaz]
    distancias_reales = torch.norm(data.pos[src_inter_t] - data.pos[dst_inter_t], dim=-1)

    dist_inter = distancias_reales.detach().cpu().numpy()
    attn_inter = attn_weights[mask_interfaz].detach().cpu().numpy()

    os.makedirs(RUTA_SALIDA, exist_ok=True)
    path_lbl = Path(PATH_MODELO_PT).stem

    # =========================================================================
    # 📊 PLOT 1: ATTENTIONAL STRATIFICATION (Quantization Scatter)
    # =========================================================================
    plt.figure(figsize=(7, 5.5), dpi=300)
    plt.scatter(dist_inter, attn_inter, color='#008080', alpha=0.35, edgecolors='none', s=35, 
                label=f'Audited Contacts ($n={len(dist_inter)}$)')
    
    try:
        lowess = sm.nonparametric.lowess
        z = lowess(attn_inter, dist_inter, frac=0.4)
        plt.plot(z[:, 0], z[:, 1], color='#dc143c', lw=2.5, linestyle='-', label='Attentional Mean Profile')
    except Exception as e:
        print(f"⚠️ LOWESS skipped in Plot 1: {e}")
        
    plt.axvspan(2.5, 3.5, color='#2ca02c', alpha=0.06, label=r'Golden Window ($2.5\AA - 3.5\AA$)')
    plt.title(f"Stratification and Quantized Information\nSingle Graph: {GRAFO_OBJETIVO}", fontsize=11, fontweight='bold', pad=10)
    plt.xlabel(r"Detected Interfacial Distance $d$ ($\text{Å}$)", fontsize=10)
    plt.ylabel(r"Attention Weight Magnitude ($\alpha$)", fontsize=10)
    plt.xlim(1.5, 4.5)
    plt.ylim(-0.02, attn_inter.max() + 0.03)
    plt.grid(True, linestyle=':', alpha=0.15)
    plt.legend(fontsize=8, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    plt.tight_layout()
    
    nombre_grafica_1 = f"individual_resonance_scatter_{GRAFO_OBJETIVO.replace('.pt', '')}_{path_lbl}.png"
    full_path_1 = os.path.join(RUTA_SALIDA, nombre_grafica_1)
    plt.savefig(full_path_1, dpi=300)
    plt.close() # Free Matplotlib backend memory

     # =========================================================================
    # 📊 PLOT 2: MEAN ATTENTION PER CONTACT vs DISTANCE (bias-free)
    # =========================================================================
    plt.figure(figsize=(7, 5.5), dpi=300)
    n_bins = 20

    # Sum of attention per bin AND number of contacts per bin
    sum_attn, bin_edges = np.histogram(dist_inter, bins=n_bins, weights=attn_inter)
    n_contacts, _       = np.histogram(dist_inter, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Mean attention per contact = sum / count  (avoids the edge-accumulation bias)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_attn = np.where(n_contacts > 0, sum_attn / n_contacts, np.nan)

    # Keep only populated bins for the curve
    valid = n_contacts > 0
    xc = bin_centers[valid]
    yc = mean_attn[valid]

    plt.bar(xc, yc, width=(bin_edges[1] - bin_edges[0]) * 0.75,
            color='#008080', alpha=0.12, label='Mean attention per contact (bin)')

    """
==========================================================================
SONDA ESPECTRAL DE 1bcu (GRAFO INDIVIDUAL REAL) — MULTI-MODELO
==========================================================================
Analiza la atención interfacial MEDIA POR CONTACTO en función de la distancia,
para el complejo cristalográfico real 1bcu, sobre los 8 modelos de lambda=0 y
los 8 de lambda=0.1.

MÉTRICA CLAVE: media por contacto (suma de atención / número de contactos por
bin), que evita el sesgo de acumulación de aristas (más contactos disponibles
a mayor distancia inflarían una suma). Es la versión corregida que revela el
perfil real, no el artefacto de la suma integrada.

Como es UN SOLO complejo, la variabilidad entre las 16 curvas proviene del
modelo (semilla), no de los datos. Es un CASO DE ESTUDIO ilustrativo, no
evidencia poblacional (para eso están las sondas sobre los 178 complejos).

Reporta:
  - curva media por contacto vs distancia, banda ± σ sobre semillas, por lambda
  - fracción de atención en la ventana [2.5, 3.5] Å por modelo (media ± σ)
  - posición del pico de atención media por contacto
  - test lambda=0 vs lambda=0.1
"""

import os
import csv
import numpy as np
import torch
import sys

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
from scipy import stats

# =========================================================================
CARPETA_MODELOS = str(config.MODELS_DIR)
PATH_CARPETA_GRAFOS = str(config.GRAFOS_CASF_DINAMICOS)
GRAFO_OBJETIVO = "1bcu.pt"
CARPETA_SALIDA = str(config.OUTPUTS_DIR / "LPE")
CUTOFF = 4.5
DMIN = 1.5   # como en la figura original (xlim 1.5-4.5)
N_BINS = 20
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0, 0.1]
VENTANA = (2.5, 3.5)
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}


def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_{lam_tag}_full.pt"


@torch.no_grad()
def atencion_interfacial_1bcu(model, data, device):
    """Devuelve (dist_inter, attn_inter) de los contactos interfaciales de 1bcu."""
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(device)
    out = model(data, return_attention=True)
    attn = out[1] if isinstance(out, tuple) else None
    if attn is None:
        return None, None
    molcode = data.x[:, -1]
    pos = data.pos
    dm = torch.cdist(pos, pos, p=2)
    mask = (dm <= CUTOFF)
    if data.batch is not None:
        bm = (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
        mask = mask & bm
    row, col = torch.where(mask)
    a = attn.mean(dim=-1).view(-1)
    inter = (molcode[row] * molcode[col]) < 0
    src_i, dst_i = row[inter], col[inter]
    dist = torch.norm(pos[src_i] - pos[dst_i], dim=-1).cpu().numpy()
    at = a[inter].cpu().numpy()
    return dist, at


def curva_media_por_contacto(dist, attn, bin_edges):
    """Media por contacto por bin (suma/conteo), sin sesgo de acumulación."""
    sum_attn, _ = np.histogram(dist, bins=bin_edges, weights=attn)
    n_cont, _ = np.histogram(dist, bins=bin_edges)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_attn = np.where(n_cont > 0, sum_attn / n_cont, np.nan)
    return mean_attn, n_cont


def fraccion_en_ventana(dist, attn):
    """Fracción de la atención total que cae en la ventana [2.5,3.5]."""
    total = attn.sum()
    if total <= 0:
        return 0.0
    m = (dist >= VENTANA[0]) & (dist <= VENTANA[1])
    return attn[m].sum() / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    ruta_grafo = os.path.join(PATH_CARPETA_GRAFOS, GRAFO_OBJETIVO)
    if not os.path.exists(ruta_grafo):
        print(f"❌ Grafo no encontrado: {ruta_grafo}"); return
    data_base = torch.load(ruta_grafo, map_location='cpu', weights_only=False)

    bin_edges = np.linspace(DMIN, CUTOFF, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    curvas = {}          # (lam, seed) -> mean_attn por bin
    frac_ventana = {}    # (lam, seed) -> fracción en ventana
    pico = {}            # (lam, seed) -> distancia del pico
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta):
                faltantes.append(ruta); continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta, map_location=device))
            model.eval()
            import copy
            data = copy.deepcopy(data_base)
            dist, at = atencion_interfacial_1bcu(model, data, device)
            if dist is None or len(dist) == 0:
                continue
            mean_attn, n_cont = curva_media_por_contacto(dist, at, bin_edges)
            curvas[(lam, seed)] = mean_attn
            frac_ventana[(lam, seed)] = fraccion_en_ventana(dist, at)
            # pico: bin con mayor media por contacto (entre bins poblados)
            valid = ~np.isnan(mean_attn)
            if valid.any():
                idx_pico = np.nanargmax(mean_attn)
                pico[(lam, seed)] = bin_centers[idx_pico]
            print(f"OK λ={lam} seed={seed:3d} | frac_ventana={frac_ventana[(lam,seed)]:.3f}  "
                  f"pico≈{pico.get((lam,seed), np.nan):.2f}Å  n_contactos={len(dist)}")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)
    if not curvas:
        print("❌ No se cargó ningún modelo."); return

    # ---- Guardar curvas ----
    ruta_c = os.path.join(CARPETA_SALIDA, "spectro_1bcu_curvas.csv")
    with open(ruta_c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["lambda", "seed", "dist_centro", "media_por_contacto"])
        for (lam, s), mean_attn in curvas.items():
            for dc, ma in zip(bin_centers, mean_attn):
                w.writerow([lam, s, f"{dc:.4f}", f"{ma:.6f}" if not np.isnan(ma) else "nan"])
    print(f"\nGuardado: {ruta_c}")

    # ---- Estadística: fracción en ventana y pico ----
    print("\n" + "=" * 64)
    print("1bcu — fracción de atención en ventana [2.5-3.5] y pico")
    print("=" * 64)
    for lam in LAMBDAS:
        fv = np.array([frac_ventana[(lam, s)] for s in SEEDS if (lam, s) in frac_ventana])
        pk = np.array([pico[(lam, s)] for s in SEEDS if (lam, s) in pico])
        if len(fv):
            print(f"  λ={lam}: frac_ventana = {fv.mean():.3f} ± {fv.std(ddof=1):.3f}  | "
                  f"pico medio ≈ {pk.mean():.2f} ± {pk.std(ddof=1):.2f} Å")

    # ---- Comparación λ=0 vs λ=0.1 (fracción en ventana) ----
    f0 = {s: frac_ventana[(0.0, s)] for s in SEEDS if (0.0, s) in frac_ventana}
    f1 = {s: frac_ventana[(0.1, s)] for s in SEEDS if (0.1, s) in frac_ventana}
    comunes = sorted(set(f0) & set(f1))
    if len(comunes) >= 2:
        a0 = np.array([f0[s] for s in comunes]); a1 = np.array([f1[s] for s in comunes])
        t, pt = stats.ttest_rel(a1, a0); W, pL = stats.levene(a0, a1, center='mean')
        print("\n" + "=" * 64)
        print("COMPARACIÓN 1bcu frac_ventana: λ=0 vs λ=0.1")
        print("=" * 64)
        print(f"  λ=0   : {a0.mean():.3f} ± {a0.std(ddof=1):.3f}")
        print(f"  λ=0.1 : {a1.mean():.3f} ± {a1.std(ddof=1):.3f}")
        print(f"  Δ (0.1-0): {(a1-a0).mean():+.3f}  | t={t:.3f} p={pt:.4f} | Levene p={pL:.4f}")

    # ---- Figura: media por contacto con banda ± σ ----
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 6), dpi=300)
        colores = {0.0: '#8c564b', 0.1: '#008080'}
        plt.axvspan(*VENTANA, color='#2ca02c', alpha=0.07, label='Non-covalent window (2.5–3.5 Å)')
        for lam in LAMBDAS:
            cs = [curvas[(l, s)] for (l, s) in curvas if l == lam]
            if not cs: continue
            M = np.vstack(cs)
            m = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(m)
            valid = ~np.isnan(m)
            plt.plot(bin_centers[valid], m[valid], color=colores[lam], linewidth=2.5,
                     marker='o', markersize=3, label=f"λ={lam}")
            plt.fill_between(bin_centers[valid], (m - sd)[valid], (m + sd)[valid],
                             color=colores[lam], alpha=0.18)
        plt.xlabel(r"Interfacial distance $d$ (Å)", fontsize=11)
        plt.ylabel(r"Mean attention per contact ($\bar{\alpha}$) ± σ", fontsize=11)
        plt.title(f"Mean attention per contact vs distance — {GRAFO_OBJETIVO.replace('.pt','')}\n"
                  "(mean ± σ over 8 seeds)", fontsize=11, fontweight='bold', pad=12)
        plt.xlim(DMIN, CUTOFF); plt.grid(True, linestyle=':', alpha=0.2)
        plt.legend(fontsize=9, loc='upper left')
        plt.tight_layout()
        ruta_fig = os.path.join(CARPETA_SALIDA, "spectro_1bcu_banda_ing.png")
        plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
        print(f"\nFigura guardada: {ruta_fig}")
    except Exception as e:
        print(f"(No se pudo generar la figura: {e})")


if __name__ == "__main__":
    main()