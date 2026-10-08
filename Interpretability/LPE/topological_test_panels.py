"""
==========================================================================
SONDA HOMONUCLEAR (LEY 1/N) — ANÁLISIS MULTI-MODELO RIGUROSO
==========================================================================
Sistema homonuclear: N átomos idénticos (mitad proteína, mitad ligando). Al no
haber señal química que rompa la simetría, la atención se reparte uniformemente
entre los contactos interfaciales -> la atención media por contacto escala como
1/N. Se verifica sobre N = 6, 12, 50, en los 8 modelos de lambda=0 y 8 de 0.1.

Dos verificaciones del 1/N (ambas prueban atención ∝ 1/N):
  (A) Razón observada vs teórica: atn(6):atn(12):atn(50) ≈ 1/6:1/12:1/50
  (B) Producto N×atención constante: si atn ≈ c/N -> N·atn ≈ c (constante)

Solo régimen controlado (el 1/N es un experimento sintético por diseño; no
tiene análogo directo en complejos reales).

NOTA: esta sonda SÍ usa el LPE (encoding posicional laplaciano), a diferencia
de las demás. Como los átomos son químicamente idénticos pero sus posiciones
difieren, el LPE es la única señal que podría romper la simetría 1/N perfecta.
"""

import os
import csv
import numpy as np
import torch
import scipy.sparse as sp
import sys
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

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
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "Population")
CUTOFF = 4.5
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0, 0.1]
N_VALORES = [6, 12, 50]
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}


def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_{lam_tag}_full.pt"


def calcular_lpe_sintetico(edge_index, num_nodes, k=15):
    edge_index_lap, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
    L_sparse = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
    try:
        k_adj = min(k + 1, num_nodes - 1)
        eig_vals, eig_vecs = sp.linalg.eigsh(L_sparse, k=k_adj, which='SA', tol=1e-3, maxiter=5000)
        idx = eig_vals.argsort()
        eig_vecs = eig_vecs[:, idx]
        lpe = eig_vecs[:, 1:k + 1]
        if lpe.shape[1] < k:
            lpe = np.concatenate([lpe, np.zeros((num_nodes, k - lpe.shape[1]))], axis=1)
    except Exception:
        lpe = np.zeros((num_nodes, k))
    return torch.from_numpy(lpe).float()


@torch.no_grad()
def sonda_homonuclear(model, device, n_atomos):
    """Atención interfacial media vs distancia para un sistema homonuclear de N átomos."""
    model.eval()
    distancias = np.linspace(1.2, CUTOFF, 50)
    n_bloque = n_atomos // 2
    feat_prot = [0.0, 7.0, 0.0, 1.0, 0.0, 0.0, 3.0, 3.0, 0.0, 1.0, 1.0, 0.0, -0.2, 1.55, -1.0]
    feat_lig  = [1.0, 6.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.05, 1.70,  1.0]
    x = torch.tensor([feat_prot] * n_bloque + [feat_lig] * n_bloque, dtype=torch.float).to(device)
    molcode = x[:, 14]
    np.random.seed(42)
    nube_prot_base = np.random.normal(0.0, 1.2, (n_bloque, 3))
    nube_lig_base = np.random.normal(0.0, 1.2, (n_bloque, 3))
    atencion = []
    for d in distancias:
        nube_prot = nube_prot_base.copy()
        nube_lig = nube_lig_base.copy()
        nube_lig[:, 0] += d
        pos = torch.tensor(np.concatenate([nube_prot, nube_lig], axis=0), dtype=torch.float).to(device)
        dm_full = torch.cdist(pos, pos, p=2)
        r_full, c_full = torch.where(dm_full <= CUTOFF)
        edge_index_full = torch.stack([r_full, c_full], dim=0)
        lpe = calcular_lpe_sintetico(edge_index_full.cpu(), num_nodes=n_atomos, k=15).to(device)
        data = Data(x=x, pos=pos, lpe=lpe,
                    batch=torch.zeros(n_atomos, dtype=torch.long).to(device))
        _, attn = model(data, return_attention=True)
        if attn is None:
            atencion.append(0.0); continue
        avg = attn.mean(dim=-1).view(-1)
        dm = torch.cdist(pos, pos, p=2)
        r, c = torch.where(dm <= CUTOFF)
        inter = (molcode[r] * molcode[c] < 0)
        atencion.append(avg[inter].mean().item() if inter.any() else 0.0)
    return distancias, np.array(atencion)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # curvas[(lam, seed, N)] = (dist, atencion)
    curvas = {}
    # atencion media (en la ventana 2.5-3.5) por modelo y N, para la ley 1/N
    atn_media = {}   # (lam, seed, N) -> atención media en ventana
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta):
                faltantes.append(ruta); continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta, map_location=device))
            for N in N_VALORES:
                dist, at = sonda_homonuclear(model, device, N)
                curvas[(lam, seed, N)] = (dist, at)
                # atención media en la ventana de contacto (2.5-3.5 A)
                win = (dist >= 2.5) & (dist <= 3.5)
                atn_media[(lam, seed, N)] = at[win].mean()
            vals = [atn_media[(lam, seed, N)] for N in N_VALORES]
            print(f"OK λ={lam} seed={seed:3d} | atn media (N=6,12,50) = "
                  f"{vals[0]:.4f}, {vals[1]:.4f}, {vals[2]:.4f} | "
                  f"N·atn = {6*vals[0]:.3f}, {12*vals[1]:.3f}, {50*vals[2]:.3f}")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)
    if not curvas:
        print("❌ No se cargó ningún modelo."); return

    # ---- Guardar curvas ----
    ruta_c = os.path.join(CARPETA_SALIDA, "homonuclear_curvas.csv")
    with open(ruta_c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["lambda", "seed", "N", "distancia", "atencion"])
        for (lam, s, N), (dist, at) in curvas.items():
            for dd, aa in zip(dist, at):
                w.writerow([lam, s, N, f"{dd:.4f}", f"{aa:.6f}"])
    print(f"\nGuardado: {ruta_c}")

    # ---- (A) Razón observada vs teórica y (B) producto N·atención ----
    print("\n" + "=" * 70)
    print("VERIFICACIÓN DE LA LEY 1/N")
    print("=" * 70)
    for lam in LAMBDAS:
        print(f"\nλ={lam}")
        # atención media por N (sobre semillas)
        for N in N_VALORES:
            vals = np.array([atn_media[(lam, s, N)] for s in SEEDS if (lam, s, N) in atn_media])
            if len(vals):
                print(f"  N={N:2d}: atn media = {vals.mean():.4f} ± {vals.std(ddof=1):.4f}  "
                      f"| N·atn = {N*vals.mean():.3f} ± {N*vals.std(ddof=1):.3f}  "
                      f"| 1/N teórico = {1/N:.4f}")

        # (A) razón observada 6:12:50 vs teórica
        a6 = np.mean([atn_media[(lam, s, 6)] for s in SEEDS if (lam, s, 6) in atn_media])
        a12 = np.mean([atn_media[(lam, s, 12)] for s in SEEDS if (lam, s, 12) in atn_media])
        a50 = np.mean([atn_media[(lam, s, 50)] for s in SEEDS if (lam, s, 50) in atn_media])
        if a50 > 0:
            print(f"  (A) Razón observada  atn(6)/atn(50) = {a6/a50:.2f}  "
                  f"(teórico 1/6÷1/50 = {(1/6)/(1/50):.2f} = {50/6:.2f})")
            print(f"      Razón observada  atn(6)/atn(12) = {a6/a12:.2f}  "
                  f"(teórico = {12/6:.2f})")
        # (B) coeficiente de variación del producto N·atn (0 = perfectamente constante)
        prods = np.array([N * np.mean([atn_media[(lam, s, N)] for s in SEEDS
                                       if (lam, s, N) in atn_media]) for N in N_VALORES])
        cv = prods.std() / prods.mean() if prods.mean() > 0 else 0
        print(f"  (B) Producto N·atn = {np.round(prods,3)}  | "
              f"coef. variación = {cv:.3f}  (0 = 1/N perfecto)")

    # ---- Comparación λ=0 vs λ=0.1 (producto N·atn, promedio sobre N) ----
    print("\n" + "=" * 70)
    print("COMPARACIÓN λ=0 vs λ=0.1 (adherencia al 1/N)")
    print("=" * 70)
    # métrica por modelo: CV del producto N·atn a través de los 3 N (0 = 1/N perfecto)
    def cv_modelo(lam, s):
        prods = [N * atn_media[(lam, s, N)] for N in N_VALORES if (lam, s, N) in atn_media]
        prods = np.array(prods)
        return prods.std() / prods.mean() if prods.mean() > 0 else np.nan
    cv0 = np.array([cv_modelo(0.0, s) for s in SEEDS if (0.0, s, 6) in atn_media])
    cv1 = np.array([cv_modelo(0.1, s) for s in SEEDS if (0.1, s, 6) in atn_media])
    cv0 = cv0[np.isfinite(cv0)]; cv1 = cv1[np.isfinite(cv1)]
    if len(cv0) and len(cv1):
        print(f"  CV del producto N·atn:")
        print(f"    λ=0   : {cv0.mean():.3f} ± {cv0.std(ddof=1):.3f}")
        print(f"    λ=0.1 : {cv1.mean():.3f} ± {cv1.std(ddof=1):.3f}")
        if len(cv0) == len(cv1) and len(cv0) >= 2:
            t, pt = stats.ttest_rel(cv1, cv0)
            print(f"    t-test pareado: t={t:.3f}, p={pt:.4f}")

    # ---- Figura: dos paneles (lambda=0 y lambda=0.1), 3 curvas cada uno ----
    # try:
    #     import matplotlib.pyplot as plt
    #     fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharey=True)
    #     colores = {6: '#9c27b0', 12: '#0288d1', 50: '#e65100'}
    #     estilos = {6: '-', 12: '--', 50: '-.'}
    #     paneles = {0.0: axes[0], 0.1: axes[1]}
    #     titulos = {0.0: 'λ = 0 (no physics)', 0.1: 'λ = 0.1'}

    #     for lam in LAMBDAS:
    #         ax = paneles[lam]
    #         for N in N_VALORES:
    #             cs = [at for (l, s, nn), (dist, at) in curvas.items() if l == lam and nn == N]
    #             if not cs:
    #                 continue
    #             dist_ref = next(dist for (l, s, nn), (dist, at) in curvas.items()
    #                             if l == lam and nn == N)
    #             M = np.vstack(cs)
    #             m = M.mean(axis=0)
    #             sd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(m)
    #             ax.plot(dist_ref, m, color=colores[N], linewidth=2.5, linestyle=estilos[N],
    #                     label=f"N = {N}")
    #             ax.fill_between(dist_ref, m - sd, m + sd, color=colores[N], alpha=0.15)
    #         ax.axvspan(2.5, 3.5, color='#2ca02c', alpha=0.07)
    #         ax.set_title(titulos[lam], fontsize=13, fontweight='bold')
    #         ax.set_xlabel("Interfacial separation distance $d$ (Å)", fontsize=12)
    #         ax.set_xlim(1.2, CUTOFF)
    #         ax.grid(True, alpha=0.2, linestyle=':')
    #         ax.legend(fontsize=11, loc='upper right')
    #         ax.tick_params(labelsize=11)

    #     axes[0].set_ylabel(r"Mean interfacial attention ($\alpha$) ± σ", fontsize=12)
    #     fig.suptitle("Interfacial attention in homonuclear systems of increasing density "
    #                  "(mean ± σ over 8 seeds)", fontsize=14, fontweight='bold')
    #     plt.tight_layout()
    #     ruta_fig = os.path.join(CARPETA_SALIDA, "homonuclear_banda_paneles.png")
    #     plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    #     print(f"\nFigura guardada: {ruta_fig}")
    # except Exception as e:
    #     print(f"(No se pudo generar la figura: {e})")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, sharey=True)
        colores = {6: '#9c27b0', 12: '#0288d1', 50: '#e65100'}
        estilos = {6: '-', 12: '--', 50: '-.'}
        paneles = {0.0: axes[0], 0.1: axes[1]}
        titulos = {0.0: 'λ = 0 (sin física)', 0.1: 'λ = 0.1'}

        for lam in LAMBDAS:
            ax = paneles[lam]
            for N in N_VALORES:
                cs = [at for (l, s, nn), (dist, at) in curvas.items() if l == lam and nn == N]
                if not cs:
                    continue
                dist_ref = next(dist for (l, s, nn), (dist, at) in curvas.items()
                                if l == lam and nn == N)
                M = np.vstack(cs)
                m = M.mean(axis=0)
                sd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(m)
                ax.plot(dist_ref, m, color=colores[N], linewidth=2.5, linestyle=estilos[N],
                        label=f"N = {N}")
                ax.fill_between(dist_ref, m - sd, m + sd, color=colores[N], alpha=0.15)
            ax.axvspan(2.5, 3.5, color='#2ca02c', alpha=0.07)
            ax.set_title(titulos[lam], fontsize=13, fontweight='bold')
            ax.set_xlabel("Distancia de separación interfacial $d$ (Å)", fontsize=12)
            ax.set_xlim(1.2, CUTOFF)
            ax.grid(True, alpha=0.2, linestyle=':')
            ax.legend(fontsize=11, loc='upper right')
            ax.tick_params(labelsize=11)

        axes[0].set_ylabel(r"Atención interfacial media ($\alpha$) ± σ", fontsize=12)
        fig.suptitle("Atención interfacial en sistemas homonucleares de densidad creciente "
                     "(media ± σ sobre 8 semillas)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        ruta_fig = os.path.join(CARPETA_SALIDA, "homonuclear_banda_paneles_es.png")
        plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
        print(f"\nFigura guardada: {ruta_fig}")
    except Exception as e:
        print(f"(No se pudo generar la figura: {e})")


if __name__ == "__main__":
    main()