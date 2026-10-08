"""
==========================================================================
MÉTRICA RL (COBERTURA ATENCIONAL DEL LIGANDO) — MULTI-MODELO
==========================================================================
Reconstrucción/Cobertura Atencional del Ligando sobre los 8 modelos de
lambda=0.0 y los 8 de lambda=0.1, con:

  1) BARRIDO DE TAU  (verificación de la selección de tau=0.1): muestra que
     la cobertura se sitúa en una meseta estable en la región de operación.
  2) RL con tau=0.1 sobre los 16 modelos: media ± σ por lambda + test
     lambda=0 vs lambda=0.1 (para confirmar equivalencia -> parsimonia).
  3) BASELINES NULOS (uniforme, barajado, atajo top-3) para dar contexto.

Definición de RL:
  Para cada átomo de ligando (molcode=+1), se acumula la atención (promediada
  sobre las cabezas) de sus contactos interfaciales (aristas cruzadas,
  molcode de signo opuesto):  A(v_i) = sum_j  alpha_ij
  Un átomo se cuenta como cubierto si A(v_i) >= tau.
  RL = (núm. átomos de ligando cubiertos / N_ligando) * 100

Descriptores adicionales (independientes de tau), como en la metodología:
  - entropía normalizada de la atención sobre el ligando
  - fracción efectiva del ligando (participation ratio / N)
  - coeficiente de Gini de concentración
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
# CONFIGURACIÓN
# =========================================================================
CARPETA_MODELOS = str(config.MODELS_DIR)
CARPETA_GRAFOS  = str(config.GRAFOS_REFINED)  # los 4407
LISTA_TEST      = str(config.SPLIT_DIR / "test.txt")    # 883 por similitud
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "RL_simsplit")
CUTOFF = 4.5
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0]          # en el split solo el modelo full con lambda=0
TAU = 0.1
TAUS_BARRIDO = np.linspace(0.0, 0.5, 26)
MAX_COMPLEJOS = None
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}


def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_SIMSPLIT_seed_{seed}_{lam_tag}_full.pt"


def leer_lista_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


@torch.no_grad()
def atencion_por_atomo_ligando(model, data, device):
    """Devuelve, para el complejo:
       - a_learned: A(v_i) por átomo de ligando (regla FIEL: suma a AMBOS extremos)
       - src, dst, attn_np, molcodes, lig_idx  (para calcular baselines igual)
    A(v_i) = suma de atención de las aristas cruzadas incidentes a i (como src O dst).
    """
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(device)
    out = model(data, return_attention=True)
    attn = out[1] if isinstance(out, tuple) else None
    if attn is None:
        return None

    molcode = data.x[:, -1]
    pos = data.pos
    dm = torch.cdist(pos, pos, p=2)
    mask = (dm <= CUTOFF)
    if data.batch is not None:
        bm = (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
        mask = mask & bm
    row, col = torch.where(mask)
    a = attn.mean(dim=-1).view(-1)  # promedio sobre cabezas

    src = row.cpu().numpy()
    dst = col.cpu().numpy()
    attn_np = a.cpu().numpy().astype(np.float64)
    molcodes = molcode.cpu().numpy()
    return src, dst, attn_np, molcodes


def accumulate_interfacial(attn_edges, src, dst, cross_mask, n_nodes):
    """A(v_i) = suma de atención sobre aristas cruzadas incidentes a i.
    REGLA FIEL: cada arista cruzada suma a AMBOS extremos (src y dst)."""
    acc = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(acc, src[cross_mask], attn_edges[cross_mask])
    np.add.at(acc, dst[cross_mask], attn_edges[cross_mask])
    return acc


def per_ligand_regimes(src, dst, attn_np, molcodes, rng):
    """Devuelve A(v_i) por átomo de ligando para learned, uniforme, barajado y atajo."""
    n_nodes = len(molcodes)
    lig = np.where(molcodes > 0)[0]
    cross = molcodes[src] * molcodes[dst] < 0

    # (1) learned
    a_learned = accumulate_interfacial(attn_np, src, dst, cross, n_nodes)[lig]
    # (2) uniforme: cada arista recibe el peso medio (mapa plano)
    uni = np.full_like(attn_np, attn_np.mean())
    a_uniform = accumulate_interfacial(uni, src, dst, cross, n_nodes)[lig]
    # (3) barajado: pesos aprendidos permutados sobre TODAS las aristas (ANTES de acumular)
    shuf = attn_np[rng.permutation(len(attn_np))]
    a_shuffled = accumulate_interfacial(shuf, src, dst, cross, n_nodes)[lig]
    # (4) atajo: toda la masa del learned en los 3 átomos de ligando más atendidos
    a_short = np.zeros_like(a_learned)
    if len(a_learned) >= 1:
        top = np.argsort(a_learned)[-3:]
        a_short[top] = a_learned[top]
    return a_learned, a_uniform, a_shuffled, a_short


def rl_cobertura(A_lig, tau):
    if A_lig is None or len(A_lig) == 0:
        return 0.0
    return 100.0 * np.sum(A_lig >= tau) / len(A_lig)


def descriptores(A_lig):
    """entropía normalizada, fracción efectiva (participation ratio/N), Gini."""
    if A_lig is None or len(A_lig) == 0 or A_lig.sum() <= 0:
        return 0.0, 0.0, 0.0
    p = A_lig / A_lig.sum()
    N = len(p)
    # entropía normalizada
    nz = p[p > 0]
    H = -np.sum(nz * np.log(nz)) / np.log(N) if N > 1 else 0.0
    # participation ratio / N
    pr = (A_lig.sum() ** 2) / (np.sum(A_lig ** 2) * N) if np.sum(A_lig**2) > 0 else 0.0
    # Gini
    s = np.sort(A_lig)
    n = len(s)
    gini = (2 * np.sum((np.arange(1, n+1)) * s) / (n * s.sum()) - (n + 1) / n) if s.sum() > 0 else 0.0
    return H, pr, gini


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    rng = np.random.default_rng(0)

    # Cargar SOLO los complejos del test por similitud (883), no toda la carpeta
    test_ids = leer_lista_ids(LISTA_TEST)
    archivos = [cid + ".pt" for cid in test_ids
                if os.path.exists(os.path.join(CARPETA_GRAFOS, cid + ".pt"))]
    if MAX_COMPLEJOS:
        archivos = archivos[:MAX_COMPLEJOS]
    print(f"Complejos del test por similitud: {len(archivos)}")

    # Estructuras: por (lam, seed) guardamos lista de A_lig por complejo
    rl_por_modelo = {}        # (lam,seed) -> RL medio (tau=0.1) sobre complejos
    desc_por_modelo = {}      # (lam,seed) -> (H, pr, gini) medios
    base_por_modelo = {}      # (lam,seed) -> dict baselines medios
    barrido_por_lam = {lam: [] for lam in LAMBDAS}  # curvas RL vs tau (media por modelo)
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta):
                faltantes.append(ruta); continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta, map_location=device))
            model.eval()

            rl_comp, H_comp, pr_comp, gini_comp = [], [], [], []
            base_comp = {"uniforme": [], "barajado": [], "atajo": []}
            curva_tau = np.zeros(len(TAUS_BARRIDO))

            for fn in archivos:
                try:
                    data = torch.load(os.path.join(CARPETA_GRAFOS, fn),
                                      map_location='cpu', weights_only=False)
                except Exception:
                    continue
                if data is None or not hasattr(data, 'pos'):
                    continue
                res_at = atencion_por_atomo_ligando(model, data, device)
                if res_at is None:
                    continue
                src, dst, attn_np, molcodes = res_at
                if np.sum(molcodes > 0) == 0:
                    continue
                a_learned, a_uniform, a_shuffled, a_short = per_ligand_regimes(
                    src, dst, attn_np, molcodes, rng)
                if len(a_learned) == 0:
                    continue
                rl_comp.append(rl_cobertura(a_learned, TAU))
                H, pr, gini = descriptores(a_learned)
                H_comp.append(H); pr_comp.append(pr); gini_comp.append(gini)
                base_comp["uniforme"].append(rl_cobertura(a_uniform, TAU))
                base_comp["barajado"].append(rl_cobertura(a_shuffled, TAU))
                base_comp["atajo"].append(rl_cobertura(a_short, TAU))
                curva_tau += np.array([rl_cobertura(a_learned, t) for t in TAUS_BARRIDO])

            n_ok = len(rl_comp)
            if n_ok == 0:
                continue
            rl_por_modelo[(lam, seed)] = np.mean(rl_comp)
            desc_por_modelo[(lam, seed)] = (np.mean(H_comp), np.mean(pr_comp), np.mean(gini_comp))
            base_por_modelo[(lam, seed)] = {k: np.mean(v) for k, v in base_comp.items()}
            barrido_por_lam[lam].append(curva_tau / n_ok)
            print(f"OK λ={lam} seed={seed:3d} | RL(τ=0.1)={np.mean(rl_comp):.2f}%  "
                  f"H={np.mean(H_comp):.3f}  fracEf={np.mean(pr_comp):.3f}  "
                  f"Gini={np.mean(gini_comp):.3f}")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)
    if not rl_por_modelo:
        print("❌ No se cargó ningún modelo."); return

    # ---- RL por lambda (media ± σ sobre semillas) ----
    print("\n" + "=" * 64)
    print("RL (τ=0.1) por lambda — media ± σ sobre semillas")
    print("=" * 64)
    rl_lam = {}
    for lam in LAMBDAS:
        vals = [rl_por_modelo[(lam, s)] for s in SEEDS if (lam, s) in rl_por_modelo]
        rl_lam[lam] = np.array(vals)
        if len(vals):
            print(f"  λ={lam}: RL = {np.mean(vals):.2f} ± {np.std(vals, ddof=1):.2f} %")

    # descriptores por lambda
    print("\nDescriptores (media sobre semillas):")
    for lam in LAMBDAS:
        Hs = [desc_por_modelo[(lam,s)][0] for s in SEEDS if (lam,s) in desc_por_modelo]
        prs = [desc_por_modelo[(lam,s)][1] for s in SEEDS if (lam,s) in desc_por_modelo]
        gs = [desc_por_modelo[(lam,s)][2] for s in SEEDS if (lam,s) in desc_por_modelo]
        if Hs:
            print(f"  λ={lam}: entropía={np.mean(Hs):.3f}  "
                  f"fracEfectiva={np.mean(prs):.3f}  Gini={np.mean(gs):.3f}")

    # baselines por lambda
    print("\nBaselines nulos (media sobre semillas, τ=0.1):")
    for lam in LAMBDAS:
        seeds_ok = [s for s in SEEDS if (lam,s) in base_por_modelo]
        if not seeds_ok: continue
        for k in ["uniforme", "barajado", "atajo"]:
            vals = [base_por_modelo[(lam,s)][k] for s in seeds_ok]
            print(f"  λ={lam} {k:9s}: {np.mean(vals):.2f} ± {np.std(vals,ddof=1):.2f} %")

    # ---- Test λ=0 vs λ=0.1 (solo si ambos lambdas están presentes) ----
    if 0.0 in rl_lam and 0.1 in rl_lam and len(rl_lam[0.0]) >= 2 and len(rl_lam[0.1]) >= 2:
        # pareado por semilla
        comunes = [s for s in SEEDS if (0.0,s) in rl_por_modelo and (0.1,s) in rl_por_modelo]
        a0 = np.array([rl_por_modelo[(0.0,s)] for s in comunes])
        a1 = np.array([rl_por_modelo[(0.1,s)] for s in comunes])
        t, pt = stats.ttest_rel(a1, a0)
        W, pL = stats.levene(a0, a1, center='mean')
        print("\n" + "=" * 64)
        print("COMPARACIÓN RL: λ=0 vs λ=0.1")
        print("=" * 64)
        print(f"  media λ=0   : {a0.mean():.2f} ± {a0.std(ddof=1):.2f} %")
        print(f"  media λ=0.1 : {a1.mean():.2f} ± {a1.std(ddof=1):.2f} %")
        print(f"  Δ (0.1-0)   : {(a1-a0).mean():+.2f} %")
        print(f"  t-test pareado: t={t:.3f}, p={pt:.4f}")
        print(f"  Levene: W={W:.3f}, p={pL:.4f}")

    # ---- Guardar ----
    ruta = os.path.join(CARPETA_SALIDA, "rl_por_modelo_split.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "seed", "RL_tau0.1", "entropia", "frac_efectiva",
                    "gini", "base_uniforme", "base_barajado", "base_atajo"])
        for (lam, s), rl in rl_por_modelo.items():
            H, pr, g = desc_por_modelo[(lam, s)]
            b = base_por_modelo[(lam, s)]
            w.writerow([lam, s, f"{rl:.4f}", f"{H:.4f}", f"{pr:.4f}", f"{g:.4f}",
                        f"{b['uniforme']:.4f}", f"{b['barajado']:.4f}", f"{b['atajo']:.4f}"])
    print(f"\nGuardado: {ruta}")

    # ---- Figura: barrido de tau (meseta) ----
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6), dpi=300)
        colores = {0.0: '#8c564b', 0.1: '#1f77b4'}
        for lam in LAMBDAS:
            if not barrido_por_lam[lam]:
                continue
            M = np.vstack(barrido_por_lam[lam])
            m = M.mean(axis=0); sd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(m)
            plt.plot(TAUS_BARRIDO, m, color=colores[lam], linewidth=2.5, label=f"λ={lam}")
            plt.fill_between(TAUS_BARRIDO, m - sd, m + sd, color=colores[lam], alpha=0.2)
        plt.axvline(x=0.1, color='green', linestyle=':', linewidth=2, label='τ = 0.1 (operating point)')
        plt.xlabel("Threshold τ", fontsize=11)
        plt.ylabel("RL coverage (%)", fontsize=11)
        plt.title("Robustness of RL to threshold τ (mean ± σ over seeds)",
                    fontsize=12, fontweight='bold', pad=12)
        plt.grid(True, alpha=0.2); plt.legend(fontsize=9)
        plt.tight_layout()
        ruta_fig = os.path.join(CARPETA_SALIDA, "rl_barrido_tau_split_ing.png")
        plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {ruta_fig}")
    except Exception as e:
        print(f"(No se pudo generar la figura: {e})")


if __name__ == "__main__":
    main()