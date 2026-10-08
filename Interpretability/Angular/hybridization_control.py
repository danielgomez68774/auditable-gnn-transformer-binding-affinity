"""
==========================================================================
SONDA ANGULAR (C-N...O) — ANÁLISIS MULTI-MODELO RIGUROSO
==========================================================================
Barre el ángulo de interacción C-N...O de 0 a 180 grados con la distancia
N...O fija, y mide la atención sobre el contacto intermolecular, sobre los
8 modelos de lambda=0.0 y los 8 de lambda=0.1.

Cuantifica la ANISOTROPÍA ANGULAR mediante la amplitud de modulación de la
atención (máx - mín sobre el barrido). Reporta:
  - amplitud de modulación por modelo -> media ± σ por lambda + test λ0 vs λ0.1
  - ángulo del máximo (dato secundario: ¿dónde pone el modelo su máx atención?)
  - figura con banda media ± σ de la curva angular, para cada lambda.

RÉGIMEN: esta sonda opera en el régimen CONTROLADO (contacto aislado). Mide
la CAPACIDAD del modelo de responder al ángulo, no que esa respuesta domine
en complejos reales (ver análisis pareado de la sonda de distancia).

NOTA: la distancia N...O se fija en DIST_NO. Con DIST_NO = cutoff exacto,
se usa una pequeña tolerancia para no perder el contacto por punto flotante.
"""

import os
import csv
import numpy as np
import torch
import sys
from torch_geometric.data import Data

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
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "Angular")
CUTOFF = 4.5
DIST_NO = 4.4      # distancia N...O fija (claramente dentro del cutoff 4.5)
DIST_CN = 1.4      # distancia C-N de soporte
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0, 0.1]
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}


def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_{lam_tag}_full.pt"


@torch.no_grad()
def sonda_angular(model, device):
    """Devuelve (angulos, atencion) sobre el barrido C-N...O."""
    model.eval()
    angulos = np.linspace(0, 180, 181)
    feat_soporte = [0.0, 6.0, 0.0, 1.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.00, 1.70, -1.0]
    feat_centro  = [0.0, 7.0, 0.0, 1.0, 0.0, 0.0, 3.0, 3.0, 0.0, 1.0, 1.0, 0.0, -0.4, 1.55, -1.0]
    feat_fijo    = [1.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0, 0.0, -0.5, 1.52,  1.0]
    x = torch.tensor([feat_soporte, feat_centro, feat_fijo], dtype=torch.float).to(device)
    molcode = x[:, 14]
    atencion = []
    for theta_deg in angulos:
        theta = np.radians(theta_deg)
        pos = torch.tensor([
            [0.0, -DIST_CN, 0.0],
            [0.0,  0.0,     0.0],
            [DIST_NO * np.sin(theta), -DIST_NO * np.cos(theta), 0.0]
        ], dtype=torch.float).to(device)
        data = Data(x=x, pos=pos,
                    lpe=torch.zeros((3, 15)).to(device),
                    batch=torch.zeros(3, dtype=torch.long).to(device))
        _, attn = model(data, return_attention=True)
        if attn is None:
            atencion.append(0.0); continue
        dm = torch.cdist(pos, pos, p=2)
        r, c = torch.where(dm <= CUTOFF)   # idéntico al modelo (sin tolerancia)
        inter = (molcode[r] * molcode[c] < 0)
        avg = attn.mean(dim=-1).view(-1)
        atencion.append(avg[inter].mean().item() if inter.any() else 0.0)
    return angulos, np.array(atencion)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    curvas = {}          # (lam, seed) -> (angulos, atencion)
    metricas = []        # lam, seed, amplitud, angulo_max, media, atencion_180
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta):
                faltantes.append(ruta); continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta, map_location=device))
            ang, at = sonda_angular(model, device)
            curvas[(lam, seed)] = (ang, at)
            amplitud = at.max() - at.min()
            ang_max = ang[np.argmax(at)]
            at_180 = at[-1]  # atención en 180 grados (H-bond ideal)
            metricas.append([lam, seed, amplitud, ang_max, at.mean(), at_180])
            print(f"OK λ={lam} seed={seed:3d} | amplitud={amplitud:.3f}  "
                  f"áng_máx={ang_max:.0f}°  media={at.mean():.3f}  α(180°)={at_180:.3f}")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)
    if not curvas:
        print("❌ No se cargó ningún modelo."); return

    # ---- Guardar curvas y métricas ----
    ruta_c = os.path.join(CARPETA_SALIDA, "angular_curvas.csv")
    with open(ruta_c, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["lambda", "seed", "angulo", "atencion"])
        for (lam, s), (ang, at) in curvas.items():
            for a, v in zip(ang, at):
                w.writerow([lam, s, f"{a:.1f}", f"{v:.6f}"])
    ruta_m = os.path.join(CARPETA_SALIDA, "angular_metricas.csv")
    with open(ruta_m, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "seed", "amplitud", "angulo_max", "media", "atencion_180"])
        for row in metricas:
            w.writerow([row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.1f}",
                        f"{row[4]:.6f}", f"{row[5]:.6f}"])
    print(f"\nGuardado: {ruta_c}\nGuardado: {ruta_m}")

    # ---- Estadística: amplitud de modulación por lambda ----
    def col(lam, idx):
        return np.array([r[idx] for r in metricas if r[0] == lam])

    print("\n" + "=" * 64)
    print("ANISOTROPÍA ANGULAR (amplitud de modulación) por lambda")
    print("=" * 64)
    for lam in LAMBDAS:
        amp = col(lam, 2)
        if len(amp) == 0: continue
        print(f"  λ={lam}: amplitud = {amp.mean():.3f} ± {amp.std(ddof=1):.3f}  "
              f"(rango {amp.min():.3f}–{amp.max():.3f})")
        angmax = col(lam, 3)
        print(f"        ángulo del máximo: {angmax.mean():.0f}° ± {angmax.std(ddof=1):.0f}°")

    # ---- Comparación λ=0 vs λ=0.1 (amplitud) ----
    a0 = {r[1]: r[2] for r in metricas if r[0] == 0.0}
    a1 = {r[1]: r[2] for r in metricas if r[0] == 0.1}
    comunes = sorted(set(a0) & set(a1))
    if len(comunes) >= 2:
        x0 = np.array([a0[s] for s in comunes]); x1 = np.array([a1[s] for s in comunes])
        t, pt = stats.ttest_rel(x1, x0); W, pL = stats.levene(x0, x1, center='mean')
        print("\n" + "=" * 64)
        print("COMPARACIÓN amplitud angular: λ=0 vs λ=0.1")
        print("=" * 64)
        print(f"  media λ=0   : {x0.mean():.3f} ± {x0.std(ddof=1):.3f}")
        print(f"  media λ=0.1 : {x1.mean():.3f} ± {x1.std(ddof=1):.3f}")
        print(f"  Δ (0.1-0)   : {(x1-x0).mean():+.3f}")
        print(f"  t-test pareado: t={t:.3f}, p={pt:.4f}")
        print(f"  Levene: W={W:.3f}, p={pL:.4f}")

    # ---- Figura: banda media ± σ de la curva angular ----
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6), dpi=300)
        colores = {0.0: '#8c564b', 0.1: '#1f77b4'}
        for lam in LAMBDAS:
            cs = [at for (l, s), (ang, at) in curvas.items() if l == lam]
            if not cs: continue
            ang_ref = next(ang for (l, s), (ang, at) in curvas.items() if l == lam)
            M = np.vstack(cs)
            m = M.mean(axis=0); sd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(m)
            plt.plot(ang_ref, m, color=colores[lam], linewidth=2.5, label=f"λ={lam}")
            plt.fill_between(ang_ref, m - sd, m + sd, color=colores[lam], alpha=0.2)
        plt.xlabel(r"C-N$\cdots$O interaction angle $\theta$ (degrees)", fontsize=14)
        plt.ylabel(r"Mean attention ($\alpha$) ± σ", fontsize=14)
        plt.title("Angular response of the attention — mean ± σ over 8 seeds\n"
                  f"fixed N···O distance = {DIST_NO} Å)", fontsize=15, fontweight='bold', pad=12)
        plt.xticks(np.arange(0, 181, 30)); plt.xlim(-5, 185)         # era 9
        plt.xticks(fontsize=12)          # añadir
        plt.yticks(fontsize=12)          # añadir
        plt.grid(True, alpha=0.2); plt.legend(fontsize=12)
        plt.tight_layout()
        ruta_fig = os.path.join(CARPETA_SALIDA, "angular_banda_ing.png")
        plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {ruta_fig}")
    except Exception as e:
        print(f"(No se pudo generar la figura: {e})")


if __name__ == "__main__":
    main()