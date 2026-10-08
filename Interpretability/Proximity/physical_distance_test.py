"""
====================================================================
SONDA DE DISTANCIA (band response) — ANÁLISIS MULTI-MODELO RIGUROSO
====================================================================
Corre la sonda de proximidad sobre los 8 modelos de lambda=0.0 y los 8 de
lambda=0.1, y cuantifica la DISTRIBUCIÓN de atención por zonas físicas:

    Zona estérica     : [1.2, 2.5) Å   (solapamiento; atención baja = coherente)
    Ventana H-bond    : [2.5, 3.5] Å   (puente de hidrógeno; zona de interés)
    Zona vdW/larga    : (3.5, 4.5] Å   (contacto débil de largo alcance)

Para cada modelo calcula la FRACCIÓN de atención en cada zona. Luego, por cada
lambda, saca media ± desviación estándar sobre las 8 semillas y compara
lambda=0 vs lambda=0.1 (t-test pareado y Levene de varianzas).

Genera:
  - CSV con TODAS las curvas punto a punto (regraficable): sonda_dist_curvas.csv
  - CSV con las fracciones por zona por modelo:            sonda_dist_fracciones.csv
  - Figura con banda media ± σ para cada lambda:           sonda_dist_banda.png
  - Resumen estadístico por consola.

NOTA: no pude ejecutar torch en el entorno donde se escribió este script, así
que la PRIMERA vez que lo corras, revisa que no haya errores de dimensión ni
de nombres de archivo (el script lista lo que encuentra y lo que falta).
"""

import os
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from scipy import stats

# --- Import del modelo (ajusta la ruta raíz si hace falta) ---
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

# =========================================================================
# CONFIGURACIÓN
# =========================================================================
CARPETA_MODELOS = str(config.MODELS_DIR)
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "RBF")
CUTOFF = 4.5
SEEDS  = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0, 0.1]
N_PUNTOS = 50  # puntos en el barrido de distancia

MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

# Zonas físicas (a priori, con base biofísica)
ZONA_ESTERICA = (1.2, 2.5)   # [inicio, fin)
ZONA_VENTANA  = (2.5, 3.5)   # ventana de puente de hidrógeno
ZONA_VDW      = (3.5, 4.5)   # contacto largo / van der Waals

def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")   # 0.1 -> lam0p1 ; 0.0 -> lam0p0
    return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_{lam_tag}_full.pt"


# =========================================================================
# LA SONDA (idéntica a la tuya, devuelve la curva atención vs distancia)
# =========================================================================
@torch.no_grad()
def ejecutar_sonda_datos(model, device):
    model.eval()
    distancias_eval = np.linspace(1.2, CUTOFF, N_PUNTOS)
    feat_centro   = [0, 7, 0, 1, 0, 0, 3, 3, 0, 1, 1, 0, -0.4, 1.55, -1.0]  # N (proteína)
    feat_fijo     = [1, 8, 0, 0, 0, 0, 3, 2, 0, 0, 1, 0, -0.5, 1.52,  1.0]  # O fijo (ligando)
    feat_variable = [1, 8, 0, 0, 0, 0, 3, 2, 0, 0, 1, 0, -0.5, 1.52,  1.0]  # O variable (ligando)
    x = torch.tensor([feat_centro, feat_fijo, feat_variable], dtype=torch.float).to(device)
    molcode = x[:, 14]
    atencion_variable = []
    for d in distancias_eval:
        pos = torch.tensor([[0.0, 0.0, 0.0],
                            [1.5, 0.0, 0.0],
                            [d,   0.0, 0.0]], dtype=torch.float).to(device)
        data = Data(x=x, pos=pos,
                    lpe=torch.zeros((3, 15)).to(device),
                    batch=torch.zeros(3, dtype=torch.long).to(device))
        _, attn_weights = model(data, return_attention=True)
        if attn_weights is None:
            atencion_variable.append(0.0)
            continue
        dm = torch.cdist(pos, pos, p=2)
        r, c = torch.where(dm <= CUTOFF)
        avg = attn_weights.mean(dim=-1).view(-1)
        pair02 = (((r == 0) & (c == 2)) | ((r == 2) & (c == 0)))
        cross = molcode[r] * molcode[c] < 0
        sel = pair02 & cross
        atencion_variable.append(avg[sel].mean().item() if sel.any() else 0.0)
    return distancias_eval, np.array(atencion_variable)


# =========================================================================
# MÉTRICA: fracción de atención por zona
# =========================================================================
def fracciones_por_zona(dist, attn):
    """Devuelve (frac_esterica, frac_ventana, frac_vdw) y la suma total."""
    total = attn.sum()
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    def frac(z):
        m = (dist >= z[0]) & (dist < z[1]) if z != ZONA_VDW else (dist >= z[0]) & (dist <= z[1])
        return attn[m].sum() / total
    return frac(ZONA_ESTERICA), frac(ZONA_VENTANA), frac(ZONA_VDW), total


# =========================================================================
# MAIN
# =========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Almacenes
    curvas = {}          # (lam, seed) -> (dist, attn)
    fracciones = []      # filas: lam, seed, frac_est, frac_ven, frac_vdw, total, attn_media_ventana
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta):
                faltantes.append(ruta)
                continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta, map_location=device))
            dist, attn = ejecutar_sonda_datos(model, device)
            curvas[(lam, seed)] = (dist, attn)
            fe, fv, fw, tot = fracciones_por_zona(dist, attn)
            # atención media absoluta en la ventana (no normalizada)
            mven = attn[(dist >= ZONA_VENTANA[0]) & (dist <= ZONA_VENTANA[1])].mean()
            fracciones.append([lam, seed, fe, fv, fw, tot, mven])
            print(f"OK  λ={lam}  seed={seed:3d} | frac_ventana={fv:.3f}  "
                  f"frac_esterica={fe:.3f}  frac_vdw={fw:.3f}")

    if faltantes:
        print("\n⚠️ CHECKPOINTS NO ENCONTRADOS (revisa nombres):")
        for f in faltantes:
            print("   ", f)
    if not curvas:
        print("\n❌ No se cargó ningún modelo. Revisa CARPETA_MODELOS y los nombres.")
        return

    # ---------------- Guardar curvas punto a punto ----------------
    ruta_curvas = os.path.join(CARPETA_SALIDA, "sonda_dist_curvas.csv")
    with open(ruta_curvas, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "seed", "distancia", "atencion"])
        for (lam, seed), (dist, attn) in curvas.items():
            for d, a in zip(dist, attn):
                w.writerow([lam, seed, f"{d:.4f}", f"{a:.6f}"])
    print(f"\nCurvas guardadas: {ruta_curvas}")

    # ---------------- Guardar fracciones ----------------
    ruta_frac = os.path.join(CARPETA_SALIDA, "sonda_dist_fracciones.csv")
    with open(ruta_frac, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "seed", "frac_esterica", "frac_ventana", "frac_vdw",
                    "atencion_total", "atencion_media_ventana"])
        for row in fracciones:
            w.writerow([row[0], row[1], f"{row[2]:.6f}", f"{row[3]:.6f}",
                        f"{row[4]:.6f}", f"{row[5]:.6f}", f"{row[6]:.6f}"])
    print(f"Fracciones guardadas: {ruta_frac}")

    # ---------------- Estadística por lambda ----------------
    def col(lam, idx):
        return np.array([r[idx] for r in fracciones if r[0] == lam])

    print("\n" + "=" * 64)
    print("RESUMEN ESTADÍSTICO (media ± σ sobre semillas)")
    print("=" * 64)
    for lam in LAMBDAS:
        fe, fv, fw = col(lam, 2), col(lam, 3), col(lam, 4)
        n = len(fv)
        if n == 0:
            continue
        print(f"\nλ={lam}  (n={n})")
        print(f"  Frac. ventana [2.5-3.5] : {fv.mean():.3f} ± {fv.std(ddof=1):.3f}")
        print(f"  Frac. estérica[1.2-2.5) : {fe.mean():.3f} ± {fe.std(ddof=1):.3f}")
        print(f"  Frac. vdW    (3.5-4.5]  : {fw.mean():.3f} ± {fw.std(ddof=1):.3f}")

    # ---------------- Comparación λ=0 vs λ=0.1 (fracción en ventana) ----------------
    fv0 = {r[1]: r[3] for r in fracciones if r[0] == 0.0}
    fv1 = {r[1]: r[3] for r in fracciones if r[0] == 0.1}
    comunes = sorted(set(fv0) & set(fv1))
    if len(comunes) >= 2:
        a0 = np.array([fv0[s] for s in comunes])
        a1 = np.array([fv1[s] for s in comunes])
        t, pt = stats.ttest_rel(a1, a0)
        W, pL = stats.levene(a0, a1, center='mean')
        print("\n" + "=" * 64)
        print("COMPARACIÓN λ=0 vs λ=0.1  (fracción en ventana H-bond)")
        print("=" * 64)
        print(f"  n pareado = {len(comunes)}")
        print(f"  media λ=0   : {a0.mean():.3f} ± {a0.std(ddof=1):.3f}")
        print(f"  media λ=0.1 : {a1.mean():.3f} ± {a1.std(ddof=1):.3f}")
        print(f"  Δ (0.1-0)   : {(a1-a0).mean():+.3f}")
        print(f"  t-test pareado: t={t:.3f}, p={pt:.4f}")
        print(f"  Levene varianzas: W={W:.3f}, p={pL:.4f}")

    # ---------------- Figura: banda media ± σ por lambda ----------------
    plt.figure(figsize=(9, 6), dpi=300)
    colores = {0.0: '#8c564b', 0.1: '#1f77b4'}
    etiquetas = {0.0: 'λ = 0.0 (no physics)', 0.1: 'λ = 0.1'}
    # sombrear zonas
    plt.axvspan(*ZONA_VENTANA, color='#2ca02c', alpha=0.10, label='H-bond window (2.5–3.5 Å)')
    for lam in LAMBDAS:
        curvas_lam = [attn for (l, s), (dist, attn) in curvas.items() if l == lam]
        if not curvas_lam:
            continue
        dist_ref = next(dist for (l, s), (dist, attn) in curvas.items() if l == lam)
        M = np.vstack(curvas_lam)
        media = M.mean(axis=0)
        sd = M.std(axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(media)
        plt.plot(dist_ref, media, color=colores[lam], linewidth=3, label=etiquetas[lam])
        plt.fill_between(dist_ref, media - sd, media + sd, color=colores[lam], alpha=0.20)
    plt.xlabel("Variable atom distance $d$ (Å)", fontsize=14)
    plt.ylabel(r"Mean attention ($\alpha$) ± σ across seeds", fontsize=14)
    plt.title("Attention response to distance — mean ± σ over 8 seeds",
              fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=12)          # era 9
    plt.xticks(fontsize=12)          # añadir
    plt.yticks(fontsize=12)          # añadir
    plt.tight_layout()
    ruta_fig = os.path.join(CARPETA_SALIDA, "sonda_dist_banda_ing.png")
    plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    print(f"\nFigura guardada: {ruta_fig}")


if __name__ == "__main__":
    main()