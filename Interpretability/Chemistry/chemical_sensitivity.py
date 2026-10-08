"""
==========================================================================
SONDA DE ESPECIFICIDAD QUÍMICA (PERTURBACIÓN) — MULTI-MODELO
==========================================================================
Perturba propiedades químicas de los átomos del LIGANDO de forma controlada y
mide el cambio en la afinidad predicha (Delta pKd = pred_perturbada - original),
promediado sobre los 178 complejos reales, sobre los 8 modelos de lambda=0 y
los 8 de lambda=0.1.

ÍNDICES CORREGIDOS según el mapa real de data.x (confirmado):
  0=categoria, 1=num_atomico, 2=en_anillo, 3=numH, 4=carga_formal,
  5=aromatico, 6=hibridacion, 7=val_explicita, 8=val_implicita,
  9=donador, 10=aceptor, 11=hidrofobico, 12=gasteiger, 13=vdw_radius, 14=molcode

Perturbaciones (todas sobre átomos de ligando, molcode > 0):
  - Tipo de átomo -> C   (col 1 -> 6): cambia la identidad química
  - Valencia fija        (col 7 -> 4): satura la valencia explícita
  - Anular aromaticidad  (col 5 -> 0)
  - Anular donador       (col 9 -> 0): quita capacidad de donar H
  - Anular aceptor       (col 10 -> 0): quita capacidad de aceptar H
  - Anular hidrofobicidad(col 11 -> 0)
  - Neutralizar cargas   (col 12 -> 0): Gasteiger a cero
  - Inversión molcode    (col 14 -> -1): pasa el ligando por proteína

Régimen REAL. Es la evidencia de especificidad química del modelo.
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
CARPETA_GRAFOS  = str(config.GRAFOS_CASF_CORE285)
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "Quimica")
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0, 0.1]
MAX_COMPLEJOS = None
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

# (columna, nuevo_valor) — índices CORREGIDOS
PERTURBACIONES = {
    'Atom_type_to_C':       (1, 6.0),
    'Fixed_valence':        (7, 4.0),
    'Nullify_aromaticity':  (5, 0.0),
    'Nullify_donor':        (9, 0.0),
    'Nullify_acceptor':     (10, 0.0),
    'Nullify_hydrophobic':  (11, 0.0),
    'Neutralize_charges':   (12, 0.0),
    'Molcode_inversion':    (14, -1.0),
}


def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_{lam_tag}_full.pt"


@torch.no_grad()
def delta_por_complejo(model, data, device):
    """Devuelve {perturbacion: delta_pKd} para un complejo."""
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(device)
    mask_lig = (data.x[:, 14] > 0)
    if not torch.any(mask_lig):
        return None
    pred_orig = model(data).item()
    deltas = {}
    for nombre, (col, val) in PERTURBACIONES.items():
        dp = data.clone()
        dp.x[mask_lig, col] = val
        pred_pert = model(dp).item()
        deltas[nombre] = pred_pert - pred_orig
    return deltas


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    archivos = [f for f in os.listdir(CARPETA_GRAFOS) if f.endswith('.pt')]
    if MAX_COMPLEJOS:
        archivos = archivos[:MAX_COMPLEJOS]
    print(f"Complejos: {len(archivos)}")

    # delta_medio[(lam, seed)][perturbacion] = media sobre complejos
    delta_medio = {}
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta):
                faltantes.append(ruta); continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta, map_location=device))
            model.eval()

            acum = {k: [] for k in PERTURBACIONES}
            for fn in archivos:
                try:
                    data = torch.load(os.path.join(CARPETA_GRAFOS, fn),
                                      map_location='cpu', weights_only=False)
                except Exception:
                    continue
                if data is None or not hasattr(data, 'x'):
                    continue
                import copy
                d = delta_por_complejo(model, copy.deepcopy(data), device)
                if d is None:
                    continue
                for k in PERTURBACIONES:
                    acum[k].append(d[k])
            delta_medio[(lam, seed)] = {k: np.mean(v) for k, v in acum.items() if v}
            resumen = "  ".join(f"{k[:10]}={np.mean(v):+.2f}" for k, v in acum.items() if v)
            print(f"OK λ={lam} seed={seed:3d} | {resumen}")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)
    if not delta_medio:
        print("❌ No se cargó ningún modelo."); return

    # ---- Tabla: Delta pKd por perturbación (media ± σ sobre semillas) ----
    print("\n" + "=" * 70)
    print("Δ pKd POR PERTURBACIÓN (media ± σ sobre semillas)")
    print("=" * 70)
    print(f"{'Perturbación':<22} | {'λ=0':>16} | {'λ=0.1':>16} | {'p (λ0 vs λ0.1)':>14}")
    print("-" * 70)
    tabla = []
    for k in PERTURBACIONES:
        v0 = np.array([delta_medio[(0.0, s)][k] for s in SEEDS
                       if (0.0, s) in delta_medio and k in delta_medio[(0.0, s)]])
        v1 = np.array([delta_medio[(0.1, s)][k] for s in SEEDS
                       if (0.1, s) in delta_medio and k in delta_medio[(0.1, s)]])
        if len(v0) == 0 or len(v1) == 0:
            continue
        p = np.nan
        if len(v0) == len(v1) and len(v0) >= 2:
            _, p = stats.ttest_rel(v1, v0)
        print(f"{k:<22} | {v0.mean():>7.3f} ± {v0.std(ddof=1):<6.3f} | "
              f"{v1.mean():>7.3f} ± {v1.std(ddof=1):<6.3f} | {p:>14.4f}")
        tabla.append([k, v0.mean(), v0.std(ddof=1), v1.mean(), v1.std(ddof=1), p])

    # ---- Guardar ----
    ruta = os.path.join(CARPETA_SALIDA, "sensibilidad_quimica_multimodelo_ing.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perturbacion", "delta_l0_media", "delta_l0_sd",
                    "delta_l01_media", "delta_l01_sd", "p_l0_vs_l01"])
        for row in tabla:
            w.writerow([row[0]] + [f"{x:.6f}" if not np.isnan(x) else "nan" for x in row[1:]])
    print(f"\nGuardado: {ruta}")

    # ---- Figura: barras Delta pKd por perturbación (λ=0) ----
    try:
        import matplotlib.pyplot as plt
        nombres = [r[0] for r in tabla]
        medias = [r[1] for r in tabla]
        errs = [r[2] for r in tabla]
        orden = np.argsort(medias)  # de más negativo a más positivo
        nombres = [nombres[i] for i in orden]
        medias = [medias[i] for i in orden]
        errs = [errs[i] for i in orden]
        colores = ['#c0392b' if m < 0 else '#27ae60' for m in medias]
        plt.figure(figsize=(10, 6), dpi=300)
        plt.barh(nombres, medias, xerr=errs, color=colores, alpha=0.8, capsize=4)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.xlabel(r"$\Delta pK_d$ (perturbed - original), mean ± σ over seeds", fontsize=14)
        plt.title("Sensitivity of the prediction to chemical perturbations of the ligand\n"
                  "(real regime, 178 complexes, λ=0)", fontsize=11, fontweight='bold', pad=15)
        plt.grid(True, alpha=0.2, axis='x')
        plt.xticks(fontsize=12)          # añadir
        plt.yticks(fontsize=12)          # añadir
        plt.tight_layout()
        ruta_fig = os.path.join(CARPETA_SALIDA, "sensibilidad_quimica_barras_ing.png")
        plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {ruta_fig}")
    except Exception as e:
        print(f"(No se pudo generar la figura: {e})")


if __name__ == "__main__":
    main()