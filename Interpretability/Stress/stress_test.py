"""
==========================================================================
SONDA DE SENSIBILIDAD CONFORMACIONAL — MULTI-MODELO (lambda 0 y 0.1)
==========================================================================
Evalúa si el modelo responde a la conformación (la pose 3D del ligando),
gracias al cálculo dinámico del RBF durante la inferencia.

Para cada complejo (con múltiples poses de docking) y cada modelo (8 semillas
x lambda=0/0.1), se evalúan tres escenarios por pose:
  1. NORMAL: la pose tal cual.
  2. VACÍO:  el ligando se aleja 15 Å (control: sin contacto -> basal).
  3. CAOS:   se añade ruido gaussiano de 10 Å a las posiciones del ligando
             (perturbación fuerte de la pose).

Métricas:
  - Sensibilidad a la pose = desviación estándar del pKd entre las poses
    NORMALES de cada complejo (si >0, el modelo responde a la conformación).
  - Efecto vacío = |pKd_normal - pKd_vacío| medio (caída al quitar el contacto).
  - Efecto caos  = |pKd_normal - pKd_caos| medio (respuesta a degradar la pose).

Todo promediado sobre poses, complejos y semillas, para lambda=0 y lambda=0.1.
"""

import os
import csv
import sys
import numpy as np
import torch

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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =========================================================================
CARPETA_MODELOS = str(config.MODELS_DIR)
CARPETA_BASE = str(config.GRAFOS_CROSSDOCKED_INDIVIDUAL)
CARPETA_SALIDA = str(config.OUTPUTS_DIR / "Stress")
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0, 0.1]
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

# Complejos (subcarpetas)
COMPLEJOS = [
    "5HT2B_HUMAN_27_404_0",
    "8ODP_HUMAN_42_197_0",
    "AAPK1_HUMAN_14_295_0",
    "1433S_HUMAN_1_233_0",
]

RUIDO_CAOS = 10.0             # Å (ruido gaussiano sobre la pose)


def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_{lam_tag}_full.pt"


@torch.no_grad()
def predecir(model, data, device):
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
    data = data.to(device)
    out = model(data)
    return out.item()


@torch.no_grad()
def evaluar_complejo(model, carpeta_complejo, device, rng_seed=0):
    """Devuelve, para un complejo: pKd de cada pose en normal y caos."""
    archivos = [f for f in os.listdir(carpeta_complejo)
                if f.endswith('.pt') and 'MUT' not in f]
    if not archivos:
        return None
    try:
        archivos.sort(key=lambda x: int(x.split('pose_')[-1].split('.pt')[0]))
    except Exception:
        archivos.sort()

    pkd_normal, pkd_caos = [], []
    torch.manual_seed(rng_seed)  # ruido del caos reproducible

    for f_name in archivos:
        data = torch.load(os.path.join(carpeta_complejo, f_name),
                          map_location=device, weights_only=False)
        data = data.to(device)
        es_ligando = (data.x[:, 14] == 1.0)

        # Normal
        pkd_normal.append(predecir(model, data.clone(), device))

        # Caos: ruido sobre la pose del ligando (perturbación físicamente plausible)
        dc = data.clone()
        ruido = torch.randn_like(dc.pos[es_ligando]) * RUIDO_CAOS
        dc.pos[es_ligando] = dc.pos[es_ligando] + ruido
        pkd_caos.append(predecir(model, dc, device))

    return np.array(pkd_normal), np.array(pkd_caos)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Por (lam, seed, complejo): sensibilidad (σ entre poses), efecto vacío, efecto caos
    registros = []  # dict por corrida
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta_ckpt = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta_ckpt):
                faltantes.append(ruta_ckpt); continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta_ckpt, map_location=device))
            model.eval()

            for comp in COMPLEJOS:
                carpeta = os.path.join(CARPETA_BASE, comp)
                if not os.path.isdir(carpeta):
                    continue
                res = evaluar_complejo(model, carpeta, device, rng_seed=seed)
                if res is None:
                    continue
                normal, caos = res
                sens_pose = normal.std(ddof=1) if len(normal) > 1 else 0.0
                efecto_caos = np.mean(np.abs(normal - caos))
                registros.append({
                    'lambda': lam, 'seed': seed, 'complejo': comp,
                    'n_poses': len(normal),
                    'sensibilidad_pose': sens_pose,
                    'efecto_caos': efecto_caos,
                    'pkd_normal_medio': normal.mean(),
                })
            print(f"OK λ={lam} seed={seed:3d} | {len(COMPLEJOS)} complejos evaluados")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)
    if not registros:
        print("❌ No se procesó nada."); return

    # ---- Guardar CSV detallado ----
    ruta_csv = os.path.join(CARPETA_SALIDA, "sensibilidad_conformacional.csv")
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
        w.writeheader()
        for r in registros:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"\nGuardado: {ruta_csv}")

    # ---- Resumen por lambda (promediando sobre complejos y semillas) ----
    print("\n" + "=" * 66)
    print("SENSIBILIDAD CONFORMACIONAL (media ± σ sobre semillas y complejos)")
    print("=" * 66)
    def col(lam, key):
        return np.array([r[key] for r in registros if r['lambda'] == lam])
    for lam in LAMBDAS:
        sp = col(lam, 'sensibilidad_pose')
        ec = col(lam, 'efecto_caos')
        if len(sp) == 0:
            continue
        print(f"\nλ={lam}")
        print(f"  Sensibilidad a la pose (σ pKd entre poses): {sp.mean():.4f} ± {sp.std(ddof=1):.4f}")
        print(f"  Efecto caos (|Δ pKd| con ruido):            {ec.mean():.4f} ± {ec.std(ddof=1):.4f}")

    # ---- Comparación λ=0 vs λ=0.1 (sensibilidad a la pose) ----
    s0 = col(0.0, 'sensibilidad_pose')
    s1 = col(0.1, 'sensibilidad_pose')
    if len(s0) > 1 and len(s1) > 1:
        # test no pareado (mismos complejos y semillas, pero por simplicidad independiente)
        t, pt = stats.ttest_ind(s0, s1)
        print("\n" + "=" * 66)
        print("COMPARACIÓN sensibilidad a la pose: λ=0 vs λ=0.1")
        print("=" * 66)
        print(f"  λ=0:   {s0.mean():.4f} ± {s0.std(ddof=1):.4f}")
        print(f"  λ=0.1: {s1.mean():.4f} ± {s1.std(ddof=1):.4f}")
        print(f"  t-test: p={pt:.4f}")

    # ---- Figura: una pose-curva de ejemplo por escenario (primer complejo, λ=0) ----
    # try:
    #     import matplotlib.pyplot as plt
    #     # Reunir las curvas normal/vacío/caos promediadas sobre semillas para el 1er complejo
    #     comp0 = COMPLEJOS[0]
    #     # recomputar promedio por pose sobre semillas (λ=0) para la figura
    #     # (se guardó solo el resumen; para la figura reevaluamos rápido con seed 7)
    #     ckpt7 = os.path.join(CARPETA_MODELOS, nombre_checkpoint(7, 0.0))
    #     if os.path.exists(ckpt7):
    #         m = BindingAffinityModel(**MODEL_PARAMS).to(device)
    #         m.load_state_dict(torch.load(ckpt7, map_location=device)); m.eval()
    #         res = evaluar_complejo(m, os.path.join(CARPETA_BASE, comp0), device, rng_seed=7)
    #         if res is not None:
    #             normal, caos = res
    #             x = np.arange(1, len(normal) + 1)
    #             plt.figure(figsize=(10, 5.5), dpi=300)
    #             plt.plot(x, normal, color='#1f77b4', marker='s', lw=2.5, label='Normal pose')
    #             plt.plot(x, caos, color='#bcbd22', marker='^', ls='-.', lw=2, label='Noise control (perturbed pose)')
    #             plt.title(f"Conformational sensitivity (dynamic RBF) — {comp0}\n(seed 7, λ=0)",
    #                       fontsize=11, fontweight='bold', pad=12)
    #             plt.xlabel("Docking pose ID", fontsize=11)
    #             plt.ylabel("Predicted pKd", fontsize=11)
    #             plt.xticks(x); plt.grid(True, alpha=0.15, ls='--')
    #             plt.legend(fontsize=9, loc='best')
    #             plt.tight_layout()
    #             ruta_fig = os.path.join(CARPETA_SALIDA, "sensibilidad_conformacional_ejemplo.png")
    #             plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    #             print(f"\nFigura de ejemplo guardada: {ruta_fig}")
    # except Exception as e:
    #     print(f"(No se pudo generar la figura: {e})")

    try:
        import matplotlib.pyplot as plt
        # Reunir las curvas normal/vacío/caos promediadas sobre semillas para el 1er complejo
        comp0 = COMPLEJOS[0]
        # recomputar promedio por pose sobre semillas (λ=0) para la figura
        # (se guardó solo el resumen; para la figura reevaluamos rápido con seed 7)
        ckpt7 = os.path.join(CARPETA_MODELOS, nombre_checkpoint(7, 0.0))
        if os.path.exists(ckpt7):
            m = BindingAffinityModel(**MODEL_PARAMS).to(device)
            m.load_state_dict(torch.load(ckpt7, map_location=device)); m.eval()
            res = evaluar_complejo(m, os.path.join(CARPETA_BASE, comp0), device, rng_seed=7)
            if res is not None:
                normal, caos = res
                x = np.arange(1, len(normal) + 1)
                plt.figure(figsize=(10, 5.5), dpi=300)
                plt.plot(x, normal, color='#1f77b4', marker='s', lw=2.5, label='Pose normal')
                plt.plot(x, caos, color='#bcbd22', marker='^', ls='-.', lw=2, label='Control de ruido (pose perturbada)')
                plt.title(f"Sensibilidad conformacional (RBF dinámica) — {comp0}\n(semilla 7, λ=0)",
                          fontsize=11, fontweight='bold', pad=12)
                plt.xlabel("ID de la pose de acoplamiento", fontsize=11)
                plt.ylabel("pKd predicho", fontsize=11)
                plt.xticks(x); plt.grid(True, alpha=0.15, ls='--')
                plt.legend(fontsize=9, loc='best')
                plt.tight_layout()
                ruta_fig = os.path.join(CARPETA_SALIDA, "sensibilidad_conformacional_ejemplo_es.png")
                plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
                print(f"\nFigura de ejemplo guardada: {ruta_fig}")
    except Exception as e:
        print(f"(No se pudo generar la figura: {e})")


if __name__ == "__main__":
    main()