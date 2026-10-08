"""
==========================================================================
ANÁLISIS PAREADO: ATENCIÓN vs POBLACIÓN (mismos contactos reales)
  + COMPOSICIÓN QUÍMICA POR ZONA
==========================================================================
Blinda el argumento de "el modelo no sigue la población": aquí atención y
población se miden sobre EXACTAMENTE los mismos contactos interfaciales de
los complejos reales (no sobre un sistema sintético). Comparación pareada
perfecta.

Además, para responder "si no es población, ¿qué propiedad química explica
la preferencia?", cuantifica la COMPOSICIÓN de cada zona: qué fracción de los
contactos de cada zona involucra carbono, nitrógeno, oxígeno, etc.

Idea del autor: quizá la zona vdW (larga, muy poblada, poco atendida) está
dominada por carbonos (contactos hidrofóbicos débiles), mientras la ventana y
la zona corta están enriquecidas en N/O (polares, H-bond). Eso explicaría por
qué el modelo suprime la zona vdW: no por distancia, sino por química.

Para cada zona reporta, sobre los contactos interfaciales reales:
  - fracción de POBLACIÓN de contactos
  - fracción de ATENCIÓN del modelo (promediada sobre los modelos)
  - composición atómica (qué elementos participan en los contactos)

CONFIGURACIÓN QUÍMICA: ajusta IDX_NUM_ATOMICO al índice real de tu número
atómico en data.x (parece ser 1, con N=7, O=8; el carbono sería 6).
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

# =========================================================================
# CONFIGURACIÓN
# =========================================================================
CARPETA_MODELOS = str(config.MODELS_DIR)
CARPETA_GRAFOS  = str(config.GRAFOS_CASF_CORE285)  # complejos reales
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "RBF")
CUTOFF = 4.5
DMIN = 1.2
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0, 0.1]
MAX_COMPLEJOS = None      # None = todos; pon un número para prueba rápida
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

ZONA_ESTERICA = (1.2, 2.5)
ZONA_VENTANA  = (2.5, 3.5)
ZONA_VDW      = (3.5, 4.5)

# ---- CONFIGURACIÓN QUÍMICA (CONFIRMADA por el autor) ----
# data.x columnas: 0=categoria,1=num_atomico,2=en_anillo,3=numH,4=carga_formal,
# 5=aromatico,6=hibridacion,7=val_explicita,8=val_implicita,9=DONADOR,
# 10=ACEPTOR,11=HIDROFOBICO,12=gasteiger,13=vdw_radius,14=molcode
IDX_NUM_ATOMICO = 1
IDX_DONADOR     = 9
IDX_ACEPTOR     = 10
IDX_HIDROFOBICO = 11
ELEMENTOS = {6: "C", 7: "N", 8: "O", 15: "P", 16: "S", 9: "F",
             17: "Cl", 35: "Br", 53: "I"}


def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_{lam_tag}_full.pt"


def zona_de(d):
    if ZONA_ESTERICA[0] <= d < ZONA_ESTERICA[1]: return 0
    if ZONA_VENTANA[0]  <= d < ZONA_VENTANA[1]:  return 1
    if ZONA_VDW[0]      <= d <= ZONA_VDW[1]:     return 2
    return -1


@torch.no_grad()
def atencion_y_contactos_reales(model, data, device):
    """Devuelve, para los contactos interfaciales reales del complejo:
       distancias, atención del modelo, números atómicos de ambos extremos,
       y flags químicos (donador/aceptor/hidrofóbico) de ambos extremos."""
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(device)
    out = model(data, return_attention=True)
    attn = out[1] if isinstance(out, tuple) else None
    pos = data.pos
    molcode = data.x[:, -1]
    znum = data.x[:, IDX_NUM_ATOMICO]
    don = data.x[:, IDX_DONADOR]
    acc = data.x[:, IDX_ACEPTOR]
    hyd = data.x[:, IDX_HIDROFOBICO]
    dm = torch.cdist(pos, pos, p=2)

    mask = (dm <= CUTOFF)
    if data.batch is not None:
        bm = (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
        mask = mask & bm
    row, col = torch.where(mask)
    dists = dm[row, col]
    inter = (molcode[row] * molcode[col]) < 0
    valid = inter & (dists >= DMIN) & (row != col)

    row, col, dists = row[valid], col[valid], dists[valid]
    if attn is not None:
        a = attn.mean(dim=-1).view(-1)[valid]
    else:
        a = torch.zeros_like(dists)
    zi, zj = znum[row], znum[col]
    # flags de contacto: un contacto es "H-bond potencial" si un extremo dona y
    # el otro acepta; es "hidrofóbico" si ambos extremos son hidrofóbicos.
    hbond = ((don[row] * acc[col]) + (acc[row] * don[col]) > 0).cpu().numpy()
    hidro = ((hyd[row] * hyd[col]) > 0).cpu().numpy()
    return (dists.cpu().numpy(), a.cpu().numpy(),
            zi.cpu().numpy().astype(int), zj.cpu().numpy().astype(int),
            hbond, hidro)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    archivos = [f for f in os.listdir(CARPETA_GRAFOS) if f.endswith('.pt')]
    if MAX_COMPLEJOS:
        archivos = archivos[:MAX_COMPLEJOS]
    print(f"Complejos: {len(archivos)}")

    # Cargamos un modelo por (lam, seed) y acumulamos atención por zona.
    # Población (contactos y química) se calcula UNA vez (no depende del modelo).
    poblacion_zona = np.zeros(3, dtype=np.int64)
    quim_zona = {z: {} for z in range(3)}   # composición por número atómico
    # nuevos: fracción de contactos H-bond e hidrofóbicos por zona
    hbond_zona = np.zeros(3, dtype=np.int64)
    hidro_zona = np.zeros(3, dtype=np.int64)
    poblacion_calculada = False

    atencion_por_lam = {lam: [] for lam in LAMBDAS}
    faltantes = []

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta):
                faltantes.append(ruta); continue
            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta, map_location=device))
            model.eval()

            attn_zona = np.zeros(3)
            for fn in archivos:
                try:
                    data = torch.load(os.path.join(CARPETA_GRAFOS, fn),
                                      map_location='cpu', weights_only=False)
                except Exception:
                    continue
                if data is None or not hasattr(data, 'pos'):
                    continue
                dists, a, zi, zj, hbond, hidro = atencion_y_contactos_reales(model, data, device)
                for k in range(len(dists)):
                    z = zona_de(dists[k])
                    if z < 0:
                        continue
                    attn_zona[z] += a[k]
                    if not poblacion_calculada:
                        poblacion_zona[z] += 1
                        for e in (zi[k], zj[k]):
                            quim_zona[z][e] = quim_zona[z].get(e, 0) + 1
                        if hbond[k]:
                            hbond_zona[z] += 1
                        if hidro[k]:
                            hidro_zona[z] += 1
            poblacion_calculada = True
            tot = attn_zona.sum()
            frac = attn_zona / tot if tot > 0 else attn_zona
            atencion_por_lam[lam].append(frac)
            print(f"OK λ={lam} seed={seed:3d} | atención por zona "
                  f"[est/ven/vdw] = {frac[0]:.3f}/{frac[1]:.3f}/{frac[2]:.3f}")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)
    if not any(atencion_por_lam.values()):
        print("❌ No se cargó ningún modelo."); return

    # ---- Población (fracciones) ----
    pob_frac = poblacion_zona / poblacion_zona.sum()
    print("\n" + "=" * 64)
    print("POBLACIÓN de contactos por zona (mismos contactos reales)")
    print("=" * 64)
    print(f"  estérica={pob_frac[0]:.3f}  ventana={pob_frac[1]:.3f}  vdW={pob_frac[2]:.3f}")

    # ---- Atención media sobre semillas, por lambda ----
    print("\n" + "=" * 64)
    print("ATENCIÓN por zona (media ± σ sobre semillas) y RATIO atención/población")
    print("=" * 64)
    for lam in LAMBDAS:
        if not atencion_por_lam[lam]:
            continue
        A = np.vstack(atencion_por_lam[lam])
        m, s = A.mean(axis=0), A.std(axis=0, ddof=1)
        ratio = m / pob_frac
        print(f"\nλ={lam}")
        for zi_, nom in enumerate(["estérica", "ventana ", "vdW     "]):
            print(f"  {nom}: atención={m[zi_]:.3f}±{s[zi_]:.3f} | "
                  f"población={pob_frac[zi_]:.3f} | ratio={ratio[zi_]:.2f}x")

    # ---- Composición química por zona ----
    print("\n" + "=" * 64)
    print("COMPOSICIÓN QUÍMICA por zona (fracción de extremos de contacto por elemento)")
    print("=" * 64)
    for z, nom in [(0, "estérica"), (1, "ventana"), (2, "vdW")]:
        total = sum(quim_zona[z].values())
        if total == 0:
            continue
        print(f"\nZona {nom}:")
        for e in sorted(quim_zona[z], key=lambda k: -quim_zona[z][k]):
            simb = ELEMENTOS.get(e, f"Z{e}")
            print(f"   {simb:3s}: {quim_zona[z][e]/total:.3f}")

    # ---- Naturaleza del contacto por zona (H-bond vs hidrofóbico) ----
    print("\n" + "=" * 64)
    print("NATURALEZA DEL CONTACTO por zona (fracción de contactos)")
    print("=" * 64)
    print("  H-bond potencial = un extremo dona y el otro acepta")
    print("  Hidrofóbico = ambos extremos hidrofóbicos")
    for z, nom in [(0, "estérica"), (1, "ventana "), (2, "vdW     ")]:
        n = poblacion_zona[z]
        if n == 0:
            continue
        print(f"  {nom}: H-bond={hbond_zona[z]/n:.3f}  hidrofóbico={hidro_zona[z]/n:.3f}")

    # ---- Guardar todo ----
    ruta = os.path.join(CARPETA_SALIDA, "pareado_atencion_poblacion_quimica.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metrica", "zona", "valor"])
        for zi_, nom in enumerate(["esterica", "ventana", "vdw"]):
            w.writerow(["poblacion_frac", nom, f"{pob_frac[zi_]:.6f}"])
        for lam in LAMBDAS:
            if not atencion_por_lam[lam]:
                continue
            A = np.vstack(atencion_por_lam[lam]); m = A.mean(axis=0)
            for zi_, nom in enumerate(["esterica", "ventana", "vdw"]):
                w.writerow([f"atencion_frac_lam{lam}", nom, f"{m[zi_]:.6f}"])
        for z, nom in [(0, "esterica"), (1, "ventana"), (2, "vdw")]:
            total = sum(quim_zona[z].values())
            for e in sorted(quim_zona[z]):
                simb = ELEMENTOS.get(e, f"Z{e}")
                w.writerow([f"quimica_{simb}", nom,
                            f"{quim_zona[z][e]/total:.6f}" if total else "0"])
    print(f"\nGuardado: {ruta}")


if __name__ == "__main__":
    main()