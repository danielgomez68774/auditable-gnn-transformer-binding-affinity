"""
====================================================================
POBLACIÓN DE CONTACTOS INTERFACIALES POR ZONA DE DISTANCIA
====================================================================
Línea base POBLACIONAL para contextualizar la distribución de atención.

Idea (hipótesis del autor): en un bolsillo proteico real hay muchos más
contactos interfaciales a corta distancia que en la ventana de puente de
hidrógeno. Un mecanismo de atención que reparte peso sobre contactos tenderá
a poner más masa donde hay más contactos, por conteo, no por preferencia
física. Este script CUANTIFICA esa asimetría poblacional, para poder comparar:

    fracción de CONTACTOS por zona   (población, este script)
        vs
    fracción de ATENCIÓN por zona    (la sonda de distancia / RL)

Si la atención sigue de cerca la población -> "distribución equilibrada".
Si la atención en la ventana EXCEDE su población -> preferencia física genuina.

Este análisis es PURAMENTE GEOMÉTRICO: cuenta contactos por distancia sobre
los grafos. NO usa el modelo. Es rápido y determinista (no depende de semilla).

Zonas (a priori, base biofísica):
    Estérica  : [1.2, 2.5)  Å
    Ventana   : [2.5, 3.5]  Å   (puente de hidrógeno)
    vdW/larga : (3.5, 4.5]  Å
"""

import os
import csv
import numpy as np
import torch

# =========================================================================
# CONFIGURACIÓN
# =========================================================================
# Carpetas de grafos (.pt). Analizamos varios conjuntos para robustez.
# --- repo-root bootstrap: make `config` importable from any folder ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config

CARPETAS = {
    "CASF_core":   str(config.GRAFOS_CASF_CORE285),
    "Refined":     str(config.GRAFOS_REFINED),
    # agrega CSAR / CrossDocked si quieres:
    # "CSAR":      str(config.DATA_ROOT / "Grafos_CSAR"),
    # "CrossDocked": str(config.DATA_ROOT / "Grafos_CrossDocked"),
}
CARPETA_SALIDA = str(config.OUTPUTS_DIR / "RBF")
CUTOFF = 4.5
DMIN = 1.2  # límite inferior coherente con la sonda

ZONA_ESTERICA = (1.2, 2.5)
ZONA_VENTANA  = (2.5, 3.5)
ZONA_VDW      = (3.5, 4.5)

# Muestreo: cuántos complejos por carpeta (None = todos)
MAX_COMPLEJOS = None


def contactos_interfaciales_distancias(data):
    """Devuelve las distancias de todos los contactos interfaciales (molcode
    de signos opuestos) dentro del cutoff, excluyendo self-loops."""
    pos = data.pos
    molcode = data.x[:, -1]
    dm = torch.cdist(pos, pos, p=2)
    n = pos.shape[0]
    # máscara: dentro de cutoff, por encima de DMIN, sin diagonal, intermolecular
    mask = (dm <= CUTOFF) & (dm >= DMIN)
    mask.fill_diagonal_(False)
    inter = (molcode.unsqueeze(1) * molcode.unsqueeze(0)) < 0
    mask = mask & inter
    # solo triangular superior para no contar cada contacto dos veces
    iu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    mask = mask & iu
    return dm[mask].cpu().numpy()


def fracciones_zona(distancias):
    total = len(distancias)
    if total == 0:
        return 0.0, 0.0, 0.0, 0
    d = np.asarray(distancias)
    fe = np.sum((d >= ZONA_ESTERICA[0]) & (d < ZONA_ESTERICA[1])) / total
    fv = np.sum((d >= ZONA_VENTANA[0]) & (d < ZONA_VENTANA[1])) / total
    fw = np.sum((d >= ZONA_VDW[0]) & (d <= ZONA_VDW[1])) / total
    return fe, fv, fw, total


def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    resumen = []
    todas_las_dist_por_carpeta = {}

    for nombre, carpeta in CARPETAS.items():
        if not os.path.isdir(carpeta):
            print(f"⚠️ Carpeta no encontrada, se omite: {carpeta}")
            continue
        archivos = [f for f in os.listdir(carpeta) if f.endswith(".pt")]
        if MAX_COMPLEJOS:
            archivos = archivos[:MAX_COMPLEJOS]
        print(f"\n=== {nombre}: {len(archivos)} complejos ===")

        dist_acumuladas = []          # todas las distancias de contacto del conjunto
        fracs_por_complejo = []       # fracciones por complejo (para media±σ)

        for i, fn in enumerate(archivos):
            try:
                data = torch.load(os.path.join(carpeta, fn),
                                  map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"   (saltando {fn}: {e})")
                continue
            if data is None or not hasattr(data, 'pos') or not hasattr(data, 'x'):
                continue
            dists = contactos_interfaciales_distancias(data)
            if len(dists) == 0:
                continue
            dist_acumuladas.append(dists)
            fe, fv, fw, tot = fracciones_zona(dists)
            fracs_por_complejo.append((fe, fv, fw, tot))

        if not dist_acumuladas:
            print("   (sin contactos interfaciales)")
            continue

        todas = np.concatenate(dist_acumuladas)
        todas_las_dist_por_carpeta[nombre] = todas

        # (A) Fracciones AGREGADAS (pool de todos los contactos del conjunto)
        fe_ag, fv_ag, fw_ag, tot_ag = fracciones_zona(todas)

        # (B) Fracciones por complejo -> media ± σ (trata cada complejo por igual)
        F = np.array([(fe, fv, fw) for (fe, fv, fw, _) in fracs_por_complejo])
        media = F.mean(axis=0)
        sd = F.std(axis=0, ddof=1) if F.shape[0] > 1 else np.zeros(3)

        print(f"   Contactos interfaciales totales: {tot_ag:,}")
        print(f"   [AGREGADO] estérica={fe_ag:.3f}  ventana={fv_ag:.3f}  vdW={fw_ag:.3f}")
        print(f"   [POR COMPLEJO media±σ] "
              f"estérica={media[0]:.3f}±{sd[0]:.3f}  "
              f"ventana={media[1]:.3f}±{sd[1]:.3f}  "
              f"vdW={media[2]:.3f}±{sd[2]:.3f}")

        resumen.append({
            "conjunto": nombre,
            "n_complejos": F.shape[0],
            "n_contactos": tot_ag,
            "frac_esterica_agregada": fe_ag,
            "frac_ventana_agregada": fv_ag,
            "frac_vdw_agregada": fw_ag,
            "frac_esterica_media": media[0], "frac_esterica_sd": sd[0],
            "frac_ventana_media": media[1], "frac_ventana_sd": sd[1],
            "frac_vdw_media": media[2], "frac_vdw_sd": sd[2],
        })

    # Guardar resumen
    if resumen:
        ruta = os.path.join(CARPETA_SALIDA, "poblacion_contactos_por_zona.csv")
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(resumen[0].keys()))
            w.writeheader()
            for r in resumen:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        print(f"\nResumen guardado: {ruta}")

    # ---- Comparación directa: población vs atención ----
    # (Pega aquí los valores de atención de la sonda de distancia para el contraste)
    print("\n" + "=" * 64)
    print("PARA EL CONTRASTE (pega junto a los de la sonda de atención):")
    print("=" * 64)
    print("Fracción de ATENCIÓN por zona (de la sonda, λ=0):")
    print("   estérica≈0.393  ventana≈0.308  vdW≈0.299")
    print("Compara con las fracciones de CONTACTOS (POBLACIÓN) de arriba.")
    print("Si atención_ventana > población_ventana  -> el modelo SOBRE-atiende la ventana.")
    print("Si son parecidas -> el modelo sigue la población (distribución equilibrada).")


if __name__ == "__main__":
    main()