#!/usr/bin/env python3
"""
====================================================================
SPLIT POR SIMILITUD (tres bandas) DESDE CLUSTERS DE MMseqs2
====================================================================
Toma el clusters.tsv de MMseqs2 (identidad 30%) y reparte los complejos en
train / val / test asignando CLUSTERS ENTEROS a cada conjunto, de modo que
ningún cluster se comparta entre conjuntos (cero fuga por similitud).

Estrategia B (test diverso, no dominado por un target):
  - Los clusters GRANDES (> UMBRAL_GRANDE miembros) van a TRAIN. Esto evita
    que un target muy representado domine el test, y aprovecha esos complejos
    abundantes para entrenar.
  - El resto de clusters (pequeños/medianos) se barajan y se reparten:
    primero se llena TEST hasta ~TEST_FRAC, luego VAL hasta ~VAL_FRAC, y el
    remanente va a TRAIN.

Objetivo de proporciones (sobre el total de complejos):
  train ~70% | val ~10% | test ~20%

NO destructivo: solo lee clusters.tsv y escribe train.txt, val.txt, test.txt.
No toca los grafos ni los PDB.

Uso:
    python split_por_similitud.py
"""

import os
import random

# =========================================================================
# CONFIGURACIÓN
# =========================================================================
# --- repo-root bootstrap: make `config` importable from any folder ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config

DIR_SPLIT = str(config.SPLIT_DIR)
CLUSTERS_TSV = os.path.join(DIR_SPLIT, "clusters.tsv")

TEST_FRAC = 0.20     # fracción objetivo para test
VAL_FRAC  = 0.10     # fracción objetivo para val
# (train recibe el resto, ~0.70)

UMBRAL_GRANDE = 30   # clusters con > este número de miembros van a train
SEED = 42            # reproducibilidad del reparto


def leer_clusters(path):
    """Devuelve dict: representante -> lista de miembros."""
    clusters = {}
    with open(path) as f:
        for linea in f:
            partes = linea.rstrip("\n").split("\t")
            if len(partes) < 2:
                continue
            rep, miembro = partes[0], partes[1]
            clusters.setdefault(rep, []).append(miembro)
    return clusters


def main():
    if not os.path.exists(CLUSTERS_TSV):
        print(f"❌ No existe {CLUSTERS_TSV}")
        return

    clusters = leer_clusters(CLUSTERS_TSV)
    total_complejos = sum(len(v) for v in clusters.values())
    print(f"Clusters: {len(clusters)}  |  Complejos: {total_complejos}")

    # Separar clusters grandes (a train) del resto
    grandes = {r: m for r, m in clusters.items() if len(m) > UMBRAL_GRANDE}
    resto   = {r: m for r, m in clusters.items() if len(m) <= UMBRAL_GRANDE}

    n_grandes_complejos = sum(len(m) for m in grandes.values())
    print(f"Clusters grandes (>{UMBRAL_GRANDE} miembros): {len(grandes)} "
          f"({n_grandes_complejos} complejos) -> TRAIN")
    print(f"Clusters restantes: {len(resto)} "
          f"({total_complejos - n_grandes_complejos} complejos) -> repartir")

    # Objetivos absolutos
    objetivo_test = int(round(TEST_FRAC * total_complejos))
    objetivo_val  = int(round(VAL_FRAC * total_complejos))

    # Barajar los clusters del resto de forma reproducible
    reps_resto = list(resto.keys())
    random.seed(SEED)
    random.shuffle(reps_resto)

    train_ids, val_ids, test_ids = [], [], []

    # Los grandes -> train
    for r, m in grandes.items():
        train_ids.extend(m)

    # Repartir el resto: primero test, luego val, luego train
    for r in reps_resto:
        m = resto[r]
        if len(test_ids) < objetivo_test:
            test_ids.extend(m)
        elif len(val_ids) < objetivo_val:
            val_ids.extend(m)
        else:
            train_ids.extend(m)

    # ---- Verificación de fuga: ningún complejo en dos conjuntos ----
    s_train, s_val, s_test = set(train_ids), set(val_ids), set(test_ids)
    fuga_tv = s_train & s_val
    fuga_tt = s_train & s_test
    fuga_vt = s_val & s_test
    assert not fuga_tv, f"FUGA train-val: {fuga_tv}"
    assert not fuga_tt, f"FUGA train-test: {fuga_tt}"
    assert not fuga_vt, f"FUGA val-test: {fuga_vt}"

    # Verificación de cobertura: todos los complejos asignados
    total_asignado = len(s_train) + len(s_val) + len(s_test)
    assert total_asignado == total_complejos, \
        f"Faltan complejos: {total_asignado} vs {total_complejos}"

    # ---- Verificación de fuga por CLUSTER (lo esencial del split) ----
    # Cada cluster debe estar entero en un solo conjunto.
    def conjunto_de(cid):
        if cid in s_train: return 'train'
        if cid in s_val: return 'val'
        if cid in s_test: return 'test'
        return None
    clusters_mixtos = 0
    for r, m in clusters.items():
        conjuntos = set(conjunto_de(c) for c in m)
        if len(conjuntos) > 1:
            clusters_mixtos += 1
    assert clusters_mixtos == 0, \
        f"❌ {clusters_mixtos} clusters están repartidos entre conjuntos (FUGA)"

    # ---- Guardar listas ----
    def guardar(nombre, ids):
        ruta = os.path.join(DIR_SPLIT, nombre)
        with open(ruta, "w") as f:
            for cid in sorted(ids):
                f.write(cid + "\n")
        return ruta

    r_train = guardar("train.txt", s_train)
    r_val   = guardar("val.txt", s_val)
    r_test  = guardar("test.txt", s_test)

    # ---- Reporte ----
    print("\n" + "=" * 60)
    print("SPLIT POR SIMILITUD (30% identidad, clusters enteros)")
    print("=" * 60)
    print(f"Train: {len(s_train):5d}  ({100*len(s_train)/total_complejos:.1f}%)")
    print(f"Val:   {len(s_val):5d}  ({100*len(s_val)/total_complejos:.1f}%)")
    print(f"Test:  {len(s_test):5d}  ({100*len(s_test)/total_complejos:.1f}%)")
    print(f"Total: {total_asignado}")
    print(f"\n✅ Cero fuga: ningún cluster compartido entre conjuntos.")
    print(f"✅ Cobertura completa: {total_asignado}/{total_complejos} complejos.")
    print(f"\nListas guardadas:")
    print(f"  {r_train}")
    print(f"  {r_val}")
    print(f"  {r_test}")


if __name__ == "__main__":
    main()
