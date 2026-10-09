"""
====================================================================
DIAGNÓSTICO DEL FLOW 285 -> 178 (CASF-2016 Core)
====================================================================
Recorre el Core-285 con el MISMO preprocesamiento del pipeline, pero registrando
la razón EXACTA de exclusión de cada complejo, para documentar el flujo de
exclusiones que pide la revisión (285 -> 178).

Razones de exclusión rastreadas:
  R1 - No es carpeta válida de complejo
  R2 - Falta archivo pocket o ligando
  R3 - Ligando no parseable por RDKit
  R4 - Pocket no parseable por RDKit
  R5 - Fallo al construir features/coordenadas
  R6 - LPE no calculable (grafo deficitario)
  R7 - Sin etiqueta pKd en el índice Refined

Genera:
  - Un resumen por consola con el conteo de cada razón.
  - Un CSV (flow_exclusiones_core.csv) con cada complejo excluido y su razón.

NO sobreescribe grafos: solo diagnostica (no guarda los .pt).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

RDLogger.DisableLog('rdApp.*')

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

RUTA_CASF_CORESET = str(config.CASF_CORESET_RAW)
PATH_INDEX_REFINED = str(config.INDEX_REFINED)
CSV_SALIDA = str(config.OUTPUTS_DIR / "flow_exclusiones_core.csv")


def cargar_label_map(path_index):
    df = pd.read_csv(path_index, sep=r'\s+', comment='#', header=None,
                     usecols=[0, 3], names=['pdb_id', 'pkd'])
    return {str(r['pdb_id']).strip().lower(): float(r['pkd']) for _, r in df.iterrows()}


def intentar_lpe(edge_index_cov, num_nodes, k=15):
    """Devuelve True si el LPE se calcula (converge), False si cae al except."""
    try:
        edge_index, edge_weight = get_laplacian(edge_index_cov, normalization='sym', num_nodes=num_nodes)
        L_sparse = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
        k_adj = min(k + 1, num_nodes - 1)
        if k_adj < 1:
            return False
        eig_vals, eig_vecs = sp.linalg.eigsh(L_sparse, k=k_adj, which='SA', tol=1e-3, maxiter=5000)
        return True
    except Exception:
        return False


def construir_features_coords(pocket_mol, ligand_mol):
    """Réplica mínima: intenta construir features + coords + edge covalente."""
    import torch
    try:
        # features (solo verificamos que se puedan extraer)
        def feats(mol):
            mol.UpdatePropertyCache(strict=False)
            return mol.GetNumAtoms()
        n_p = feats(pocket_mol)
        n_l = feats(ligand_mol)
        c_p = pocket_mol.GetConformer().GetPositions()
        c_l = ligand_mol.GetConformer().GetPositions()
        num_nodes = n_p + n_l
        off = n_p
        adj = []
        for b in pocket_mol.GetBonds():
            adj.append([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
            adj.append([b.GetEndAtomIdx(), b.GetBeginAtomIdx()])
        for b in ligand_mol.GetBonds():
            adj.append([b.GetBeginAtomIdx()+off, b.GetEndAtomIdx()+off])
            adj.append([b.GetEndAtomIdx()+off, b.GetBeginAtomIdx()+off])
        edge_index_cov = torch.tensor(adj, dtype=torch.long).t().contiguous()
        return num_nodes, edge_index_cov
    except Exception:
        return None, None


def main():
    label_map = cargar_label_map(PATH_INDEX_REFINED)
    print(f"Etiquetas del índice Refined: {len(label_map)}")

    razones = {
        'R1_no_carpeta': [], 'R2_falta_archivo': [], 'R3_ligando_no_parseable': [],
        'R4_pocket_no_parseable': [], 'R5_fallo_features': [],
        'R6_lpe_no_calculable': [], 'R7_sin_label': [],
    }
    validos = []
    total_carpetas = 0

    for nombre in sorted(os.listdir(RUTA_CASF_CORESET)):
        ruta = os.path.join(RUTA_CASF_CORESET, nombre)
        if not (os.path.isdir(ruta) and len(nombre) == 4):
            razones['R1_no_carpeta'].append(nombre)
            continue
        total_carpetas += 1
        pid = nombre.strip().lower()

        f_pocket = os.path.join(ruta, f"{nombre}_pocket.pdb")
        f_ligand = os.path.join(ruta, f"{nombre}_ligand.sdf")
        if not (os.path.exists(f_pocket) and os.path.exists(f_ligand)):
            razones['R2_falta_archivo'].append(pid); continue

        # Parseo RDKit
        pocket_mol = Chem.MolFromPDBFile(f_pocket, removeHs=False)
        supplier = Chem.SDMolSupplier(f_ligand, removeHs=False, sanitize=True)
        ligand_mol = supplier[0] if len(supplier) > 0 else None

        if ligand_mol is None:
            razones['R3_ligando_no_parseable'].append(pid); continue
        if pocket_mol is None:
            razones['R4_pocket_no_parseable'].append(pid); continue

        # Features + coords + edge covalente
        num_nodes, edge_index_cov = construir_features_coords(pocket_mol, ligand_mol)
        if num_nodes is None:
            razones['R5_fallo_features'].append(pid); continue

        # LPE
        if not intentar_lpe(edge_index_cov, num_nodes, k=15):
            razones['R6_lpe_no_calculable'].append(pid); continue

        # Etiqueta
        if pid not in label_map:
            razones['R7_sin_label'].append(pid); continue

        validos.append(pid)

    # ---- Resumen ----
    print("\n" + "=" * 60)
    print("FLOW DE EXCLUSIONES: CASF-2016 Core")
    print("=" * 60)
    print(f"Carpetas de complejo válidas (285 esperadas): {total_carpetas}")
    total_excluidos = 0
    for r, lista in razones.items():
        if lista:
            print(f"  {r}: {len(lista)} excluidos")
            total_excluidos += len(lista)
    print(f"\nTotal excluidos: {total_excluidos}")
    print(f"Complejos finales (grafos válidos): {len(validos)}")
    print(f"Verificación: {total_carpetas} - {total_excluidos} = {total_carpetas - total_excluidos}")

    # ---- CSV con cada excluido y su razón ----
    filas = []
    for r, lista in razones.items():
        for pid in lista:
            filas.append({'complejo': pid, 'razon_exclusion': r})
    df = pd.DataFrame(filas)
    df.to_csv(CSV_SALIDA, index=False)
    print(f"\nCSV de exclusiones guardado en: {CSV_SALIDA}")
    print(f"  ({len(filas)} complejos excluidos registrados)")


if __name__ == "__main__":
    main()