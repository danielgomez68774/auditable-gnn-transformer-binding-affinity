"""
====================================================================
GENERADOR DE PDBs ChimeraX — SOLO 1bcu (PROTEÍNA COMPLETA), en CPU
====================================================================
Versión acotada para el caso de estudio: procesa únicamente el complejo 1bcu
con el grafo de la proteína completa (~4470 átomos), en CPU para evitar el
out-of-memory de la GPU.

Genera los 3 PDBs (ligando, proteína, hotspots) con la atención en el B-factor,
y reporta el pKd predicho y la cobertura (R_mol) de 1bcu sobre la proteína
completa.

Modelo: benchmark (Protocolo A, seed 7). Ajusta si quieres el del split.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
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
SEED = 7
COMPLEJO = "1bcu"   # el complejo del caso de estudio

# Grafo de PROTEÍNA COMPLETA de 1bcu
PATH_GRAFOS = str(config.GRAFOS_CASF_CORE285)
# Modelo benchmark (Protocolo A). Para el split, usa:
#   best_model_SIMSPLIT_seed_7_lam0p0_full.pt
PATH_MODELO = str(config.MODELS_DIR / f"best_model_REFINEDsinCore_CASFtest_seed_{SEED}_lam0p0_full.pt")
RUTA_SALIDA = str(config.OUTPUTS_DIR / "Graficas_Validaciones")

UMBRAL_TAO = 0.1               # atención considerable en el ligando (cobertura)
UMBRAL_PROTEINA_CRITICA = 0.05 # umbral para los hotspots del sitio
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

# CPU forzado para evitar out-of-memory con la proteína completa
DEVICE = torch.device('cpu')


def obtener_simbolo_elemento(num_atomico):
    mapeo = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S',
             17: 'Cl', 30: 'ZN', 35: 'Br', 53: 'I'}
    return mapeo.get(int(num_atomico), 'X')


def escribir_pdb_estricto(coords, elementos, valores_bfactor, res_name, ruta_guardado):
    with open(ruta_guardado, 'w') as f:
        for idx in range(len(coords)):
            x, y, z = coords[idx]
            elem = elementos[idx]
            val = valores_bfactor[idx]
            atom_name = f"{elem}{idx+1}"
            if len(atom_name) > 4:
                atom_name = atom_name[:4]
            linea = (
                f"ATOM  {idx+1:5d}  {atom_name:<4}{res_name:<4}A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00{val:6.2f}           {elem:>2}\n"
            )
            f.write(linea)
        f.write("END\n")


def main():
    ruta_grafo = os.path.join(PATH_GRAFOS, f"{COMPLEJO}.pt")
    if not os.path.exists(ruta_grafo):
        print(f"❌ No existe el grafo: {ruta_grafo}")
        return
    if not os.path.exists(PATH_MODELO):
        print(f"❌ No existe el modelo: {PATH_MODELO}")
        return

    print(f"Procesando SOLO {COMPLEJO} (proteína completa) en {DEVICE}")
    print(f"  Grafo: {ruta_grafo}")
    print(f"  Modelo: {PATH_MODELO}")

    model = BindingAffinityModel(**MODEL_PARAMS).to(DEVICE)
    model.load_state_dict(torch.load(PATH_MODELO, map_location=DEVICE))
    model.eval()

    os.makedirs(RUTA_SALIDA, exist_ok=True)
    carpeta_pdbs = os.path.join(RUTA_SALIDA, f"PDB_XAI_{COMPLEJO}_full_seed_{SEED}")
    os.makedirs(carpeta_pdbs, exist_ok=True)

    data = torch.load(ruta_grafo, map_location=DEVICE, weights_only=False)
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(DEVICE)
    data = data.to(DEVICE)

    print(f"  Átomos totales: {data.x.shape[0]}")
    print("  Ejecutando inferencia (puede tardar en CPU)...")

    with torch.no_grad():
        pkd_predicho, attn_weights = model(data, return_attention=True)
    if attn_weights is None:
        print("❌ El modelo no devolvió atención."); return
    if attn_weights.dim() > 1:
        attn_weights = attn_weights.mean(dim=-1).view(-1)

    dm = torch.cdist(data.pos, data.pos, p=2)
    mask = (dm <= model.cutoff)
    if data.batch is not None:
        mask = mask & (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
    row, col = torch.where(mask)
    src, dst = row.cpu().numpy(), col.cpu().numpy()
    attn_np = attn_weights.detach().cpu().numpy()
    molcodes = data.x[:, -1].cpu().numpy()

    indices_ligando = np.where(molcodes > 0)[0]
    indices_proteina = np.where(molcodes < 0)[0]

    atn_nodo = np.zeros(len(molcodes))
    for i in range(len(src)):
        u, v = src[i], dst[i]
        if molcodes[u] * molcodes[v] < 0:
            atn_nodo[u] += attn_np[i]
            atn_nodo[v] += attn_np[i]

    pos_ligando = data.pos[indices_ligando].cpu().numpy()
    pos_proteina = data.pos[indices_proteina].cpu().numpy()
    elementos_ligando = [obtener_simbolo_elemento(data.x[idx, 1].item()) for idx in indices_ligando]
    atenciones_ligando = [atn_nodo[idx] for idx in indices_ligando]
    elementos_proteina = [obtener_simbolo_elemento(data.x[idx, 1].item()) for idx in indices_proteina]
    atenciones_proteina = [atn_nodo[idx] for idx in indices_proteina]

    n_total = len(indices_ligando)
    n_activos = int(np.sum(np.array(atenciones_ligando) >= UMBRAL_TAO))
    r_mol = n_activos / (n_total + 1e-9)

    coords_h, elem_h, attn_h = [], [], []
    for il in range(len(indices_proteina)):
        if atenciones_proteina[il] >= UMBRAL_PROTEINA_CRITICA:
            coords_h.append(pos_proteina[il])
            elem_h.append(elementos_proteina[il])
            attn_h.append(atenciones_proteina[il])

    escribir_pdb_estricto(pos_ligando, elementos_ligando, atenciones_ligando,
                          "LIG", os.path.join(carpeta_pdbs, f"LIGAND_{COMPLEJO}.pdb"))
    escribir_pdb_estricto(pos_proteina, elementos_proteina, atenciones_proteina,
                          "PRO", os.path.join(carpeta_pdbs, f"PROTEIN_{COMPLEJO}.pdb"))
    if len(coords_h) > 0:
        escribir_pdb_estricto(coords_h, elem_h, attn_h,
                              "REC", os.path.join(carpeta_pdbs, f"SITE_{COMPLEJO}.pdb"))

    print("\n" + "=" * 60)
    print(f"RESULTADO — {COMPLEJO} (proteína completa)")
    print("=" * 60)
    print(f"  pKd predicho:          {pkd_predicho.item():.4f}")
    print(f"  Cobertura R_mol:       {r_mol*100:.2f}%  ({n_activos}/{n_total} átomos del ligando)")
    print(f"  Átomos del ligando:    {n_total}")
    print(f"  Átomos del sitio (hotspots): {len(coords_h)}")
    print(f"  PDBs guardados en:     {carpeta_pdbs}")


if __name__ == "__main__":
    main()