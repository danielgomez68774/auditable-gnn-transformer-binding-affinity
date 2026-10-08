"""
====================================================================
GENERADOR DE PDBs PARA ChimeraX (reconstrucción de atención)
====================================================================
Proyecta la atención interfacial del modelo sobre el B-factor de archivos PDB,
para visualizar en ChimeraX qué átomos recibe atención. Genera, por complejo:
  1. Ligando aislado (atención por átomo en B-factor)
  2. Proteína completa (el sitio se ilumina, el fondo queda ~0)
  3. Hotspots del sitio (átomos de proteína con atención >= umbral)

Usa el MODELO CORREGIDO directamente (sin monkey-patch): el forward del modelo
ya expone return_attention y tiene el gating con pair_proj.

Para las DOS figuras (un protocolo cada una), cambia PATH_MODELO_PT y
PATH_GRAFOS_TARGET según el caso (CASF o split) y vuelve a correr.

NO incluye el FRF (métrica descartada por tautológica). Mantiene R_mol
(cobertura) en el reporte, que sí es válido.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist
# 🎯 IMPORTACIÓN DEL MODELO DINÁMICO CORREGIDO (CON GATING)
import sys

# =========================================================================
# 🎯 ANCLA DE RUTAS: Forzar a Python a encontrar la subcarpeta 'Modelo'
# =========================================================================
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
# CONFIGURACIÓN — cambia estas dos rutas según el protocolo
# =========================================================================
SEED = 7

# --- CASF (Protocolo A) ---
# PATH_GRAFOS_TARGET = str(config.GRAFOS_CASF_CORE285)
# PATH_MODELO_PT = str(config.MODELS_DIR / f"best_model_REFINEDsinCore_CASFtest_seed_{SEED}_lam0p0_full.pt")
# ETIQUETA = "CASF"

# --- SPLIT (Protocolo B) — descomenta para el segundo caso ---
PATH_GRAFOS_TARGET = str(config.GRAFOS_REFINED)
PATH_MODELO_PT = str(config.MODELS_DIR / f"best_model_SIMSPLIT_seed_{SEED}_lam0p0_full.pt")
ETIQUETA = "SPLIT"

RUTA_RAIZ_SALIDA = str(config.OUTPUTS_DIR / "Graficas_finales")

# Si se define (para el split), procesa SOLO los complejos de esta lista (el test).
# Para CASF, déjalo en None (procesa todos los del core).
LISTA_IDS = None
# Para el split, descomenta:
LISTA_IDS = str(config.SPLIT_DIR / "test.txt")

UMBRAL_TAO = 0.1               # atención considerable en el ligando (cobertura)
UMBRAL_PROTEINA_CRITICA = 0.05 # umbral para los hotspots del sitio
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}


def obtener_simbolo_elemento(num_atomico):
    mapeo = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S',
             17: 'Cl', 30: 'ZN', 35: 'Br', 53: 'I'}
    return mapeo.get(int(num_atomico), 'X')


def escribir_pdb_estricto(coords, elementos, valores_bfactor, res_name, ruta_guardado):
    """PDB con formato rígido; la atención va en la columna de B-factor."""
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modelo corregido, forward normal (sin monkey-patch)
    model = BindingAffinityModel(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(PATH_MODELO_PT, map_location=device))
    model.eval()

    os.makedirs(RUTA_RAIZ_SALIDA, exist_ok=True)
    carpeta_pdbs = os.path.join(RUTA_RAIZ_SALIDA, f"PDB_XAI_{ETIQUETA}_seed_{SEED}")
    os.makedirs(carpeta_pdbs, exist_ok=True)

    file_list = [f for f in os.listdir(PATH_GRAFOS_TARGET) if f.endswith('.pt')]

    # Filtro opcional por lista de IDs (para el test del split)
    if LISTA_IDS is not None and os.path.exists(LISTA_IDS):
        with open(LISTA_IDS) as f:
            ids_permitidos = {l.strip() for l in f if l.strip()}
        file_list = [f for f in file_list if f[:-3] in ids_permitidos]
        print(f"Filtrado por lista: {len(file_list)} complejos (de la lista {os.path.basename(LISTA_IDS)})")

    print(f"Generando PDBs de atención | {ETIQUETA} | {len(file_list)} complejos")
    print(f"Salida: {carpeta_pdbs}")

    registros = []

    with torch.no_grad():
        for file_name in tqdm(file_list, desc="Procesando"):
            data = torch.load(os.path.join(PATH_GRAFOS_TARGET, file_name),
                              map_location=device, weights_only=False)
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
            data = data.to(device)

            # Forward normal del modelo corregido, pidiendo la atención
            pkd_predicho, attn_weights = model(data, return_attention=True)
            if attn_weights is None:
                continue
            if attn_weights.dim() > 1:
                attn_weights = attn_weights.mean(dim=-1).view(-1)

            # Reconstruir el edge_index como lo hace el modelo (cdist <= cutoff)
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
            if len(indices_ligando) == 0 or len(indices_proteina) == 0:
                continue

            # Acumular atención interfacial (contactos cruzados) por átomo
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

            # Cobertura del ligando (R_mol): fracción de átomos con atención >= umbral
            n_total = len(indices_ligando)
            n_activos = int(np.sum(np.array(atenciones_ligando) >= UMBRAL_TAO))
            r_mol = n_activos / (n_total + 1e-9)

            # Hotspots del sitio (proteína con atención >= umbral)
            coords_h, elem_h, attn_h = [], [], []
            for il, idx_g in enumerate(indices_proteina):
                if atenciones_proteina[il] >= UMBRAL_PROTEINA_CRITICA:
                    coords_h.append(pos_proteina[il])
                    elem_h.append(elementos_proteina[il])
                    attn_h.append(atenciones_proteina[il])

            # --- Escribir los 3 PDBs ---
            gid = file_name.replace('.pt', '')
            escribir_pdb_estricto(pos_ligando, elementos_ligando, atenciones_ligando,
                                  "LIG", os.path.join(carpeta_pdbs, f"LIGAND_{gid}.pdb"))
            escribir_pdb_estricto(pos_proteina, elementos_proteina, atenciones_proteina,
                                  "PRO", os.path.join(carpeta_pdbs, f"PROTEIN_{gid}.pdb"))
            if len(coords_h) > 0:
                escribir_pdb_estricto(coords_h, elem_h, attn_h,
                                      "REC", os.path.join(carpeta_pdbs, f"SITE_{gid}.pdb"))

            registros.append({
                'Graph': file_name,
                'pKd_pred': pkd_predicho.item(),
                'R_mol_cobertura': r_mol,
                'Atomos_ligando_activos': n_activos,
                'Atomos_ligando_total': n_total,
                'Atomos_sitio': len(coords_h),
            })

    df = pd.DataFrame(registros)
    csv_path = os.path.join(RUTA_RAIZ_SALIDA, f"reporte_pdbs_{ETIQUETA}_seed_{SEED}.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print(f"COMPLETADO | {ETIQUETA}")
    print("=" * 60)
    print(f"  Complejos procesados: {len(df)}")
    print(f"  PDBs generados:       ~{len(df) * 3}")
    print(f"  Cobertura media R_mol: {df['R_mol_cobertura'].mean()*100:.2f}%")
    print(f"  CSV: {csv_path}")
    print(f"  PDBs: {carpeta_pdbs}")


if __name__ == "__main__":
    main()