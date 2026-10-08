import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

# =========================================================================
# ⚙️ CONFIGURACIÓN DE RUTAS MAESTRAS
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

RUTA_MI_PC = str(config.PDBBIND_REFINED_RAW)

# 🎯 NUEVO: carpeta del Core-285 de CASF-2016 (una subcarpeta por complejo)
RUTA_CASF_CORESET = str(config.CASF_CORESET_RAW)

# 🎯 DOS SALIDAS SEPARADAS con roles distintos:
PATH_OUTPUT_TRAIN = str(config.GRAFOS_REFINED)   # Refined SIN Core -> train/val
PATH_OUTPUT_TEST  = str(config.GRAFOS_CASF_CORE285)         # Core-285          -> test

os.makedirs(PATH_OUTPUT_TRAIN, exist_ok=True)
os.makedirs(PATH_OUTPUT_TEST, exist_ok=True)

# --- PATRONES SMARTS ESTILO SIGN ---
DONADOR_SMARTS = Chem.MolFromSmarts("[#7,#8,#16;!H0]")
ACEPTOR_SMARTS = Chem.MolFromSmarts("[#7,#8,#16;!$(*-[+1]);!$(*-[+2])]")
HIDROFOBICO_SMARTS = Chem.MolFromSmarts('[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]')


def obtener_categoria_atomo(atom, es_ligando):
    if es_ligando: return 1
    metales = [12, 20, 25, 26, 27, 28, 29, 30, 48, 80]
    return 2 if atom.GetAtomicNum() in metales else 0


def extraer_nodos_features(mol, es_ligando):
    if mol is None: return []
    molcode = 1.0 if es_ligando else -1.0

    try: Chem.SanitizeMol(mol)
    except: mol.UpdatePropertyCache(strict=False)

    mol.UpdatePropertyCache(strict=False)
    try: AllChem.ComputeGasteigerCharges(mol)
    except: pass

    donadores_idx = {idx[0] for idx in mol.GetSubstructMatches(DONADOR_SMARTS)}
    aceptores_idx = {idx[0] for idx in mol.GetSubstructMatches(ACEPTOR_SMARTS)}
    hidrofobicos_idx = {idx[0] for idx in mol.GetSubstructMatches(HIDROFOBICO_SMARTS)}

    ptable = Chem.GetPeriodicTable()
    nodos = []

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        try:
            g_charge = float(atom.GetProp('_GasteigerCharge'))
            if not np.isfinite(g_charge): g_charge = 0.0
        except:
            g_charge = 0.0

        vdw_radius = ptable.GetRvdw(atom.GetAtomicNum())
        hybrid = int(atom.GetHybridization())
        if hybrid < 0 or hybrid > 7: hybrid = 0

        fila = [
            float(obtener_categoria_atomo(atom, es_ligando)),
            float(atom.GetAtomicNum()),
            float(int(atom.IsInRing())),
            float(atom.GetTotalNumHs()),
            float(atom.GetFormalCharge()),
            float(int(atom.GetIsAromatic())),
            float(hybrid),
            float(atom.GetExplicitValence()),
            float(atom.GetImplicitValence()),
            float(1.0 if idx in donadores_idx else 0.0),
            float(1.0 if idx in aceptores_idx else 0.0),
            float(1.0 if idx in hidrofobicos_idx else 0.0),
            g_charge,
            vdw_radius,
            molcode
        ]

        fila_limpia = [x if np.isfinite(x) else 0.0 for x in fila]
        nodos.append(fila_limpia)

    return nodos


def construir_esqueleto_covalente_masivo(pocket_mol, ligand_mol):
    """
    Extrae la matriz atómica pura y las coordenadas tridimensionales desnudas.
    Ya no calcula la matriz de adyacencia intermolecular (cdist) ni bines RBF,
    pues esto se resolverá dinámicamente en el forward del modelo.
    """
    if not pocket_mol or not ligand_mol: return None

    try:
        x_p = extraer_nodos_features(pocket_mol, False)
        x_l = extraer_nodos_features(ligand_mol, True)
        x_total = torch.tensor(x_p + x_l, dtype=torch.float)

        c_p = torch.tensor(pocket_mol.GetConformer().GetPositions(), dtype=torch.float)
        c_l = torch.tensor(ligand_mol.GetConformer().GetPositions(), dtype=torch.float)
        coords_total = torch.cat([c_p, c_l], dim=0)
    except:
        return None

    # Reconstrucción mínima del mapa de adyacencia covalente intramolecular básica
    # Requerida únicamente para inyectar los autovectores Laplacianos globales (LPE)
    off = len(x_p)
    adj_cov = []

    for b in pocket_mol.GetBonds():
        idx1, idx2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj_cov.append([idx1, idx2]); adj_cov.append([idx2, idx1])

    for b in ligand_mol.GetBonds():
        idx1, idx2 = b.GetBeginAtomIdx() + off, b.GetEndAtomIdx() + off
        adj_cov.append([idx1, idx2]); adj_cov.append([idx2, idx1])

    edge_index_cov = torch.tensor(adj_cov, dtype=torch.long).t().contiguous()
    return x_total, coords_total, edge_index_cov


def agregar_lpe_estricto(data, edge_index_cov, k=15):
    """
    Calcula los descriptores de autovectores basados estrictamente en el esqueleto covalente del complejo.
    """
    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(edge_index_cov, normalization='sym', num_nodes=num_nodes)
    L_sparse = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    try:
        k_adj = min(k + 1, num_nodes - 1)
        if k_adj < 1: raise ValueError("Grafo deficitario")

        eig_vals, eig_vecs = sp.linalg.eigsh(L_sparse, k=k_adj, which='SA', tol=1e-3, maxiter=5000)
        idx = eig_vals.argsort()
        eig_vals, eig_vecs = eig_vals[idx], eig_vecs[:, idx]
        lpe = eig_vecs[:, 1:k + 1]

        if lpe.shape[1] < k:
            padding = np.zeros((num_nodes, k - lpe.shape[1]))
            lpe = np.concatenate([lpe, padding], axis=1)
    except:
        lpe = np.zeros((num_nodes, k))

    data.lpe = torch.from_numpy(lpe).float()
    return data


def configurar_rutas_locales(base_dir):
    path_raiz = Path(base_dir)
    path_refined = path_raiz / "refined-set"
    path_index = path_refined / "index" / "INDEX_refined_data.2020"

    if not path_refined.exists():
        print(f"❌ Error: Directorio no localizado en {path_refined}")
        return None, None
    return path_refined, path_index


def generar_lista_procesamiento(path_index, path_refined):
    if path_index is None: return None
    df_index = pd.read_csv(path_index, sep=r'\s+', comment='#', header=None, usecols=[0, 3], names=['pdb_id', 'pkd'])
    datos_listos = []

    print(f"Analizando {len(df_index)} complejos del Refined...")
    for _, row in df_index.iterrows():
        pdb_id = row['pdb_id']
        pkd = row['pkd']

        folder_complejo = os.path.join(path_refined, pdb_id)
        f_pocket = os.path.join(folder_complejo, f"{pdb_id}_pocket.pdb")
        f_ligand = os.path.join(folder_complejo, f"{pdb_id}_ligand.sdf")

        if os.path.exists(f_pocket) and os.path.exists(f_ligand):
            datos_listos.append({
                'id': pdb_id, 'pkd': pkd, 'pocket_path': f_pocket, 'ligand_path': f_ligand
            })

    df_final = pd.DataFrame(datos_listos)
    print(f"Listo para procesar (Refined): {len(df_final)} complejos válidos.")
    return df_final


# =========================================================================
# 🎯 NUEVO: leer los IDs del Core-285 de CASF-2016 desde su carpeta
# =========================================================================
def cargar_ids_core(coreset_path):
    """Devuelve el conjunto de PDB IDs del Core-285 (una subcarpeta por complejo).
    Se usa para EXCLUIRLOS del entrenamiento y evitar la fuga de datos, ya que
    el Core es un subconjunto del Refined."""
    if not os.path.isdir(coreset_path):
        print(f"❌ No se encontró el coreset en {coreset_path}. SIN EXCLUSIÓN => RIESGO DE FUGA.")
        return set()
    ids = set()
    for nombre in os.listdir(coreset_path):
        ruta = os.path.join(coreset_path, nombre)
        # cada complejo del core es una subcarpeta con nombre = PDB ID (4 caracteres)
        if os.path.isdir(ruta) and len(nombre) == 4:
            ids.add(nombre.strip().lower())
    print(f"🔒 Core-285 cargado: {len(ids)} IDs para separar train/test.")
    if len(ids) != 285:
        print(f"⚠️ Se esperaban 285 IDs y se encontraron {len(ids)}. Revisa la ruta del coreset.")
    return ids


# =========================================================================
# 🎯 NUEVO: procesamiento de UN complejo (reutilizado por train y test)
# =========================================================================
def procesar_y_guardar(pdb_id, path_pocket, path_ligand, pkd, output_dir):
    """Construye el grafo (features + coords + LPE) y lo serializa. True si OK."""
    try:
        pocket_mol = Chem.MolFromPDBFile(path_pocket, removeHs=False)
        supplier = Chem.SDMolSupplier(path_ligand, removeHs=False, sanitize=True)
        ligand_mol = supplier[0] if len(supplier) > 0 else None

        if ligand_mol is None or pocket_mol is None:
            return False

        res = construir_esqueleto_covalente_masivo(pocket_mol, ligand_mol)
        if not res:
            return False

        X, coords, edge_index_cov = res
        data = Data(
            x=X,
            pos=coords,
            y=torch.tensor([[float(pkd)]]).float(),
            pdb_id=pdb_id
        )
        data = agregar_lpe_estricto(data, edge_index_cov, k=15)
        torch.save(data, os.path.join(output_dir, f"{pdb_id}.pt"))

        # Gestión explícita de memoria (Windows)
        del pocket_mol, ligand_mol, data, res
        return True
    except Exception as e:
        print(f"❌ Error crítico procesando {pdb_id}: {e}")
        return False


# =========================================================================
# 🏎️ BUCLE MAESTRO DE PROCESAMIENTO
# =========================================================================
PATH_REFINED, PATH_INDEX = configurar_rutas_locales(RUTA_MI_PC)
df_trabajo = generar_lista_procesamiento(PATH_INDEX, PATH_REFINED)
ids_core = cargar_ids_core(RUTA_CASF_CORESET)

if df_trabajo is None:
    print("❌ Cancelado por error de rutas.")
else:
    # Mapa etiqueta pKd desde el índice Refined (para etiquetar el Core en el test)
    label_map = {str(r['id']).strip().lower(): float(r['pkd']) for _, r in df_trabajo.iterrows()}

    # ---------------------------------------------------------------------
    # (1) TRAIN/VAL  ->  Refined SIN Core-285
    # ---------------------------------------------------------------------
    print(f"\n===== (1) Generando TRAIN/VAL (Refined sin Core) en: {PATH_OUTPUT_TRAIN} =====")
    danados = []
    n_train, n_excluidos = 0, 0

    for i in range(len(df_trabajo)):
        pdb_id = str(df_trabajo.iloc[i]["id"])
        if pdb_id.strip().lower() in ids_core:
            n_excluidos += 1
            continue  # 🔒 pertenece al Core-285 -> NO entra al entrenamiento (irá al test)

        ok = procesar_y_guardar(
            pdb_id,
            df_trabajo.iloc[i]["pocket_path"],
            df_trabajo.iloc[i]["ligand_path"],
            df_trabajo.iloc[i]["pkd"],
            PATH_OUTPUT_TRAIN
        )
        if ok:
            n_train += 1
        else:
            danados.append(pdb_id)

        if (i + 1) % 500 == 0:
            print(f"📊 [TRAIN] Progreso: [{i + 1}/{len(df_trabajo)}]")

    print(f"✅ TRAIN/VAL: {n_train} grafos guardados | excluidos por Core: {n_excluidos} | dañados: {len(danados)}")

    # ---------------------------------------------------------------------
    # (2) TEST  ->  Core-285 (desde su propia carpeta, mismo preprocesamiento)
    # ---------------------------------------------------------------------
    print(f"\n===== (2) Generando TEST (Core-285) en: {PATH_OUTPUT_TEST} =====")
    n_test = 0
    core_encontrados = set()
    sin_label, core_danados = [], []

    for nombre in sorted(os.listdir(RUTA_CASF_CORESET)):
        ruta = os.path.join(RUTA_CASF_CORESET, nombre)
        if not (os.path.isdir(ruta) and len(nombre) == 4):
            continue

        pid = nombre.strip().lower()
        f_pocket = os.path.join(ruta, f"{nombre}_pocket.pdb")
        f_ligand = os.path.join(ruta, f"{nombre}_ligand.sdf")

        if not (os.path.exists(f_pocket) and os.path.exists(f_ligand)):
            core_danados.append(pid); continue

        # Etiqueta desde el índice Refined (mismo pKd que reporta PDBbind).
        # Si algún complejo del Core no está en el índice Refined v2020, se
        # omite y se avisa (puedes completarlo luego desde CASF CoreSet.dat).
        if pid not in label_map:
            sin_label.append(pid); continue

        ok = procesar_y_guardar(nombre, f_pocket, f_ligand, label_map[pid], PATH_OUTPUT_TEST)
        if ok:
            n_test += 1
            core_encontrados.add(pid)
        else:
            core_danados.append(pid)

    print(f"✅ TEST (Core-285): {n_test} grafos guardados")
    if sin_label:
        print(f"⚠️ {len(sin_label)} IDs del Core sin etiqueta en el índice Refined "
              f"(complétalos con CASF CoreSet.dat): {sorted(sin_label)}")
    if core_danados:
        print(f"⚠️ {len(core_danados)} complejos del Core no procesables: {sorted(core_danados)}")

    # ---------------------------------------------------------------------
    # (3) VERIFICACIÓN ANTI-FUGA: 0 complejos del Core deben estar en TRAIN
    # ---------------------------------------------------------------------
    train_ids = {f[:-3].lower() for f in os.listdir(PATH_OUTPUT_TRAIN) if f.endswith('.pt')}
    solapamiento = train_ids.intersection(ids_core)
    print(f"\n🔎 VERIFICACIÓN DE FUGA: {len(solapamiento)} complejos del Core presentes en TRAIN (debe ser 0).")
    if solapamiento:
        print(f"❌ FUGA DETECTADA -> {sorted(solapamiento)}")
    else:
        print("✅ Sin fuga: ningún complejo del Core-285 está en el conjunto de entrenamiento.")

    print(f"\n🥇 Fase completada.")
    print(f"   TRAIN/VAL (Refined sin Core): {PATH_OUTPUT_TRAIN}")
    print(f"   TEST (Core-285):              {PATH_OUTPUT_TEST}")