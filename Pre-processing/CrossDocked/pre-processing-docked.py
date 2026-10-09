import os
import gzip  
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import scipy.sparse as sp
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

RDLogger.DisableLog('rdApp.*')
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config

# --- CONFIGURACIÓN DE RUTA ÚNICA PARA TU PRUEBA LOCAL ---
CARPETA_TEST = str(config.CROSS_DOCKED_RAW / "1433S_HUMAN_1_233_0")
PATH_REC = os.path.join(CARPETA_TEST, "4dhr_A_rec.pdb")
PATH_SDF_GZ = os.path.join(CARPETA_TEST, "4dhr_A_rec_3smn_fc7_lig_it1_tt_docked.sdf.gz")

NOMBRE_FAMILIA = os.path.basename(CARPETA_TEST)
PATH_OUTPUT = os.path.join(str(config.GRAFOS_CROSSDOCKED_INDIVIDUAL), NOMBRE_FAMILIA)

os.makedirs(PATH_OUTPUT, exist_ok=True)

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
        except: g_charge = 0.0

        vdw_radius = ptable.GetRvdw(atom.GetAtomicNum())
        hybrid = int(atom.GetHybridization())
        if hybrid < 0 or hybrid > 7: hybrid = 0

        fila = [
            float(obtener_categoria_atomo(atom, es_ligando)), float(atom.GetAtomicNum()),
            float(int(atom.IsInRing())), float(atom.GetTotalNumHs()), float(atom.GetFormalCharge()),
            float(int(atom.GetIsAromatic())), float(hybrid), float(atom.GetExplicitValence()),
            float(atom.GetImplicitValence()), float(1.0 if idx in donadores_idx else 0.0),
            float(1.0 if idx in aceptores_idx else 0.0), float(1.0 if idx in hidrofobicos_idx else 0.0),
            g_charge, vdw_radius, molcode
        ]
        nodos.append([x if np.isfinite(x) else 0.0 for x in fila])
    return nodos

def construir_esqueleto_dinamico_pose(pocket, ligand):
    """
    🎯 REESTRUCTURADO: Ya no calcula cdist, ni aristas intermoleculares, ni RBF.
    Solo extrae la conectividad covalente para inyectar el LPE correctamente.
    """
    if not pocket or not ligand: return None
    try:
        x_p = extraer_nodos_features(pocket, False)
        x_l = extraer_nodos_features(ligand, True)
        x_total = torch.tensor(x_p + x_l, dtype=torch.float)
        
        c_p = torch.tensor(pocket.GetConformer().GetPositions(), dtype=torch.float)
        c_l = torch.tensor(ligand.GetConformer().GetPositions(), dtype=torch.float)
        coords_total = torch.cat([c_p, c_l], dim=0)
    except: 
        return None

    # Extraer estrictamente el esqueleto covalente interno para el cálculo de LPE
    off = len(x_p)
    adj_cov = []
    for b in pocket.GetBonds():
        idx1, idx2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj_cov.append([idx1, idx2]); adj_cov.append([idx2, idx1])

    for b in ligand.GetBonds():
        idx1, idx2 = b.GetBeginAtomIdx() + off, b.GetEndAtomIdx() + off
        adj_cov.append([idx1, idx2]); adj_cov.append([idx2, idx1])

    edge_index_cov = torch.tensor(adj_cov, dtype=torch.long).t().contiguous()
    return x_total, coords_total, edge_index_cov

def agregar_lpe_estricto(data, edge_index_cov, k=15):
    num_nodes = data.num_nodes
    edge_index, edge_weight = get_laplacian(edge_index_cov, normalization='sym', num_nodes=num_nodes)
    L_sparse = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)
    try:
        k_adj = min(k + 1, num_nodes - 1)
        eig_vals, eig_vecs = sp.linalg.eigsh(L_sparse, k=k_adj, which='SA', tol=1e-3, maxiter=5000)
        idx = eig_vals.argsort()
        eig_vecs = eig_vecs[:, idx]
        lpe = eig_vecs[:, 1:k+1]
        if lpe.shape[1] < k:
            lpe = np.concatenate([lpe, np.zeros((num_nodes, k - lpe.shape[1]))], axis=1)
    except: 
        lpe = np.zeros((num_nodes, k))
    data.lpe = torch.from_numpy(lpe).float()
    return data

def ejecutar_prueba_local():
    print(f"====== INICIANDO PROCESAMIENTO DINÁMICO INDIVIDUAL ({NOMBRE_FAMILIA}) ======")
    if not os.path.exists(PATH_REC) or not os.path.exists(PATH_SDF_GZ):
        print("❌ Error: Estructuras no localizadas en la ruta especificada.")
        return

    # 1. Parsing del Receptor PDB
    rec_mol = Chem.MolFromPDBFile(PATH_REC, removeHs=False)
    
    # 2. Parsing directo en memoria del .SDF.GZ
    try:
        with gzip.open(PATH_SDF_GZ, 'rt') as f:
            sdf_data = f.read()
        suppl = Chem.SDMolSupplier()
        suppl.SetData(sdf_data, removeHs=False, sanitize=True)
    except Exception as e:
        print(f"❌ Error al descomprimir el archivo SDF.GZ en memoria: {e}")
        return

    if rec_mol is None or not suppl:
        print("❌ Error crítico: RDKit falló en el parsing molecular.")
        return

    print(f"✅ Archivos cargados. Detectadas {len(suppl)} poses en el archivo comprimido SDF.")
    
    procesados = 0
    for idx_pose, lig_mol in enumerate(suppl):
        if lig_mol is None: continue

        score_tradicional = float(lig_mol.GetProp("minimizedAffinity")) if lig_mol.HasProp("minimizedAffinity") else 0.0

        try:
            # Obtención limpia del esqueleto y las posiciones espaciales continuas
            res = construir_esqueleto_dinamico_pose(rec_mol, lig_mol)
            
            if res:
                x_total, coords_total, edge_index_cov = res
                pkd_positivo_target = score_tradicional / -1.363 if score_tradicional != 0.0 else 0.0

                # 🎯 NUEVO OBJETO DATA: Esbelto, sin tensores redundantes precalculados
                data = Data(
                    x=x_total, 
                    pos=coords_total,
                    y=torch.tensor([[pkd_positivo_target]], dtype=torch.float),
                    score_vina_original=torch.tensor([[score_tradicional]], dtype=torch.float)
                )
                
                # Inyección del descriptor Laplaciano global usando el esqueleto covalente básico
                data = agregar_lpe_estricto(data, edge_index_cov, k=15)
                
                # Nomenclatura limpia basada en la subcarpeta del complejo
                out_name = f"{NOMBRE_FAMILIA}_pose_{idx_pose+1:02d}.pt"
                torch.save(data, os.path.join(PATH_OUTPUT, out_name))
                
                procesados += 1
                print(f"   [OK] Pose {idx_pose+1:02d} -> Score Vina: {score_tradicional:.2f} | Target pKd: {pkd_positivo_target:.4f}")
        except Exception as e:
            print(f"   [FAIL] Pose {idx_pose+1} falló: {str(e)}")
            continue

    print(f"\n🎉 ¡Procesamiento completado! {procesados} grafos dinámicos serializados en {PATH_OUTPUT}.")

if __name__ == "__main__":
    ejecutar_prueba_local()