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

import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config
RUTA_CASF = str(config.CASF_CORESET_RAW)
PATH_OUTPUT_GRAFOS_CASF = str(config.GRAFOS_CASF_PROT_COMPLETA)

os.makedirs(PATH_OUTPUT_GRAFOS_CASF, exist_ok=True)

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
    🎯 VERSIÓN PURA: Extrae exclusivamente la matriz de características y coordenadas.
    Deja el cálculo de aristas intermoleculares y bines RBF completamente libre 
    para que el forward del modelo maestro los construya dinámicamente.
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

    # Mapeo intramolecular básico necesario únicamente para guiar los autovectores LPE
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
    Inyecta los 15 autovectores Laplacianos requeridos para blindar el benchmark del Core.
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
        lpe = eig_vecs[:, 1:k+1]
        
        if lpe.shape[1] < k:
            padding = np.zeros((num_nodes, k - lpe.shape[1]))
            lpe = np.concatenate([lpe, padding], axis=1)
    except:
        lpe = np.zeros((num_nodes, k))

    data.lpe = torch.from_numpy(lpe).float()
    return data

def configurar_rutas_casf(base_dir):
    path_raiz = Path(base_dir)
    path_coreset = path_raiz / "coreset"
    path_index = path_raiz / "power_scoring" / "CoreSet.dat"
    
    if not path_coreset.exists() or not path_index.exists():
        print(f"❌ Error: Estructura CASF-2016 no localizada en {path_raiz}")
        return None, None
    return path_coreset, path_index

def generar_lista_procesamiento_casf(path_index, path_coreset):
    if path_index is None: return None
    # Cambiamos para mapear las columnas 0 (ID) y 3 (pKd) del CoreSet.dat oficial de CASF
    df_index = pd.read_csv(path_index, sep=r'\s+', comment='#', header=None, usecols=[0, 3], names=['pdb_id', 'pkd'])
    datos_listos = []

    print(f"Analizando {len(df_index)} complejos del Core Set...")
    for _, row in df_index.iterrows():
        pdb_id = row['pdb_id']
        pkd = row['pkd']

        # En CASF, cada complejo es una carpeta con su ID dentro de 'coreset'
        folder_complejo = os.path.join(path_coreset, pdb_id)
        f_pocket = os.path.join(folder_complejo, f"{pdb_id}_protein.pdb")
        
        f_ligand_sdf = os.path.join(folder_complejo, f"{pdb_id}_ligand.sdf")
        f_ligand_mol2 = os.path.join(folder_complejo, f"{pdb_id}_ligand.mol2")
        f_ligand = f_ligand_sdf if os.path.exists(f_ligand_sdf) else f_ligand_mol2

        if os.path.exists(f_pocket) and os.path.exists(f_ligand):
            datos_listos.append({
                'id': pdb_id, 'pkd': pkd, 'pocket_path': f_pocket, 'ligand_path': f_ligand
            })

    df_final = pd.DataFrame(datos_listos)
    print(f"Listo para procesar: {len(df_final)} complejos válidos en CASF.")
    return df_final

# =========================================================================
# 🏎️ BUCLE MAESTRO DE ADAPTACIÓN DINÁMICA DE GRAFOS CASF
# =========================================================================
PATH_CORESET, PATH_INDEX_CASF = configurar_rutas_casf(RUTA_CASF)
df_trabajo = generar_lista_procesamiento_casf(PATH_INDEX_CASF, PATH_CORESET)

if df_trabajo is None:
    print("❌ Cancelado por inconsistencia en rutas de CASF.")
else:
    archivos_danados = []
    
    for i in range(len(df_trabajo)):
        pdb_id = df_trabajo.iloc[i]["id"]
        path_pocket = df_trabajo.iloc[i]["pocket_path"]
        path_ligand = df_trabajo.iloc[i]["ligand_path"]
        
        pocket_mol = Chem.MolFromPDBFile(path_pocket, removeHs=False)
        
        ligand_mol = None
        if path_ligand.endswith('.sdf'):
            supplier = Chem.SDMolSupplier(path_ligand, removeHs=False, sanitize=True)
            ligand_mol = supplier[0] if len(supplier) > 0 else None
        elif path_ligand.endswith('.mol2'):
            ligand_mol = Chem.MolFromMol2File(path_ligand, removeHs=False, sanitize=True)
        
        if ligand_mol is None or pocket_mol is None:
            print(f"[{i+1}/{len(df_trabajo)}] ⚠️ Omitiendo complejo inválido de CASF: {pdb_id}")
            archivos_danados.append(pdb_id)
            continue
        
        try:
            # Extracción limpia idéntica a tu pipeline maestro del Refined Set
            res = construir_esqueleto_covalente_masivo(pocket_mol, ligand_mol)
            
            if res:
                X, coords, edge_index_cov = res
                
                # 🎯 OBJETO DATA COMPACTO DEFINITIVO: Sin expansión radial prefijada
                data = Data(
                    x=X, 
                    pos=coords, 
                    y=torch.tensor([[df_trabajo.iloc[i]["pkd"]]]).float(), 
                    pdb_id=pdb_id
                )
                
                # Agregamos los 15 descriptores espectrales covalentes
                data = agregar_lpe_estricto(data, edge_index_cov, k=15)
                
                # Serialización directa en tu disco duro
                ruta_guardado = os.path.join(PATH_OUTPUT_GRAFOS_CASF, f"{pdb_id}.pt")
                torch.save(data, ruta_guardado)
                
                del pocket_mol, ligand_mol, data, res
                
            if (i + 1) % 50 == 0:
                print(f"📊 Progreso CASF: [{i+1}/{len(df_trabajo)}] grafos puros exportados.")
                
        except Exception as e:
            print(f"❌ Error crítico en complejo CASF {pdb_id}: {e}")
            archivos_danados.append(pdb_id)

    print(f"\n🥇 ¡Fase Completada! Corpus de CASF-2016 guardado sin expansión en: {PATH_OUTPUT_GRAFOS_CASF}")
    if archivos_danados:
        print(f"⚠️ Complejos CASF descartados por corrupción: {len(archivos_danados)}")