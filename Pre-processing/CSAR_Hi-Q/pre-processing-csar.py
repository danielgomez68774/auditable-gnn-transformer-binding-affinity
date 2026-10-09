import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

# Desactivar logs de RDKit
RDLogger.DisableLog('rdApp.*')
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config
PATH_SUMMARY = str(config.CSAR_RAW / "SUMMARY_FILES" / "set1.csv")
PATH_STRUCTURES = str(config.CSAR_RAW / "Structures" / "set1")
PATH_OUTPUT_GRAFOS = str(config.GRAFOS_CSAR)

os.makedirs(PATH_OUTPUT_GRAFOS, exist_ok=True)

# --- PATRONES SMARTS ESTILO SIGN ---
DONADOR_SMARTS = Chem.MolFromSmarts("[#7,#8,#16;!H0]")
ACEPTOR_SMARTS = Chem.MolFromSmarts("[#7,#8,#16;!$(*-[+1]);!$(*-[+2])]")
HIDROFOBICO_SMARTS = Chem.MolFromSmarts("[$([C,S;H0,H1,H2,H3]),$([Cl,Br,I]),$(*-[Cl,Br,I]),$([P,As,Sb,Bi])] ")

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

# =========================================================================
# 🧬 SEPARACIÓN Y EXTRACCIÓN DE RECONOCIMIENTO (CSAR)
# =========================================================================
def separar_prot_lig_csar(path_mol2):
    mol = Chem.MolFromMol2File(path_mol2, removeHs=False, sanitize=False)
    if mol is None: return None, None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    
    prot_frags = []
    lig_candidatos = []
    for frag in frags:
        n = frag.GetNumAtoms()
        if n > 300: prot_frags.append(frag)
        elif 6 <= n <= 150 and any(a.GetAtomicNum() == 6 for a in frag.GetAtoms()):
            lig_candidatos.append(frag)
            
    if not prot_frags or not lig_candidatos: return None, None
    
    proteina = prot_frags[0]
    for f in prot_frags[1:]: proteina = Chem.CombineMols(proteina, f)
    ligando = max(lig_candidatos, key=lambda x: x.GetNumAtoms())
    return proteina, ligando

def extraer_pocket(protein_mol, ligand_mol, cutoff=6.0):
    prot_conf = protein_mol.GetConformer()
    lig_conf = ligand_mol.GetConformer()
    prot_coords = np.array(prot_conf.GetPositions())
    lig_coords = np.array(lig_conf.GetPositions())
    
    dist_mat = cdist(prot_coords, lig_coords)
    keep_atoms = np.any(dist_mat <= cutoff, axis=1)
    atom_indices = np.where(keep_atoms)[0].tolist()
    
    if len(atom_indices) == 0: return None
    
    pocket = Chem.RWMol()
    atom_map = {}
    for idx in atom_indices:
        atom = protein_mol.GetAtomWithIdx(idx)
        new_idx = pocket.AddAtom(Chem.Atom(atom))
        atom_map[idx] = new_idx
        
    for bond in protein_mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a1 in atom_map and a2 in atom_map:
            pocket.AddBond(atom_map[a1], atom_map[a2], bond.GetBondType())
            
    pocket = pocket.GetMol()
    conf = Chem.Conformer(len(atom_indices))
    for old_idx, new_idx in atom_map.items():
        conf.SetAtomPosition(new_idx, prot_conf.GetAtomPosition(old_idx))
    pocket.AddConformer(conf)
    return pocket

# =========================================================================
# 📐 CONSTRUCCIÓN DINÁMICA DE CARACTERÍSTICAS COVALENTES (LPE)
# =========================================================================
def construir_esqueleto_covalente_masivo(pocket_mol, ligand_mol):
    """
    🎯 ADAPTACIÓN DINÁMICA: Extrae descriptores quimiométricos y coordenadas puras.
    Ya no calcula distancias intermoleculares fijas ni expansiones Gaussianas.
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

    # Mapa topológico esquelético para guiar los descriptores LPE
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

# =========================================================================
# 🏎️ BUCLE PRINCIPAL CON ARCHIVO SUMMARY
# =========================================================================
def ejecutar():
    df = pd.read_csv(PATH_SUMMARY, skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    
    print(f"Columnas detectadas en CSAR: {list(df.columns)}") 
    print(f"Iniciando: {len(df)} complejos detectados en Summary CSV.")
    
    for _, row in df.iterrows():
        try:
            num = str(row['number']).strip()
            pdb_id = str(row['PDBID']).strip()
            pkd = float(row['-log10(K)'])
            
            path_complex = os.path.join(PATH_STRUCTURES, num, f"set1_{num}_complex.mol2")
            
            if not os.path.exists(path_complex): 
                continue

            prot, lig = separar_prot_lig_csar(path_complex)
            if prot and lig:
                pock = extraer_pocket(prot, lig, cutoff=6.0)
                
                # Inyección del extractor dinámico esbelto
                res = construir_esqueleto_covalente_masivo(pock, lig)
                if res:
                    X, coords, edge_index_cov = res
                    
                    # Generamos el objeto puro compatible con el modelo maestro
                    data = Data(
                        x=X, 
                        pos=coords, 
                        y=torch.tensor([[pkd]]).float(), 
                        pdb_id=pdb_id
                    )
                    
                    # Conservamos k=15 descriptores para blindar el benchmark de evaluación
                    data = agregar_lpe_estricto(data, edge_index_cov, k=15)
                    
                    torch.save(data, os.path.join(PATH_OUTPUT_GRAFOS, f"{pdb_id}.pt"))
                    print(f"✅ Complejo {pdb_id} (Nº {num}) exportado en formato dinámico puro.")
        
        except KeyError as e:
            print(f"❌ Error de mapeo posicional: Falta la columna {e}. Valida el encabezado del archivo CSV.")
            break 
        except Exception as e:
            print(f"⚠️ Omisión de fila: {e}")

if __name__ == "__main__":
    ejecutar()