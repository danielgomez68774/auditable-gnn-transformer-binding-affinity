import os
import gzip
import zipfile
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd
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

RUTA_RAIZ_CROSSDOCKED = str(config.CROSS_DOCKED_RAW)
PATH_OUTPUT_GRAFOS = str(config.GRAFOS_CROSSDOCKED_MASIVO)
PATH_CSV_AUDITORIA = str(config.GRAFOS_CROSSDOCKED_MASIVO / "mapeo_auditoria_crossdocked_dinamico.csv")
LIMITE_CARPETAS = 50  # Primeras 50 familias/carpetas biológicas

os.makedirs(PATH_OUTPUT_GRAFOS, exist_ok=True)

# --- PATRONES QUÍMICOS ESTILO SIGN ---
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

def procesar_complejo_a_grafo_dinamico(pocket, ligand):
    """
    🎯 MODIFICACIÓN DINÁMICA DE ALTA FIDELIDAD:
    Calcula ÚNICAMENTE las aristas covalentes internas rígidas.
    Elimina por completo el cdist y las RBFs estáticas de la CPU.
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

    off = len(x_p)
    adj_cov = []
    
    # 1. Aristas Covalentes del Receptor (Proteína)
    for b in pocket.GetBonds():
        idx1, idx2 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj_cov.append([idx1, idx2])
        adj_cov.append([idx2, idx1])

    # 2. Aristas Covalentes del Ligando (Añadiendo el offset de la proteína)
    for b in ligand.GetBonds():
        idx1, idx2 = b.GetBeginAtomIdx() + off, b.GetEndAtomIdx() + off
        adj_cov.append([idx1, idx2])
        adj_cov.append([idx2, idx1])

    # Si por alguna anomalía estructural el complejo no tiene aristas covalentes, 
    # creamos un auto-bucle básico de control para que no falle el Laplaciano
    if len(adj_cov) == 0:
        edge_index_cov = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        edge_index_cov = torch.tensor(adj_cov, dtype=torch.long).t().contiguous()
        
    return x_total, edge_index_cov, coords_total

def agregar_lpe_covalente(data, k=15):
    """
    🧠 NÚCLEO TOPOLÓGICO SEGURO:
    El Laplaciano se calcula ESTRICTAMENTE sobre el esqueleto covalente estable,
    haciendo que los autovectores sean invariantes ante el movimiento del ligando.
    """
    num_nodes = data.num_nodes
    try:
        edge_index_lap, edge_weight = get_laplacian(data.edge_index, normalization='sym', num_nodes=num_nodes)
        L_sparse = to_scipy_sparse_matrix(edge_index_lap, edge_weight, num_nodes)
        
        k_adj = min(k + 1, num_nodes - 1)
        eig_vals, eig_vecs = sp.linalg.eigsh(L_sparse, k=k_adj, which='SA', tol=1e-3, maxiter=5000)
        idx = eig_vals.argsort()
        eig_vecs = eig_vecs[:, idx]
        lpe = eig_vecs[:, 1:k+1]
        
        if lpe.shape[1] < k:
            lpe = np.concatenate([lpe, np.zeros((num_nodes, k - lpe.shape[1]))], axis=1)
    except: 
        # Sistema de amortiguación en caso de grafos disjuntos o problemas de convergencia Lanczos
        lpe = np.zeros((num_nodes, k))
        
    data.lpe = torch.from_numpy(lpe).float()
    return data

def descompresor_inteligente(ruta_archivo, destino_ext):
    if ruta_archivo.endswith('.gz'):
        with gzip.open(ruta_archivo, 'rb') as f_in:
            with open(destino_ext, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return True
    elif ruta_archivo.endswith('.zip'):
        with zipfile.ZipFile(ruta_archivo, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(destino_ext))
        return True
    return False

def pipeline_masivo_dinamico():
    print(f"🚀 Iniciando Pipeline Masivo CrossDocked DINÁMICO (Límite: {LIMITE_CARPETAS} carpetas)")
    
    carpetas_validas = [d for d in os.listdir(RUTA_RAIZ_CROSSDOCKED) 
                        if os.path.isdir(os.path.join(RUTA_RAIZ_CROSSDOCKED, d))][:LIMITE_CARPETAS]
    
    total_global_grafos = 0
    registros_auditoria = []
    
    for idx_c, carpeta in enumerate(carpetas_validas):
        path_carpeta_actual = os.path.join(RUTA_RAIZ_CROSSDOCKED, carpeta)
        print(f"\n📂 [{idx_c+1}/{LIMITE_CARPETAS}] Procesando familia: {carpeta}")
        
        todos_los_archivos = os.listdir(path_carpeta_actual)
        receptores_pdb = [f for f in todos_los_archivos if f.endswith('_rec.pdb')]
        
        for rec_pdb in receptores_pdb:
            prefix_key = rec_pdb.replace('_rec.pdb', '') 
            path_rec_completo = os.path.join(path_carpeta_actual, rec_pdb)
            
            sdf_candidatos = [
                f for f in todos_los_archivos 
                if prefix_key in f 
                and os.path.isfile(os.path.join(path_carpeta_actual, f))
                and (f.endswith('.sdf') or f.endswith('.sdf.gz') or f.endswith('.zip'))
            ]
            if not sdf_candidatos: continue
                
            archivo_sdf_target = sdf_candidatos[0]
            path_sdf_origen = os.path.join(path_carpeta_actual, archivo_sdf_target)
            path_sdf_descomprimido = os.path.join(path_carpeta_actual, f"temp_runtime_{prefix_key}.sdf")
            
            necesita_limpieza = False
            if path_sdf_origen.endswith('.gz') or path_sdf_origen.endswith('.zip'):
                exito = descompresor_inteligente(path_sdf_origen, path_sdf_descomprimido)
                if not exito: continue
                necesita_limpieza = True
            else:
                path_sdf_descomprimido = path_sdf_origen 
                
            rec_mol = Chem.MolFromPDBFile(path_rec_completo, removeHs=False)
            suppl = Chem.SDMolSupplier(path_sdf_descomprimido, removeHs=False, sanitize=True)
            
            if rec_mol is None or not suppl:
                if necesita_limpieza and os.path.exists(path_sdf_descomprimido): 
                    try: os.remove(path_sdf_descomprimido)
                    except: pass
                continue
                
            print(f"   ↳ 🧬 Vinculando {rec_pdb} -> {archivo_sdf_target} ({len(suppl)} poses encontradas)")
            
            registros_auditoria.append({
                'familia_biologica': carpeta,
                'receptor_pdb': rec_pdb,
                'archivo_ligandos_sdf': archivo_sdf_target,
                'numero_total_poses': len(suppl)
            })
            
            for idx_pose, lig_mol in enumerate(suppl):
                if lig_mol is None: continue
                
                score_vina = float(lig_mol.GetProp("minimizedAffinity")) if lig_mol.HasProp("minimizedAffinity") else 0.0
                
                try:
                    # Extracción limpia y esbelta alineada a tu constructor dinámico de GPU
                    res = procesar_complejo_a_grafo_dinamico(rec_mol, lig_mol)
                    if res:
                        # Convertir Score de Vina a pKd (Unidades de afinidad biológica estándar)
                        pkd_target = score_vina / -1.363 if score_vina != 0.0 else 0.0
                        
                        # 🎯 OBJETO DATA PURO: Guardamos únicamente x, pos y el edge_index covalente.
                        # El modelo se encargará del resto dentro de la GPU.
                        data = Data(
                            x=res[0], 
                            edge_index=res[1], 
                            pos=res[2],
                            y=torch.tensor([[pkd_target]], dtype=torch.float),
                            score_vina_original=torch.tensor([[score_vina]], dtype=torch.float)
                        )
                        
                        # Añadimos el codificador topológico estructural estable
                        data = agregar_lpe_covalente(data, k=15)
                        
                        out_name = f"{carpeta}_{prefix_key}_pose_{idx_pose+1:02d}.pt"
                        torch.save(data, os.path.join(PATH_OUTPUT_GRAFOS, out_name))
                        total_global_grafos += 1
                except:
                    continue
            
            del suppl
                    
            if necesita_limpieza and os.path.exists(path_sdf_descomprimido):
                try: os.remove(path_sdf_descomprimido)
                except: pass
                    
    # Exportación automática de la bitácora de trazabilidad para los Métodos de tu tesis
    df_auditoria = pd.DataFrame(registros_auditoria)
    df_auditoria.to_csv(PATH_CSV_AUDITORIA, index=False, encoding='utf-8')
    print(f"\n📝 ¡Bitácora de auditoría serializada con éxito en: {PATH_CSV_AUDITORIA}")
    print(f"🥇 ¡Pipeline Completado! {total_global_grafos} grafos dinámicos puros generados en {PATH_OUTPUT_GRAFOS}.")

if __name__ == "__main__":
    pipeline_masivo_dinamico()