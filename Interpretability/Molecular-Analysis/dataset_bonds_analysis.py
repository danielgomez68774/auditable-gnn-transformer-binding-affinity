import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Force PyTorch to ignore library duplication conflicts on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- repo-root bootstrap: make `config` importable from any folder ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config

from Model.model import BindingAffinityModel

# Configure your local input and output folders
TAG = ""
SEED = 7
PATH_GRAFOS_TARGET = str(config.GRAFOS_CASF_CORE285)
PATH_MODELO_PT = str(config.MODELS_DIR / f"best_model_REFINEDsinCore_CASFtest_seed_{SEED}.pt")
RUTA_RAIZ_SALIDA = str(config.OUTPUTS_DIR / "Graficas_finales")

UMBRAL_TAO = 0.1             # Considerable attention threshold for the ligand (FRF Metric)
UMBRAL_PROTEINA_CRITICA = 0.05 # Threshold to isolate the hotspots of the protein's active site
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

# =========================================================================
# 🛠️ DYNAMIC HOT INTERCEPTOR (MONKEY PATCHING) - ULTRA FAST
# =========================================================================
def forward_interceptor_5angstrom(self, data, return_attention=False):
    from torch_geometric.nn import global_mean_pool
    self.cutoff = 4.5
    dist_matrix = torch.cdist(data.pos, data.pos, p=2)
    mask_cutoff = (dist_matrix <= self.cutoff)
    if data.batch is not None:
        mask_cutoff = mask_cutoff & (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
        
    row, col = torch.where(mask_cutoff)
    edge_index_dinamico = torch.stack([row, col], dim=0)
    data.edge_index_vivas = edge_index_dinamico
    
    edge_attr_dinamico = self.rbf_generator(dist_matrix[row, col])
    h = self.mol_encoder(data.x, data.lpe)
    edge_attr = self.edge_encoder(edge_attr_dinamico)
    
    h = self.gine_block(h, edge_index_dinamico, edge_attr)
    res = self.transformer_block(h, edge_index_dinamico, edge_attr, pos=data.pos, return_attn=return_attention)
    
    h_graph = global_mean_pool(res[0] if return_attention else res, data.batch)
    out = self.readout(h_graph)
    return (out, res[1]) if return_attention else out

def obtener_simbolo_elemento(num_atomico):
    mapeo = {6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 30: 'ZN', 35: 'Br', 53: 'I'}
    return mapeo.get(int(num_atomico), 'X')

def escribir_pdb_estricto(coords, elementos, valores_bfactor, res_name, ruta_guardado):
    """
    Writes a PDB file with rigid positional formatting (official PDB standard)
    mapping the attention scores onto the B-factor column to avoid errors in Chimera.
    """
    with open(ruta_guardado, 'w') as f:
        for idx in range(len(coords)):
            x, y, z = coords[idx]
            elem = elementos[idx]
            val = valores_bfactor[idx]
            
            atom_name = f"{elem}{idx+1}"
            if len(atom_name) > 4: atom_name = atom_name[:4]
                
            # Rigid by column positions:
            linea = (
                f"ATOM  {idx+1:5d}  {atom_name:<4}{res_name:<4}A   1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00{val:6.2f}           {elem:>2}\n"
            )
            f.write(linea)
        f.write("END\n")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BindingAffinityModel.forward = forward_interceptor_5angstrom
    model = BindingAffinityModel(**MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(PATH_MODELO_PT, map_location=device))
    model.eval()
    
    # Create the root and the dedicated subfolder for the three-dimensional bank
    os.makedirs(RUTA_RAIZ_SALIDA, exist_ok=True)
    carpeta_pdbs = os.path.join(RUTA_RAIZ_SALIDA, "PDB_TRILOGIES_XAI_CrossDocked")
    os.makedirs(carpeta_pdbs, exist_ok=True)
    
    file_list = [f for f in os.listdir(PATH_GRAFOS_TARGET) if f.endswith('.pt')]
    print(f"🚀 Starting Massive Pipeline for Coherent Reconstruction and Catalytic Site Mapping...")
    print(f"📁 3D Repository (.pdb): {carpeta_pdbs}")
    
    registros_dataset = []
    
    with torch.no_grad():
        for file_name in tqdm(file_list, desc="Processing and injecting latent matrices"):
            data = torch.load(os.path.join(PATH_GRAFOS_TARGET, file_name), map_location=device, weights_only=False)
            if not hasattr(data, 'batch') or data.batch is None:
                data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
                
            data = data.to(device)
            pkd_predicho, attn_weights = model(data, return_attention=True)
            delta_g_ia = -1.363 * pkd_predicho.item()
            
            edge_index_dinamico = data.edge_index_vivas if hasattr(data, 'edge_index_vivas') else data.edge_index
            if attn_weights is None or edge_index_dinamico is None: continue
            if attn_weights.dim() > 1: attn_weights = attn_weights.mean(dim=-1).view(-1)
                
            src, dst = edge_index_dinamico[0].cpu().numpy(), edge_index_dinamico[1].cpu().numpy()
            attn_np = attn_weights.detach().cpu().numpy()
            molcodes = data.x[:, -1].cpu().numpy()
            
            indices_ligando = np.where(molcodes > 0)[0]
            indices_proteina = np.where(molcodes < 0)[0]
            if len(indices_ligando) == 0 or len(indices_proteina) == 0: continue
                
            # 1. Accumulate pure cross intermolecular attention for each node of the graph
            atencion_acumulada_nodos = {idx: 0.0 for idx in range(len(molcodes))}
            for i in range(len(src)):
                u, v = src[i], dst[i]
                if molcodes[u] * molcodes[v] < 0:
                    atencion_acumulada_nodos[u] += attn_np[i]
                    atencion_acumulada_nodos[v] += attn_np[i]
                    
            # 2. Extract isolated spatial arrays
            pos_ligando = data.pos[indices_ligando].cpu().numpy()
            pos_proteina = data.pos[indices_proteina].cpu().numpy()
            dist_mat = cdist(pos_ligando, pos_proteina, metric='euclidean')
            
            # 3. Map ordered structural data for the Ligand
            elementos_ligando = [obtener_simbolo_elemento(data.x[idx, 1].item()) for idx in indices_ligando]
            atenciones_ligando = [atencion_acumulada_nodos[idx] for idx in indices_ligando]
            
            # 4. Map ordered structural data for the Protein
            elementos_proteina = [obtener_simbolo_elemento(data.x[idx, 1].item()) for idx in indices_proteina]
            atenciones_proteina = [atencion_acumulada_nodos[idx] for idx in indices_proteina]
            
            # 5. Compute the mathematical logic of the metrics (FRF and Reconstruction)
            n_total_ligando = len(indices_ligando)
            n_activos = 0
            n_no_huerfanos = 0
            n_huerfanos = 0
            suma_atencion_activos = 0.0
            penalizacion_huerfanos = 0.0
            
            for i_loc, idx_global in enumerate(indices_ligando):
                attn_atomo = atencion_por_atomo_ligando = atenciones_ligando[i_loc]
                if attn_atomo >= UMBRAL_TAO:
                    n_activos += 1
                    suma_atencion_activos += attn_atomo
                    
                    min_dist_a_proteina = np.min(dist_mat[i_loc])
                    if min_dist_a_proteina > 5.0:
                        penalizacion_huerfanos += attn_atomo
                        n_huerfanos +=1
                    else:
                        n_no_huerfanos+=1
                        
            r_mol = n_activos / (n_total_ligando + 1e-9)
            if(n_activos>0):
                frf_score = ((n_no_huerfanos * suma_atencion_activos) - penalizacion_huerfanos)/((n_activos * suma_atencion_activos))
            else:
                frf_score =  penalizacion_huerfanos*(-1)
            
            # 6. Automatically isolate the hotspots of the Catalytic Site
            coords_hotspots = []
            elem_hotspots = []
            attn_hotspots = []
            for idx_local, idx_global in enumerate(indices_proteina):
                attn_val = atenciones_proteina[idx_local]
                if attn_val >= UMBRAL_PROTEINA_CRITICA:
                    coords_hotspots.append(pos_proteina[idx_local])
                    elem_hotspots.append(elementos_proteina[idx_local])
                    attn_hotspots.append(attn_val)
            
            # =========================================================================
            # 🖨️ EXHAUSTIVE WRITING OF THE TRILOGY OF CORRECTED PDB FILES
            # =========================================================================
            id_grafo = file_name.replace('.pt', '')
            
            # PDB 1: The individual Ligand
            path_lig = os.path.join(carpeta_pdbs, f"XAI_LIGAND_{id_grafo}_{SEED}_{TAG}.pdb")
            escribir_pdb_estricto(pos_ligando, elementos_ligando, atenciones_ligando, "LIG", path_lig)
            
            # PDB 2: The Full Protein (The catalytic site will glow, the far background will be 0.0)
            path_prot = os.path.join(carpeta_pdbs, f"XAI_FULL_PROTEIN_{id_grafo}_{SEED}_{TAG}.pdb")
            escribir_pdb_estricto(pos_proteina, elementos_proteina, atenciones_proteina, "PRO", path_prot)
            
            # PDB 3: The Isolated Catalytic Site (Exclusively subsampled pocket residues)
            if len(coords_hotspots) > 0:
                path_sitio = os.path.join(carpeta_pdbs, f"XAI_CATALYTIC_SITE_HOTSPOTS_{id_grafo}_{SEED}_{TAG}.pdb")
                escribir_pdb_estricto(coords_hotspots, elem_hotspots, attn_hotspots, "REC", path_sitio)
            
            # Save numerical records for final statistical reduction
            registros_dataset.append({
                'Graph': file_name,
                'Predicted_Delta_G': pkd_predicho.item(),
                'R_mol_Reconstruction': r_mol,
                'Active_Ligand_Atoms': n_activos,
                'Total_Ligand_Atoms': n_total_ligando,
                'FRF_Score': frf_score,
                'Catalytic_Site_Protein_Atoms': len(coords_hotspots)
            })
            
    # =========================================================================
    # 📊 STATISTICAL ANALYSIS OF FINAL DATASET REDUCTION
    # =========================================================================
    df_final = pd.DataFrame(registros_dataset)
    csv_path = os.path.join(RUTA_RAIZ_SALIDA, f"consolidated_report_FRF_metric_CASF_seed_{SEED}_{TAG}.csv")
    df_final.to_csv(csv_path, index=False)
    
    print("\n" + "="*85)
    print(" 🏁 INDUSTRIAL PIPELINE COMPLETED SUCCESSFULLY: REPORTS AND PDB TRILOGIES GENERATED")
    print("="*85)
    print(f"🔹 CSV File Saved at:                       {csv_path}")
    print(f"🔹 Total PDBs Exported:                     {len(df_final) * 3}")
    print("-"*85)
    print("📊 STATISTICAL SUMMARY OF THE INTEGRAL LATENT SPACE:")
    print(f"  • Average Affinity (μ):                    {df_final['Predicted_Delta_G'].mean():.2f} ± {df_final['Predicted_Delta_G'].std():.2f} kcal/mol")
    print(f"  • Mean Reconstruction Coefficient:          {df_final['R_mol_Reconstruction'].mean()*100:.2f} %  Seed {SEED}")
    print(f"  • Average Dataset FRF Score:               {df_final['FRF_Score'].mean():.4f} ± {df_final['FRF_Score'].std():.4f}")
    print(f"  • Average Catalytic Site Size:             {df_final['Catalytic_Site_Protein_Atoms'].mean():.1f} atoms")
    print("="*85 + "\n")

if __name__ == "__main__":
    main()