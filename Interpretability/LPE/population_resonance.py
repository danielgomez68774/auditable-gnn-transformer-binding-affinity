"""
==========================================================================
CONSISTENCIA POBLACIONAL — atención media vs distancia sobre 4 datasets
==========================================================================
Evalúa el perfil de atención interfacial media en función de la distancia,
sobre cuatro conjuntos (CASF-2016, CSAR-HiQ, CrossDocked, PDBbind Refined),
promediando sobre las 8 semillas (media ± σ).

Se genera una figura por MODELO:
  - Modelo benchmark (Protocolo A): entrenado en Refined-sin-core.
  - Modelo split (Protocolo B): entrenado en el split por similitud.
Comparando ambas figuras se evidencia que el patrón de atención es consistente
entre datasets y entre los dos modelos.

NOTA: el conjunto PDBbind Refined corresponde (en parte) al conjunto de
entrenamiento de ambos modelos; se muestra como REFERENCIA DE CONSISTENCIA
INTERNA, no de generalización.

Usa el forward normal del modelo corregido (con pair_proj). Nomenclatura sobria.

Uso:
    python consistencia_poblacional.py A     # modelo benchmark
    python consistencia_poblacional.py B     # modelo split
"""

import os
import sys
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from scipy.stats import binned_statistic

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
CARPETA_MODELOS = str(config.MODELS_DIR)
DIR_CASF        = str(config.GRAFOS_CASF_CORE285)
DIR_CSAR        = str(config.GRAFOS_CSAR)
DIR_CROSSDOCKED = str(config.GRAFOS_CROSSDOCKED_MASIVO)
DIR_REFINED     = str(config.GRAFOS_REFINED)
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "LPE")

SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
CUTOFF = 4.5
MAX_SAMPLES = 100   # complejos por dataset (muestreo, como el original)
N_BINS = 30
MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

DATASETS = {
    "CASF-2016":       (DIR_CASF, '#008080', '-'),
    "CSAR-HiQ":        (DIR_CSAR, '#dc143c', '--'),
    "CrossDocked":     (DIR_CROSSDOCKED, '#e65100', '-.'),
    "PDBbind Refined": (DIR_REFINED, '#800080', ':'),
}


def nombre_checkpoint(seed, protocolo):
    if protocolo == "A":
        return f"best_model_REFINEDsinCore_CASFtest_seed_{seed}_lam0p0_full.pt"
    else:  # B
        return f"best_model_SIMSPLIT_seed_{seed}_lam0p0_full.pt"


class DatasetDinamico(Dataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pt')]
    def len(self): return len(self.file_list)
    def get(self, idx):
        data = torch.load(os.path.join(self.root_dir, self.file_list[idx]),
                          map_location='cpu', weights_only=False)
        if data is not None and hasattr(data, 'y'):
            data.y = data.y.view(-1, 1).float()
        return data


@torch.no_grad()
def extraer_dist_atn(loader, model, device, max_samples=100):
    """Distancias y atención de contactos interfaciales sobre un dataset."""
    model.eval()
    dists, atns = [], []
    for i, data in enumerate(loader):
        if i >= max_samples:
            break
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
        data = data.to(device)
        out = model(data, return_attention=True)
        attn = out[1] if isinstance(out, tuple) else None
        if attn is None:
            continue
        if attn.dim() > 1:
            attn = attn.mean(dim=-1).view(-1)
        # reconstruir edge_index como el modelo (cdist <= cutoff)
        dm = torch.cdist(data.pos, data.pos, p=2)
        mask = (dm <= CUTOFF)
        if data.batch is not None:
            mask = mask & (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
        row, col = torch.where(mask)
        molcode = data.x[:, -1]
        inter = (molcode[row] * molcode[col]) < 0
        if not inter.any():
            continue
        d = torch.norm(data.pos[row[inter]] - data.pos[col[inter]], dim=-1)
        a = attn[inter]
        dists.extend(d.cpu().numpy())
        atns.extend(a.cpu().numpy())
    return np.array(dists), np.array(atns)


def curva_binned(dist, attn, n_bins=30):
    mask = np.isfinite(dist) & np.isfinite(attn)
    dist, attn = dist[mask], attn[mask]
    if len(dist) == 0:
        return None, None
    bin_means, bin_edges, _ = binned_statistic(dist, attn, statistic='mean', bins=n_bins,
                                               range=(1.5, CUTOFF))
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return centers, bin_means


def main():
    if len(sys.argv) < 2 or sys.argv[1].upper() not in ("A", "B"):
        print("Uso: python consistencia_poblacional.py [A | B]")
        print("  A = modelo benchmark | B = modelo split")
        sys.exit(1)
    protocolo = sys.argv[1].upper()
    etiqueta = "benchmark" if protocolo == "A" else "split"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Para cada dataset: acumular la curva (media por bin) de cada semilla
    curvas_por_dataset = {nombre: [] for nombre in DATASETS}
    centers_ref = None
    faltantes = []

    for seed in SEEDS:
        ruta = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, protocolo))
        if not os.path.exists(ruta):
            faltantes.append(ruta); continue
        model = BindingAffinityModel(**MODEL_PARAMS).to(device)
        model.load_state_dict(torch.load(ruta, map_location=device))
        model.eval()

        for nombre, (carpeta, _, _) in DATASETS.items():
            if not os.path.isdir(carpeta):
                continue
            loader = DataLoader(DatasetDinamico(carpeta), batch_size=1, shuffle=False)
            dist, atn = extraer_dist_atn(loader, model, device, MAX_SAMPLES)
            centers, means = curva_binned(dist, atn, N_BINS)
            if means is not None:
                curvas_por_dataset[nombre].append(means)
                centers_ref = centers
        print(f"OK seed {seed:3d} ({protocolo}) | 4 datasets procesados")

    if faltantes:
        print("\n⚠️ Faltan checkpoints:")
        for f in faltantes: print("  ", f)

    # ---- Figura: media ± σ sobre semillas, una curva por dataset ----
    # try:
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(11, 6.5), dpi=300)
    #     plt.axvspan(2.5, 3.5, color='#2ca02c', alpha=0.05, label='H-bond window (2.5–3.5 Å)')
    #     for nombre, (_, color, estilo) in DATASETS.items():
    #         cs = curvas_por_dataset[nombre]
    #         if not cs:
    #             continue
    #         M = np.vstack(cs)
    #         m = np.nanmean(M, axis=0)
    #         sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(m)
    #         valid = ~np.isnan(m)
    #         plt.plot(centers_ref[valid], m[valid], color=color, lw=3, linestyle=estilo, label=nombre)
    #         plt.fill_between(centers_ref[valid], np.maximum(0, (m-sd)[valid]), (m+sd)[valid],
    #                          color=color, alpha=0.08)
    #     plt.title(f"Mean interfacial attention across datasets — {etiqueta} model\n"
    #               "(mean ± σ over 8 seeds)", fontsize=12, fontweight='bold', pad=12)
    #     plt.xlabel(r"Interfacial distance $d$ (Å)", fontsize=11)
    #     plt.ylabel(r"Mean attention ($\alpha$)", fontsize=11)
    #     plt.xlim(1.5, CUTOFF)
    #     plt.grid(True, linestyle=':', alpha=0.15)
    #     plt.legend(fontsize=9, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
    #     plt.tight_layout()
    #     ruta_fig = os.path.join(CARPETA_SALIDA, f"consistencia_datasets_{etiqueta}.png")
    #     plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
    #     print(f"\nFigura guardada: {ruta_fig}")
    # except Exception as e:
    #     print(f"(No se pudo generar la figura: {e})")

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 6.5), dpi=300)
        plt.axvspan(2.5, 3.5, color='#2ca02c', alpha=0.05, label='Ventana de enlace H (2.5–3.5 Å)')
        for nombre, (_, color, estilo) in DATASETS.items():
            cs = curvas_por_dataset[nombre]
            if not cs:
                continue
            M = np.vstack(cs)
            m = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(m)
            valid = ~np.isnan(m)
            plt.plot(centers_ref[valid], m[valid], color=color, lw=3, linestyle=estilo, label=nombre)
            plt.fill_between(centers_ref[valid], np.maximum(0, (m-sd)[valid]), (m+sd)[valid],
                             color=color, alpha=0.08)
        plt.title(f"Atención interfacial media entre conjuntos de datos — modelo {etiqueta}\n"
                  "(media ± σ sobre 8 semillas)", fontsize=12, fontweight='bold', pad=12)
        plt.xlabel(r"Distancia interfacial $d$ (Å)", fontsize=11)
        plt.ylabel(r"Atención media ($\alpha$)", fontsize=11)
        plt.xlim(1.5, CUTOFF)
        plt.grid(True, linestyle=':', alpha=0.15)
        plt.legend(fontsize=9, loc='upper right', frameon=True, facecolor='white', edgecolor='none')
        plt.tight_layout()
        ruta_fig = os.path.join(CARPETA_SALIDA, f"consistencia_datasets_{etiqueta}_es.png")
        plt.savefig(ruta_fig, dpi=300, bbox_inches='tight')
        print(f"\nFigura guardada: {ruta_fig}")
    except Exception as e:
        print(f"(No se pudo generar la figura: {e})")


if __name__ == "__main__":
    main()