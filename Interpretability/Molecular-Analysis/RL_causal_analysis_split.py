"""
==========================================================================
PRUEBA DE ATRIBUCIÓN CAUSAL (DELTA S) - PARTICIÓN POR SIMILITUD (883 TEST)
==========================================================================
Evaluación de relevancia predictiva mediante intervención dirigida G \\ S.

Mide el impacto en el pK predicho: Delta S = |y_hat(G) - y_hat(G \\ S)|
para tres conjuntos de contactos interfaciales S de igual tamaño (k = 20%):
  1) S_alta: 20% de contactos interfaciales con mayor atención
  2) S_baja: 20% de contactos interfaciales con menor atención
  3) S_aleatoria: 20% de contactos interfaciales seleccionados al azar (control)
"""

import os
import sys
import csv
import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt

# =========================================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
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

CARPETA_MODELOS = str(config.MODELS_DIR)
CARPETA_GRAFOS  = str(config.GRAFOS_REFINED)  # 4407 grafos
LISTA_TEST      = str(config.SPLIT_DIR / "test.txt")    # 883 test IDs
CARPETA_SALIDA  = str(config.OUTPUTS_DIR / "RL_simsplit_causal")

CUTOFF = 4.5
SEEDS = [1, 2, 3, 4, 7, 42, 64, 123]
LAMBDAS = [0.0]  # En el split por similitud se evalúa lambda = 0.0
TOP_K_PCT = 0.20
MAX_COMPLEJOS = None

MODEL_PARAMS = {'d_model': 512, 'rbf_dim': 64, 'lpe_dim': 15}

def nombre_checkpoint(seed, lam):
    lam_tag = f"lam{lam}".replace(".", "p")
    return f"best_model_SIMSPLIT_seed_{seed}_{lam_tag}_full.pt"

def leer_lista_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

# =========================================================================
# FUNCIONES AUXILIARES DE INTERVENCIÓN Y ESTADÍSTICA
# =========================================================================
def predict_with_patched_cdist(model, data, cdist_override, device):
    orig_cdist = torch.cdist

    def patched_cdist(x1, x2, p=2):
        if x1.shape == data.pos.shape and x2.shape == data.pos.shape:
            return cdist_override
        return orig_cdist(x1, x2, p=p)

    torch.cdist = patched_cdist
    try:
        out = model(data, return_attention=False)
        y_pred = out[0].item() if isinstance(out, (tuple, list)) else out.item()
    finally:
        torch.cdist = orig_cdist

    return y_pred


def calcular_ic_95(datos):
    arr = np.array(datos)
    n = len(arr)
    if n < 2:
        return arr.mean(), 0.0, 0.0
    media = arr.mean()
    std = arr.std(ddof=1)
    sem = stats.sem(arr)
    ic_margin = sem * stats.t.ppf((1 + 0.95) / 2., n - 1)
    return media, std, ic_margin


@torch.no_grad()
def evaluar_complejo_causal(model, data, device, rng, k_pct=0.20):
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    data = data.to(device)

    out = model(data, return_attention=True)
    y_orig = out[0].item() if isinstance(out, (tuple, list)) else out.item()
    attn = out[1] if isinstance(out, (tuple, list)) else None

    if attn is None:
        return None

    pos = data.pos
    dm_orig = torch.cdist(pos, pos, p=2)
    molcode = data.x[:, -1]

    mask = (dm_orig <= CUTOFF)
    if data.batch is not None:
        bm = (data.batch.unsqueeze(1) == data.batch.unsqueeze(0))
        mask = mask & bm

    row, col = torch.where(mask)
    cross_mask = (molcode[row] * molcode[col]) < 0
    cross_indices = torch.where(cross_mask)[0]

    n_cross = len(cross_indices)
    if n_cross < 5:
        return None

    attn_heads_mean = attn.mean(dim=-1).view(-1)
    attn_interfacial = attn_heads_mean[cross_mask].cpu().numpy()

    k = max(1, int(n_cross * k_pct))

    sorted_order = np.argsort(attn_interfacial)
    idx_low_rel = sorted_order[:k]
    idx_high_rel = sorted_order[-k:]
    idx_rand_rel = rng.choice(n_cross, size=k, replace=False)

    edges_low = (row[cross_indices[idx_low_rel]], col[cross_indices[idx_low_rel]])
    edges_high = (row[cross_indices[idx_high_rel]], col[cross_indices[idx_high_rel]])
    edges_rand = (row[cross_indices[idx_rand_rel]], col[cross_indices[idx_rand_rel]])

    deltas = {}
    for grupo, (r_edges, c_edges) in [('alta', edges_high), ('baja', edges_low), ('aleatoria', edges_rand)]:
        dm_mod = dm_orig.clone()
        dm_mod[r_edges, c_edges] = 100.0
        dm_mod[c_edges, r_edges] = 100.0

        y_mod = predict_with_patched_cdist(model, data, dm_mod, device)
        deltas[grupo] = abs(y_orig - y_mod)

    return deltas


# =========================================================================
# RUTINA PRINCIPAL
# =========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    rng = np.random.default_rng(42)

    test_ids = leer_lista_ids(LISTA_TEST)
    archivos = [cid + ".pt" for cid in test_ids
                if os.path.exists(os.path.join(CARPETA_GRAFOS, cid + ".pt"))]
    if MAX_COMPLEJOS:
        archivos = archivos[:MAX_COMPLEJOS]

    print(f"Iniciando evaluación causal Delta S en {len(archivos)} complejos (Partición por Similitud)...")

    resultados_agregados = []
    resumen_por_lambda = {lam: {'alta': [], 'baja': [], 'aleatoria': []} for lam in LAMBDAS}

    for lam in LAMBDAS:
        for seed in SEEDS:
            ruta_ckpt = os.path.join(CARPETA_MODELOS, nombre_checkpoint(seed, lam))
            if not os.path.exists(ruta_ckpt):
                print(f"⚠️ Checkpoint no encontrado: {ruta_ckpt}")
                continue

            model = BindingAffinityModel(**MODEL_PARAMS).to(device)
            model.load_state_dict(torch.load(ruta_ckpt, map_location=device))
            model.eval()

            d_alta, d_baja, d_rand = [], [], []

            for fn in archivos:
                try:
                    data = torch.load(os.path.join(CARPETA_GRAFOS, fn), map_location='cpu', weights_only=False)
                except Exception:
                    continue

                if data is None or not hasattr(data, 'pos'):
                    continue

                res = evaluar_complejo_causal(model, data, device, rng, k_pct=TOP_K_PCT)
                if res is not None:
                    d_alta.append(res['alta'])
                    d_baja.append(res['baja'])
                    d_rand.append(res['aleatoria'])

            n_samples = len(d_alta)
            if n_samples == 0:
                continue

            m_alta, std_alta, ic_alta = calcular_ic_95(d_alta)
            m_baja, std_baja, ic_baja = calcular_ic_95(d_baja)
            m_rand, std_rand, ic_rand = calcular_ic_95(d_rand)

            t_stat, p_val = stats.ttest_rel(d_alta, d_rand)

            resultados_agregados.append({
                'lambda': lam, 'seed': seed, 'n': n_samples,
                'delta_alta_media': m_alta, 'delta_alta_ic95': ic_alta,
                'delta_baja_media': m_baja, 'delta_baja_ic95': ic_baja,
                'delta_rand_media': m_rand, 'delta_rand_ic95': ic_rand,
                't_stat': t_stat, 'p_value': p_val
            })

            resumen_por_lambda[lam]['alta'].append(m_alta)
            resumen_por_lambda[lam]['baja'].append(m_baja)
            resumen_por_lambda[lam]['aleatoria'].append(m_rand)

            print(f"OK λ={lam} seed={seed:3d} (n={n_samples}) | "
                  f"ΔS_alta={m_alta:.4f}±{ic_alta:.4f} | "
                  f"ΔS_rand={m_rand:.4f}±{ic_rand:.4f} | "
                  f"ΔS_baja={m_baja:.4f}±{ic_baja:.4f} | p-val={p_val:.4e}")

    # Exportar CSV
    csv_path = os.path.join(CARPETA_SALIDA, "delta_s_causal_split_resultados.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'lambda', 'seed', 'n',
            'delta_alta_media', 'delta_alta_ic95',
            'delta_baja_media', 'delta_baja_ic95',
            'delta_rand_media', 'delta_rand_ic95',
            't_stat', 'p_value'
        ])
        writer.writeheader()
        writer.writerows(resultados_agregados)

    print(f"\n✅ Resultados exportados a: {csv_path}")

    # Gráfica
    try:
        lam_eval = 0.0
        if resumen_por_lambda[lam_eval]['alta']:
            alt_m, _, alt_ic = calcular_ic_95(resumen_por_lambda[lam_eval]['alta'])
            rnd_m, _, rnd_ic = calcular_ic_95(resumen_por_lambda[lam_eval]['aleatoria'])
            low_m, _, low_ic = calcular_ic_95(resumen_por_lambda[lam_eval]['baja'])

            fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
            categorias = ['Alta atención\n(S_alta)', 'Aleatoria\n(S_aleatoria)', 'Baja atención\n(S_baja)']
            medias = [alt_m, rnd_m, low_m]
            errores = [alt_ic, rnd_ic, low_ic]
            colores = ['#d95f02', '#7570b3', '#1b9e77']

            bars = ax.bar(categorias, medias, yerr=errores, capsize=6, color=colores, alpha=0.85, edgecolor='black', linewidth=1.2)

            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                            xytext=(0, 0), textcoords="offset points",
                            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

            ax.set_ylabel(r'Efecto de la intervención $\Delta S = |\hat{y}(G) - \hat{y}(G \setminus S)|$ ($pK$)', fontsize=11)
            ax.set_title(f'Prueba de Atribución Causal (Partición por Similitud, k = {int(TOP_K_PCT*100)}%)\n(Media ± IC 95% sobre 8 semillas, λ = {lam_eval})', fontsize=12, fontweight='bold', pad=12)
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            ax.set_ylim(0, max(medias) * 1.35)

            plt.tight_layout()
            fig_path = os.path.join(CARPETA_SALIDA, "delta_s_intervencion_causal_split.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfica exportada a: {fig_path}")

    except Exception as e:
        print(f"⚠️ No se pudo generar la gráfica: {e}")

if __name__ == "__main__":
    main()