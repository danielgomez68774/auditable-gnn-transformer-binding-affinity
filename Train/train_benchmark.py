import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import random
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import sys

# =========================================================================
#  ANCLA DE RUTAS: Forzar a Python a encontrar la subcarpeta 'Modelo'
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

# Importación segura desde el paquete local
from Model.model import BindingAffinityModel
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn


# =========================================================================
#  CONSTANTE CRÍTICA: semilla del SPLIT train/val (sobre el Refined-sin-Core).
#  DEBE permanecer FIJA en todas las corridas de todas las semillas y
#  ablaciones, para que la partición train/val sea idéntica y las
#  comparaciones entre semillas sean pareadas. NO la cambies entre corridas.
#  (El test NO se sortea: es el Core-285 completo, en carpeta aparte.)
# =========================================================================
SPLIT_SEED = 42

# Peso del término físico. Poner a 0.0 para la ablación "No Physics".
LAMBDA_FISICA = 0.1


def setup_folders(base_path):
    os.makedirs(base_path, exist_ok=True)
    results_path = config.ensure_dir(config.MODELS_DIR)
    os.makedirs(results_path, exist_ok=True)
    return results_path


def write_log(log_path, message, console=True):
    if console:
        print(message)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def init_csv(csv_path, header):
    """Crea el CSV con encabezado si no existe."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")


def append_csv(csv_path, row):
    """Añade una fila (lista de valores) al CSV."""
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in row) + "\n")


def set_seed(seed):
    """Reproducibilidad: fija la inicialización de pesos y el orden de los
    batches. NO afecta al split train/val (ese usa SPLIT_SEED vía un
    generador propio) ni al test (Core-285 fijo)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def guardar_predicciones_por_complejo(model, dataset, device, csv_path,
                                      seed, lambda_fisica, variant):
    """
    Evalua el test complejo por complejo (batch=1) y guarda un CSV con
    pdb_id, y_true, y_pred. Necesario para la estadistica que pide la
    revision (3.8): bootstrap sobre complejos, intervalos de confianza y
    tests pareados entre variantes -- todo se puede calcular DESPUES a
    partir de este CSV, sin reentrenar.
    """
    model.eval()
    from torch_geometric.loader import DataLoader as _DL
    loader = _DL(dataset, batch_size=1, shuffle=False)
    init_csv(csv_path, ["seed", "lambda_fisica", "variant", "pdb_id", "y_true", "y_pred"])
    for data in loader:
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
        pid = data.pdb_id[0] if isinstance(data.pdb_id, (list, tuple)) else data.pdb_id
        data = data.to(device)
        out = model(data)
        yt = float(data.y.view(-1)[0].cpu())
        yp = float(out.view(-1)[0].cpu())
        append_csv(csv_path, [seed, lambda_fisica, variant, pid,
                              f"{yt:.6f}", f"{yp:.6f}"])


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)

        data = data.to(device)
        out = model(data)
        y_true.extend(data.y.view(-1).cpu().numpy())
        y_pred.extend(out.view(-1).cpu().numpy())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]

    r_pearson, _ = pearsonr(y_t, y_p)
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mae = mean_absolute_error(y_t, y_p)
    return rmse, mae, r_pearson


def calcular_perdida_hibrida(pred, target, data, model, physics, lambda_fisica=0.1,
                             d_soft=4.0):
    """
    REGULARIZACIÓN FÍSICA DIFERENCIABLE (informada por la atención).

    La versión anterior penalizaba SOLO las distancias intermoleculares, que son
    coordenadas de entrada fijas; por tanto su gradiente respecto a los parámetros
    del modelo era CERO y no regularizaba nada (el modelo no genera coordenadas).

    Esta versión penaliza la ATENCIÓN que el modelo asigna a contactos
    físicamente inviables. Como los coeficientes de atención son salida del
    modelo y dependen de los parámetros theta, el gradiente NO es nulo: la
    pérdida enseña al modelo a NO concentrar atención en configuraciones que
    violan la física del reconocimiento no covalente.

      - Penalización estérica: castiga atención alta en pares en solapamiento
        (d < 2.0 Å), ponderada por la gravedad del choque (2.0 - d)^2.
      - Penalización de frontera: castiga atención alta en contactos que se
        aproximan al límite del radio de corte (d > d_soft), ponderada por
        (d - d_soft)^2.

    'physics' es el diccionario devuelto por model(..., return_physics=True):
        physics["attn"]       -> [E, heads]  coeficientes de atención (con grad)
        physics["edge_index"] -> [2, E]      aristas dinámicas (cdist <= cutoff)
        physics["dist"]       -> [E]         distancias de esas aristas
    """
    loss_mse = F.mse_loss(pred, target)

    attn = physics["attn"]                 # [E, heads]
    edge_index = physics["edge_index"]     # [2, E]
    dist = physics["dist"]                 # [E]

    # Atención media por arista sobre las cabezas (mantiene el grafo de gradiente)
    if attn is not None and attn.dim() > 1:
        attn_edge = attn.mean(dim=-1)      # [E]
    else:
        attn_edge = attn.view(-1) if attn is not None else None

    loss_fisica = torch.tensor(0.0, device=pred.device)
    if attn_edge is not None and edge_index.numel() > 0:
        row, col = edge_index[0], edge_index[1]
        # Filtrado intermolecular vía molcode (última columna, con signo ±)
        m1, m2 = data.x[row, -1], data.x[col, -1]
        mask_inter = (m1 * m2 < 0)

        if mask_inter.any():
            d_int = dist[mask_inter]              # [Ei]  (constante, sirve de peso)
            a_int = attn_edge[mask_inter]         # [Ei]  (depende de theta -> grad)

            # Castigo A: atención en solapamiento estérico (d < 2.0 Å)
            w_steric = torch.clamp(2.0 - d_int, min=0.0) ** 2      # peso geométrico
            loss_esterica = torch.mean(a_int * w_steric)

            # Castigo B: atención en la zona externa (d > d_soft, hacia el corte)
            w_boundary = torch.clamp(d_int - d_soft, min=0.0) ** 2
            loss_frontera = torch.mean(a_int * w_boundary)

            loss_fisica = loss_esterica + loss_frontera

    return loss_mse + (lambda_fisica * loss_fisica), loss_mse.detach(), loss_fisica.detach()


def train_one_epoch(model, loader, optimizer, model_param_ref, device,
                    lambda_fisica=0.1):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_phys = 0.0

    for data in loader:
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)

        data = data.to(device)
        optimizer.zero_grad()

        # Pedimos la física: atención + edge_index + distancias (todo con grad)
        prediction, physics = model(data, return_physics=True)

        loss, mse_val, phys_val = calcular_perdida_hibrida(
            prediction.view(-1), data.y.view(-1), data, model_param_ref,
            physics, lambda_fisica=lambda_fisica
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += float(mse_val)
        total_phys += float(phys_val)

    n = len(loader)
    return total_loss / n, total_mse / n, total_phys / n


@torch.enable_grad()
def auditar_gradientes(model, loader, device, lambda_fisica=0.1, n_batches=10):
    """
    EVIDENCIA PARA LA CRÍTICA 3.1: mide, en varios lotes, la norma del gradiente
    que CADA término de la pérdida (MSE y física) produce sobre los parámetros
    entrenables. Si ||grad(L_fisica)|| > 0, el término físico SÍ regulariza.

    Devuelve (norma_media_mse, norma_media_fisica, n_params_con_grad_fisica).
    """
    model.train()
    norms_mse, norms_phys, params_touched = [], [], []

    for i, data in enumerate(loader):
        if i >= n_batches:
            break
        if not hasattr(data, 'batch') or data.batch is None:
            data.batch = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
        data = data.to(device)

        # --- Gradiente SOLO del término físico ---
        model.zero_grad(set_to_none=True)
        pred, physics = model(data, return_physics=True)
        _, _, _ = calcular_perdida_hibrida(pred.view(-1), data.y.view(-1), data,
                                           model, physics, lambda_fisica=lambda_fisica)
        # Reconstruir el término físico puro para retropropagarlo aislado:
        attn = physics["attn"]
        ei = physics["edge_index"]
        dist = physics["dist"]
        attn_edge = attn.mean(dim=-1) if (attn is not None and attn.dim() > 1) else attn.view(-1)
        row, col = ei[0], ei[1]
        m1, m2 = data.x[row, -1], data.x[col, -1]
        mask_inter = (m1 * m2 < 0)
        phys_term = torch.tensor(0.0, device=device)
        if mask_inter.any():
            d_int = dist[mask_inter]
            a_int = attn_edge[mask_inter]
            w_s = torch.clamp(2.0 - d_int, min=0.0) ** 2
            w_b = torch.clamp(d_int - 4.0, min=0.0) ** 2
            phys_term = torch.mean(a_int * w_s) + torch.mean(a_int * w_b)
        (lambda_fisica * phys_term).backward()

        g_phys = 0.0
        n_touched = 0
        for p in model.parameters():
            if p.grad is not None:
                gn = p.grad.detach().norm().item()
                g_phys += gn ** 2
                if gn > 1e-12:
                    n_touched += 1
        norms_phys.append(g_phys ** 0.5)
        params_touched.append(n_touched)

        # --- Gradiente SOLO del término MSE (referencia) ---
        model.zero_grad(set_to_none=True)
        pred2 = model(data)
        mse = F.mse_loss(pred2.view(-1), data.y.view(-1))
        mse.backward()
        g_mse = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g_mse += p.grad.detach().norm().item() ** 2
        norms_mse.append(g_mse ** 0.5)

    model.zero_grad(set_to_none=True)
    import numpy as _np
    return (_np.mean(norms_mse), _np.mean(norms_phys),
            int(_np.mean(params_touched)) if params_touched else 0)


class PDBbindDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
        self.root_dir = root_dir
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.pt')]

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        if data is not None and hasattr(data, 'y'):
            data.y = data.y.view(-1, 1).float()
        # Adjuntar el identificador (nombre de archivo sin .pt) para poder
        # guardar predicciones por-complejo en la evaluacion del test.
        if data is not None:
            data.pdb_id = self.file_list[idx].replace(".pt", "")
        return data


if __name__ == '__main__':
    # -------------------------------------------------------------------
    # Semilla parametrizable: cambia SOLO esto entre corridas.
    #   python train.py --seed 7
    #   python train.py --seed 45     (luego 64, 123)
    # o vía variable de entorno:  set TRAIN_SEED=7  &&  python train.py
    # -------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=int(os.environ.get('TRAIN_SEED', 42)))
    parser.add_argument('--lambda_fisica', type=float,
                        default=float(os.environ.get('LAMBDA_FISICA', LAMBDA_FISICA)),
                        help="Peso del termino fisico. 0.0 = ablacion No Physics.")
    parser.add_argument('--variant', type=str,
                        default=os.environ.get('MODEL_VARIANT', 'full'),
                        help="Etiqueta de la variante: full / no_physics / no_gating / no_rbf / no_lpe")
    parser.add_argument('--solo_auditar', action='store_true',
                        help="Solo ejecuta la auditoria de gradientes (3.1) y termina, sin entrenar.")
    args = parser.parse_args()
    SEED = args.seed
    LAMBDA_FISICA = args.lambda_fisica          # sobreescribe la constante global
    model_variant = args.variant

    set_seed(SEED)  # varía init de pesos + shuffle de batches (NO el split ni el test)

    base_dir = str(config.DATA_ROOT)
    results_dir = config.ensure_dir(config.MODELS_DIR)

    # El tag incluye semilla, lambda y variante para que NADA se pise entre corridas
    # (checkpoints, logs y CSV por epoca quedan separados por configuracion).
    lam_tag = f"lam{LAMBDA_FISICA}".replace(".", "p")
    tag = f"REFINEDsinCore_CASFtest_seed_{SEED}_{lam_tag}_{model_variant}"
    log_file = os.path.join(results_dir, f"training_log_{tag}.txt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------------
    # Registro de entorno (reproducibilidad, criticas 3.3 y 3.12).
    # -------------------------------------------------------------------
    try:
        import torch_geometric as _pyg
        pyg_ver = _pyg.__version__
    except Exception:
        pyg_ver = "N/D"
    import platform as _plat
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    write_log(log_file, f"[ENTORNO] Python {_plat.python_version()} | torch {torch.__version__} | "
                        f"torch_geometric {pyg_ver} | numpy {np.__version__} | "
                        f"GPU: {gpu_name} | device: {device}")
    #   - Refined SIN Core-285  -> se divide en train/val
    #   - Core-285 (CASF-2016)  -> test FIJO, entero (comparable con SIGN)
    # -------------------------------------------------------------------
    ruta_train_pool = str(config.GRAFOS_REFINED)   # Refined sin Core
    ruta_test_core  = str(config.GRAFOS_CASF_CORE285)  # Core-285

    pool_dataset = PDBbindDataset(root_dir=ruta_train_pool)   # train + val
    test_set     = PDBbindDataset(root_dir=ruta_test_core)    # test (Core-285 completo)

    if len(pool_dataset) == 0:
        raise RuntimeError(f"No hay grafos en {ruta_train_pool}. Corre primero el preprocesamiento.")
    if len(test_set) == 0:
        raise RuntimeError(f"No hay grafos de test en {ruta_test_core}. Genera el Core-285 en el preprocesamiento.")
    if len(test_set) != 285:
        print(f"⚠️ El test tiene {len(test_set)} complejos (se esperaban 285 para CASF-2016 Core). Continúo igualmente.")

    # Split train/val SOLO sobre el pool Refined-sin-Core (el test NO se toca).
    # Fracción de validación (ajústala si lo deseas):
    VAL_FRAC = 0.10
    val_size = int(VAL_FRAC * len(pool_dataset))
    train_size = len(pool_dataset) - val_size

    split_gen = torch.Generator().manual_seed(SPLIT_SEED)  # FIJO en todas las semillas
    train_set, val_set = torch.utils.data.random_split(
        pool_dataset, [train_size, val_size], generator=split_gen
    )

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False, pin_memory=True)  # Core-285

    # Traducir la variante de ablación a las flags del modelo.
    # 'full' activa todo; cada 'no_*' desactiva el componente correspondiente.
    # (No Physics NO es una flag del modelo: se controla con lambda_fisica=0.)
    use_gating = (model_variant != "no_gating")
    use_rbf    = (model_variant != "no_rbf")
    use_lpe    = (model_variant != "no_lpe")
    if model_variant == "no_physics":
        LAMBDA_FISICA = 0.0  # la ablacion de fisica se hace anulando el termino
    write_log(log_file, f"[VARIANTE] {model_variant} | use_gating={use_gating} "
                        f"use_rbf={use_rbf} use_lpe={use_lpe} | lambda_fisica={LAMBDA_FISICA}")

    model = BindingAffinityModel(
        d_model=512, rbf_dim=64, lpe_dim=15,
        use_rbf=use_rbf, use_lpe=use_lpe, use_gating=use_gating
    ).to(device)

    # -------------------------------------------------------------------
    # AUDITORÍA DE GRADIENTES (evidencia para la crítica 3.1).
    # Verifica, en el modelo recién inicializado, que el término físico
    # produce gradiente NO nulo sobre los parámetros. Se ejecuta siempre
    # (es rápida). Con --solo_auditar termina aquí sin entrenar.
    # -------------------------------------------------------------------
    gm, gp, nt = auditar_gradientes(model, train_loader, device,
                                    lambda_fisica=LAMBDA_FISICA, n_batches=10)
    write_log(log_file,
              f"[AUDIT 3.1] ||grad MSE|| = {gm:.4e} | "
              f"||grad Fisica|| = {gp:.4e} | "
              f"params con grad de la fisica = {nt} | lambda = {LAMBDA_FISICA}")
    if gp <= 1e-12 and LAMBDA_FISICA > 0:
        write_log(log_file, "  ⚠️ ADVERTENCIA: el gradiente fisico es ~0. Revisar la perdida.")
    else:
        write_log(log_file, "  ✅ El termino fisico produce gradiente no nulo (regulariza).")

    if args.solo_auditar:
        write_log(log_file, "Modo --solo_auditar: no se entrena. Fin.")
        sys.exit(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    # -------------------------------------------------------------------
    # Programador de LR ÚNICO para la fase pre-SWA (sin overrides manuales).
    # MultiStepLR con milestones=[50, 100] y gamma=0.5:
    #   épocas 1-49  -> 1e-4
    #   épocas 50-99 -> 5e-5
    #   épocas 100+  -> 2.5e-5   (hasta que entra SWA)
    # -------------------------------------------------------------------
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    # SWA: promedia pesos en las épocas finales (145 -> 160 = 16 épocas).
    swa_model = AveragedModel(model)
    swa_start = 115
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5)
    epochs = 130

    best_ckpt = os.path.join(results_dir, f"best_model_{tag}.pt")
    swa_ckpt = os.path.join(results_dir, f"best_model_SWA_{tag}.pt")

    # CSV estructurado por época (para reconstruir curvas train vs val sin parsear texto).
    # El 'tag' ya incluye semilla, lambda y variante, así que no se pisan corridas.
    epoch_csv = os.path.join(results_dir, f"epoch_metrics_{tag}.csv")
    init_csv(epoch_csv, [
        "seed", "lambda_fisica", "epoch",
        "train_loss", "train_mse", "train_phys",
        "train_rmse", "train_mae", "train_pearson",
        "val_rmse", "val_mae", "val_pearson", "lr", "time_s"
    ])

    write_log(log_file, f"Entrenamiento HÍBRIDO REGULARIZADO — Refined(sin Core) -> train/val | CASF-2016 Core-285 -> test\n{'=' * 60}")
    write_log(log_file, (f"Pool Refined(sin Core): {len(pool_dataset)} | "
                         f"Train/Val: {train_size}/{val_size} | "
                         f"Test (Core-285): {len(test_set)} | "
                         f"Epocas: {epochs} | LPE: 15 | RBF Bins: 64 | batch: 4 | "
                         f"lr0: 1e-4 | SWA_start: {swa_start} | SPLIT_SEED: {SPLIT_SEED} | "
                         f"seed: {SEED} | lambda_fisica: {LAMBDA_FISICA}"))

    best_val_rmse = float('inf')

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Se envía 'model' como referencia para extraer propiedades internas (p. ej. model.cutoff)
        avg_train_loss, avg_mse, avg_phys = train_one_epoch(
            model, train_loader, optimizer, model, device, lambda_fisica=LAMBDA_FISICA
        )
        val_rmse, val_mae, val_pearson = evaluate_metrics(model, val_loader, device)
        # Métricas de TRAIN (mismas unidades que val) para curvas train-vs-val.
        # Se evalúan cada EVAL_TRAIN_EVERY épocas (y en la última) para no duplicar
        # el tiempo por época: el train es grande y no necesita resolución por época.
        EVAL_TRAIN_EVERY = 5
        if (epoch % EVAL_TRAIN_EVERY == 0) or (epoch == 1) or (epoch == epochs):
            train_rmse, train_mae, train_pearson = evaluate_metrics(model, train_loader, device)
        else:
            train_rmse, train_mae, train_pearson = float('nan'), float('nan'), float('nan')

        # Programador según la etapa (SWA a partir de swa_start)
        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        duration = time.time() - start_time

        status = (f"Epoch: {epoch:03d} | Loss: {avg_train_loss:.4f} "
                  f"(MSE: {avg_mse:.4f} | Phys: {avg_phys:.6f}) | "
                  f"Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | "
                  f"Val R: {val_pearson:.4f} | LR: {current_lr:.2e} | Time: {duration:.1f}s")
        write_log(log_file, status)

        # Fila estructurada al CSV (para reconstruir cualquier curva después)
        append_csv(epoch_csv, [
            SEED, LAMBDA_FISICA, epoch,
            f"{avg_train_loss:.6f}", f"{avg_mse:.6f}", f"{avg_phys:.8f}",
            f"{train_rmse:.6f}", f"{train_mae:.6f}", f"{train_pearson:.6f}",
            f"{val_rmse:.6f}", f"{val_mae:.6f}", f"{val_pearson:.6f}",
            f"{current_lr:.3e}", f"{duration:.1f}"
        ])

        # Selección del modelo SOLO por validación (nunca por el test Core-285)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_ckpt)
            write_log(log_file, f" >>> Nuevo récord de Val RMSE: {best_val_rmse:.4f} (Guardado)")

    # -------------------------------------------------------------------
    # Cierre SWA
    # -------------------------------------------------------------------
    write_log(log_file, f"{'=' * 60}\nFinalizando con ensamble SWA...")
    update_bn(train_loader, swa_model, device=device)
    torch.save(swa_model.module.state_dict(), swa_ckpt)

    # -------------------------------------------------------------------
    # EVALUACIÓN FINAL SOBRE EL CORE-285 (CASF-2016).
    # Este es el número comparable con SIGN/PIGNet y el que va al reporte.
    # -------------------------------------------------------------------
    # (a) Modelo seleccionado por validación (mismas flags que la variante entrenada)
    best_model = BindingAffinityModel(
        d_model=512, rbf_dim=64, lpe_dim=15,
        use_rbf=use_rbf, use_lpe=use_lpe, use_gating=use_gating
    ).to(device)
    best_model.load_state_dict(torch.load(best_ckpt, map_location=device))
    te_rmse, te_mae, te_r = evaluate_metrics(best_model, test_loader, device)
    va_rmse, va_mae, va_r = evaluate_metrics(best_model, val_loader, device)

    # Guardar predicciones por-complejo del test (para bootstrap / IC / tests
    # pareados posteriores, sin reentrenar). Un CSV por corrida.
    preds_csv = os.path.join(results_dir, f"test_predictions_{tag}.csv")
    guardar_predicciones_por_complejo(best_model, test_set, device, preds_csv,
                                      SEED, LAMBDA_FISICA, model_variant)
    write_log(log_file, f"Predicciones por-complejo del test guardadas en: {preds_csv}")

    # (b) Modelo SWA (promedio de pesos)
    swa_te_rmse, swa_te_mae, swa_te_r = evaluate_metrics(swa_model, test_loader, device)
    swa_va_rmse, swa_va_mae, swa_va_r = evaluate_metrics(swa_model, val_loader, device)

    write_log(log_file, f"{'=' * 60}")
    write_log(log_file, f"[SEED {SEED}] MEJOR-POR-VALIDACION  ->  "
                        f"CASF-Core285  RMSE: {te_rmse:.4f} | MAE: {te_mae:.4f} | Pearson: {te_r:.4f}")
    write_log(log_file, f"[SEED {SEED}] MEJOR-POR-VALIDACION  ->  "
                        f"VAL           RMSE: {va_rmse:.4f} | MAE: {va_mae:.4f} | Pearson: {va_r:.4f}")
    write_log(log_file, f"[SEED {SEED}] SWA                   ->  "
                        f"CASF-Core285  RMSE: {swa_te_rmse:.4f} | MAE: {swa_te_mae:.4f} | Pearson: {swa_te_r:.4f}")
    write_log(log_file, f"[SEED {SEED}] SWA                   ->  "
                        f"VAL           RMSE: {swa_va_rmse:.4f} | MAE: {swa_va_mae:.4f} | Pearson: {swa_va_r:.4f}")

    # -------------------------------------------------------------------
    # CSV RESUMEN por corrida: una fila por (semilla, lambda). Todas las
    # corridas escriben al MISMO archivo, de modo que al terminar las 4
    # semillas (o el barrido de lambda) puedas calcular media ± std y las
    # bandas de error directamente, sin parsear los .txt.
    # -------------------------------------------------------------------
    summary_csv = os.path.join(results_dir, "summary_all_runs.csv")
    init_csv(summary_csv, [
        "seed", "lambda_fisica", "model_variant",
        "test_rmse", "test_mae", "test_pearson",
        "val_rmse", "val_mae", "val_pearson",
        "swa_test_rmse", "swa_test_mae", "swa_test_pearson",
        "best_epoch_val_rmse"
    ])
    # Etiqueta de la variante (viene de --variant / MODEL_VARIANT, definida arriba):
    # full / no_physics / no_gating / no_rbf / no_lpe
    append_csv(summary_csv, [
        SEED, LAMBDA_FISICA, model_variant,
        f"{te_rmse:.6f}", f"{te_mae:.6f}", f"{te_r:.6f}",
        f"{va_rmse:.6f}", f"{va_mae:.6f}", f"{va_r:.6f}",
        f"{swa_te_rmse:.6f}", f"{swa_te_mae:.6f}", f"{swa_te_r:.6f}",
        f"{best_val_rmse:.6f}"
    ])
    write_log(log_file, f"{'=' * 60}\nResumen añadido a: {summary_csv}")