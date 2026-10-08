"""
config.py - Single source of truth for every path in this project.

All paths derive from the repository root (the folder containing this file),
so no script needs an absolute or machine-specific path. Heavy data
(preprocessed graphs and trained checkpoints) is NOT stored in the repo.

Default layout (everything stays inside the repo):
    <repo>/data/            preprocessed graph datasets (download from Zenodo)
    <repo>/trained_models/  checkpoints written by training / read by analysis
    <repo>/outputs/         figures, CSVs and logs produced by the scripts

Relocate heavy data without editing code via environment variables:
    BA_DATA_ROOT    -> where the graph datasets live
    BA_MODELS_ROOT  -> where checkpoints live
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Make the repo importable (so `from Model.model import ...` works anywhere).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Heavy-data roots (inside the repo by default; override with env vars) ---
DATA_ROOT   = Path(os.environ.get("BA_DATA_ROOT",   PROJECT_ROOT / "data"))
MODELS_DIR  = Path(os.environ.get("BA_MODELS_ROOT", PROJECT_ROOT / "trained_models"))
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# --- Preprocessed graph datasets (place the Zenodo folders inside DATA_ROOT) ---
GRAFOS_REFINED                = DATA_ROOT / "Graphs_refined_corrected"
GRAFOS_CASF_CORE285           = DATA_ROOT / "Graphs_CASF_core285"
GRAFOS_CASF_DINAMICOS         = DATA_ROOT / "Graphs_CASF_dynamic"
GRAFOS_CASF_PROT_COMPLETA     = DATA_ROOT / "Graphs_CASF_dynamic_full_protein"
GRAFOS_CSAR                   = DATA_ROOT / "Graphs_CSAR_dynamic"
GRAFOS_CROSSDOCKED_MASIVO     = DATA_ROOT / "Graphs_CrossDocked_massive_dynamic"
GRAFOS_CROSSDOCKED_INDIVIDUAL = DATA_ROOT / "Graphs_CrossDocked_individual"

# --- Raw datasets (only needed to RE-RUN preprocessing from scratch) ---
PDBBIND_REFINED_RAW = DATA_ROOT / "PDBbind_v2020_refined"
CASF_CORESET_RAW    = DATA_ROOT / "CASF-2016" / "coreset"
INDEX_REFINED       = PDBBIND_REFINED_RAW / "refined-set" / "index" / "INDEX_refined_data.2020"

# --- Files that ARE in the repository ---
SPLIT_DIR = PROJECT_ROOT / "Pre-processing" / "split_proteins"   # train/val/test .txt (Protocol B)


def ensure_dir(path):
    """Create `path` (and parents) if needed; return it as a string."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)
