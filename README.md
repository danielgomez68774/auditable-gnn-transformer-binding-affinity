# Physics-Inspired Geometric Biases in a GNN–Transformer Architecture for Auditable 3D Protein–Ligand Binding Affinity Prediction

Official PyTorch implementation for the manuscript **"Physics-Inspired Geometric Biases in a GNN–Transformer Architecture for Auditable 3D Protein–Ligand Binding Affinity Prediction"**.

---

## 💻 Hardware & PyTorch Setup

This project was developed and tested using a preview/preview-capable PyTorch build required for modern GPU architectures (e.g., NVIDIA GeForce RTX 50xx series / Blackwell architecture).

### Environment Specs
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **CUDA Version**: 12.8
- **NVIDIA Driver**: 610.88
- **PyTorch**: `2.11.0+cu128`
- **PyTorch Geometric**: `2.7.0` (`torch-scatter==2.1.2+pt211cu128`)

---

## ⚙️ Installation

### 1. Create Virtual Environment
```bash
# Clone the repository
git clone [https://github.com/USER/REPO.git](https://github.com/USER/REPO.git)
cd REPO

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 2. Install PyTorch & PyG
Install PyTorch separately before the rest of the requirements using the PyTorch CUDA 12.8 wheel index:

```bash
# Install PyTorch preview/build with CUDA 12.8 support
pip install --pre torch --index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)

# Install PyTorch Geometric and PyG extensions
pip install torch-geometric torch-scatter -f [https://data.pyg.org/whl/torch-2.11.0+cu128.html](https://data.pyg.org/whl/torch-2.11.0+cu128.html)
```

### 3. Install Pinned Dependencies
The remaining dependencies are pinned in `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Repository Structure

```text
├── config.py                     # Centralized project path resolution
├── Model/
│   ├── __init__.py
│   └── model.py                  # Core hybrid GNN-Transformer architecture
├── Pre-processing/
│   ├── CASF-2016/
│   │   ├── pre-processing-casf.py
│   │   └── pre-processing-full-protein.py
│   ├── CrossDocked/
│   │   ├── pre-processing-crossdocked-masive.py
│   │   └── pre-processing-docked.py
│   ├── CSAR_Hi-Q/
│   │   └── pre-processing-csar.py
│   ├── PDBBind_Refined_2020/
│   │   ├── diagnose_complexes.py
│   │   └── pre-processing.py     # Converts PDB/MOL2 files to PyTorch Geometric graphs
│   └── split_proteins/
│       ├── extract_sequences.py
│       ├── mmseqs2_commands.txt  # MMseqs2 commands for 30% sequence identity split
│       ├── split_by_similarity.py
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── Train/
│   ├── train_benchmark.py        # Training & evaluation on CASF-2016 benchmark (Protocol A)
│   └── train_simsplit.py         # Training & evaluation on 30% sequence split (Protocol B)
└── Interpretability/
    ├── Molecular-Analysis/
    │   ├── RL_multimodel.py      # Ligand Attentional Reconstruction (RL) metric
    │   ├── RL_multimodel_split.py
    │   ├── RL_multimodel_ablations.py
    │   ├── RL_causal_analysis.py
    │   ├── RL_causal_analysis_split.py
    │   ├── dataset_bonds_analysis_v2.py
    │   ├── generate_chimera_full_protein.py
    │   └── generate_chimerax.py  # Export attention B-factor files for ChimeraX
    ├── Proximity/                # Distance response probes & contact zone analysis
    ├── Angular/                  # Directional anisotropy & local coherence probes
    ├── Chemistry/                # Chemical sensitivity & atomic descriptor perturbations
    ├── LPE/                      # Spectral Laplacian positional encoding audits
    └── Stress/                   # Conformational sensitivity & pose noise stress tests
```

---

## ⚙️ Project paths (`config.py`)

All paths are centralized in **`config.py`** at the repository root and derived
from the repo location, so **no script contains a machine-specific path** and
nothing is read from outside the repository.

- Preprocessed graphs are expected in `data/`; they are generated from the public datasets with the preprocessing scripts (see below and `data/README.md`).
- Training writes checkpoints to `trained_models/`; the checkpoints used in the paper can be downloaded from Zenodo.
- Figures, CSVs and logs are written to `outputs/`.

To keep the heavy data elsewhere, set environment variables (no code changes):

```bash
export BA_DATA_ROOT=/path/to/graphs        # where the Grafos_* folders live
export BA_MODELS_ROOT=/path/to/checkpoints # where trained_models live
```

Scripts can be run from any folder; each one locates `config.py` automatically.

## 🔄 Pre-processing & Data Preparation

1. **Download Datasets** (from their original sources, under their respective terms of use; the datasets are **not** redistributed in this repository):
   - **PDBbind v2020 (General & Refined sets)**: http://www.pdbbind.org.cn/
   - **CASF-2016 Benchmark (Core set)**: distributed with PDBbind
   - **CrossDocked2020**: https://bits.csb.pitt.edu/files/crossdock2020/
   - **CSAR-HiQ NRC Set**: available from the CSAR benchmark repository

2. **Graph Conversion**:
   Run graph preprocessing to compute atomic features, dynamic $4.5\,\text{Å}$ edge cutoffs, and 15-eigenvector LPE signatures:
   ```bash
   python Pre-processing/"PDBBind Refined 2020"/pre-processing.py
   ```

3. **Sequence Identity Clustering (30% Split)**:
   Extract protein sequences and split whole sequence clusters using MMseqs2:
   ```bash
   python Pre-processing/split_proteins/extract_sequences.py
   # Execute MMseqs2 commands listed in Pre-processing/split_proteins/mmseqs2_commands.txt
   python Pre-processing/split_proteins/split_by_similarity.py
   ```
4. **Reproducing results without retraining** (optional):
   The trained checkpoints for all eight seeds are archived on Zenodo
   (https://doi.org/<ZENODO-DOI>). Download and extract them into `trained_models/`
   to run the evaluation and interpretability analyses directly, without repeating
   the preprocessing and training steps above.

---

## 🚀 Training & Evaluation

### Protocol A: CASF-2016 Core Set Benchmark (8-Seed Protocol)
To train and evaluate the model on the processable 178-complex subset of CASF-2016 across 8 seeds:
```bash
python Train/train_benchmark.py
```

### Protocol B: Out-of-Distribution 30% Sequence Identity Split
To train and evaluate under the similarity-controlled partition:
```bash
python Train/train_simsplit.py
```

---

## 🔍 Interpretability & XAI Probes

Run the suite of interpretability scripts to reproduce the paper's XAI findings:

- **Ligand Attentional Reconstruction ($RL$)**:
  ```bash
  python Interpretability/Molecular-Analysis/RL_multimodel.py
  ```
- **Chemical Sensitivity Perturbations**:
  ```bash
  python Interpretability/Chemistry/chemical_sensitivity.py
  ```
- **Distance & Angular Probes**:
  ```bash
  python Interpretability/Proximity/physical_distance_test.py
  python Interpretability/Angular/hybridization_control.py
  ```
- **ChimeraX Visualizations**:
  Generate 3D attention-mapped `.pdb` files for ChimeraX rendering:
  ```bash
  python Interpretability/Molecular-Analysis/generate_chimerax.py
  ```

---

## 📜 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@article{gomez2026physics,
  title={Physics-Inspired Geometric Biases in a GNN--Transformer Architecture for Auditable 3D Protein--Ligand Binding Affinity Prediction},
  author={Gomez Bohorquez, Daniel and Tabares Soto, Reinel and Bravo Ortiz, Mario Alejandro},
  journal={Journal of Chemical Information and Modeling},
  year={2026}
}
```

---

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.