# data/

Preprocessed 3D protein–ligand interaction graphs (PyTorch Geometric format) live
here. They are **not** tracked by git (see `.gitignore`) and are **not** distributed
on Zenodo. Instead, they are generated locally from the public datasets using the
scripts in `Pre-processing/` (see the main README). All code resolves these paths
through `config.py`, so nothing needs editing once the folders are in place.

**How to obtain them:** download the raw datasets from their original sources
(PDBbind v2020, CASF-2016, CrossDocked2020, CSAR-HiQ — under their respective terms
of use), place them where `config.py` expects them, and run the preprocessing
scripts. This produces the following folders:

| Folder | Used by |
|---|---|
| `Graphs_refined_corrected` | Protocol A train/val pool; Protocol B (via split lists) |
| `Graphs_CASF_core285` | Protocol A test; chemistry probe; cross-dataset consistency |
| `Graphs_CASF_dynamic` | distance / angular / LPE probes; 1bcu case study |
| `Graphs_CASF_dynamic_full_protein` | 1bcu complete-protein case study |
| `Graphs_CSAR_dynamic` | CSAR-HiQ cross-dataset consistency |
| `Graphs_CrossDocked_massive_dynamic` | CrossDocked cross-dataset consistency |
| `Graphs_CrossDocked_individual` | conformational sensitivity (pose stress) |

You may keep the data elsewhere by setting the `BA_DATA_ROOT` environment variable.

Note: the **trained model checkpoints** (not the graphs) are archived on Zenodo and
go in `trained_models/` — see the main README.