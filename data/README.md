# data/

Heavy preprocessed graph datasets live here. They are **not** tracked by git
(see `.gitignore`); download them from the Zenodo archive linked in the paper
and extract the folders directly into this directory. All code resolves these
paths through `config.py`, so nothing needs editing.

Expected folders (names must match exactly):

| Folder | Used by |
|---|---|
| `Graphs_refined_corrected` | Protocol A train/val pool; Protocol B (via split lists) |
| `Graphs_CASF_core285` | Protocol A test; chemistry probe; cross-dataset consistency |
| `Graphs_CASF_dynamic` | distance / angular / LPE probes; 1bcu case study |
| `Graphs_CASF_dynamic_full_protein` | 1bcu complete-protein case study |
| `Graphs_CSAR_dynamic` | CSAR-HiQ cross-dataset consistency |
| `Graphs_CrossDocked_massive_dynamic` | CrossDocked cross-dataset consistency |
| `Graphs_CrossDocked_individual` | conformational sensitivity (pose stress) |

To re-run preprocessing from scratch instead, place the raw datasets here
(`PDBbind_v2020_refined/`, `CASF-2016/coreset/`) and run the scripts in
`Pre-processing/`.

You may keep the data elsewhere by setting the `BA_DATA_ROOT` environment
variable to that location.
