#!/usr/bin/env python3
"""
====================================================================
EXTRACCIÓN DE SECUENCIAS DE PROTEÍNA (PDBbind) -> FASTA
====================================================================
Lee los archivos {ID}_protein.pdb de PDBbind refined-set y extrae la secuencia
de aminoácidos completa de cada proteína (todas las cadenas concatenadas),
generando un único FASTA para el clustering con MMseqs2.

Solo procesa los complejos que:
  (a) tienen carpeta con _protein.pdb en refined-set, Y
  (b) tienen un grafo .pt generado (para clusterizar exactamente lo que usa
      el modelo).

Uso (en WSL):
    python extraer_secuencias.py

Las rutas se resuelven mediante config.py (relativas al repositorio).
"""

import os
import sys

# --- Biopython ---
try:
    from Bio import SeqIO
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.SeqUtils import seq1
except ImportError:
    print("❌ Falta biopython. Actívalo: conda activate split_env")
    sys.exit(1)

# =========================================================================
# CONFIGURACIÓN (rutas en formato WSL)
# =========================================================================
# Carpeta refined-set con subcarpetas por complejo (1a1e/, 1a4k/, ...)
# --- repo-root bootstrap: make `config` importable from any folder ---
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve()
while not (_root / "config.py").exists() and _root != _root.parent:
    _root = _root.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))
import config

DIR_REFINED = str(config.PDBBIND_REFINED_RAW / "refined-set")
# Carpeta con los grafos .pt (para cruzar: solo clusterizamos lo que tiene grafo)
DIR_GRAFOS  = str(config.GRAFOS_REFINED)
# Salida
FASTA_SALIDA = str(config.SPLIT_DIR / "proteinas_refined.fasta")
MAPA_SALIDA  = str(config.SPLIT_DIR / "mapa_complejos.txt")

# Códigos de 3 letras estándar de aminoácidos (para filtrar heteroátomos/agua)
AA_ESTANDAR = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL',
    'MSE','SEC','PYL'  # selenometionina y otros comunes
}


def ids_con_grafo(dir_grafos):
    """IDs de complejo que tienen un grafo .pt generado."""
    if not os.path.isdir(dir_grafos):
        print(f"⚠️ No existe la carpeta de grafos: {dir_grafos}")
        return None
    ids = set()
    for f in os.listdir(dir_grafos):
        if f.endswith('.pt'):
            ids.add(f.replace('.pt', ''))
    return ids


def extraer_secuencia(pdb_path, parser, ppb):
    """Extrae la secuencia de aminoácidos (todas las cadenas concatenadas)."""
    try:
        estructura = parser.get_structure('x', pdb_path)
    except Exception as e:
        return None
    # Concatenar la secuencia de todos los polipéptidos de todas las cadenas
    seqs = []
    for pp in ppb.build_peptides(estructura):
        seqs.append(str(pp.get_sequence()))
    if not seqs:
        return None
    return "".join(seqs)


def main():
    os.makedirs(os.path.dirname(FASTA_SALIDA), exist_ok=True)

    # 1. Qué complejos tienen grafo
    ids_grafo = ids_con_grafo(DIR_GRAFOS)
    if ids_grafo is not None:
        print(f"Grafos .pt encontrados: {len(ids_grafo)}")
    else:
        print("Sin filtro por grafos (se procesan todas las carpetas).")

    # 2. Recorrer carpetas de refined-set
    if not os.path.isdir(DIR_REFINED):
        print(f"❌ No existe refined-set: {DIR_REFINED}")
        return

    carpetas = [d for d in os.listdir(DIR_REFINED)
                if os.path.isdir(os.path.join(DIR_REFINED, d))]
    # Excluir carpetas que no son complejos (index, readme, etc.)
    carpetas = [d for d in carpetas if d not in ('index', 'readme')]
    print(f"Carpetas de complejos en refined-set: {len(carpetas)}")

    parser = PDBParser(QUIET=True)
    ppb = PPBuilder()

    registros = []   # (id, secuencia)
    sin_pdb = []
    sin_seq = []
    sin_grafo = []

    for i, cid in enumerate(sorted(carpetas)):
        if ids_grafo is not None and cid not in ids_grafo:
            sin_grafo.append(cid)
            continue
        pdb_path = os.path.join(DIR_REFINED, cid, f"{cid}_protein.pdb")
        if not os.path.exists(pdb_path):
            sin_pdb.append(cid)
            continue
        seq = extraer_secuencia(pdb_path, parser, ppb)
        if not seq or len(seq) < 5:
            sin_seq.append(cid)
            continue
        registros.append((cid, seq))
        if (i + 1) % 500 == 0:
            print(f"  procesados {i+1}/{len(carpetas)}...")

    # 3. Escribir FASTA
    with open(FASTA_SALIDA, 'w') as f:
        for cid, seq in registros:
            f.write(f">{cid}\n{seq}\n")

    # 4. Mapa (por si se necesita)
    with open(MAPA_SALIDA, 'w') as f:
        f.write("complejo\tlongitud_secuencia\n")
        for cid, seq in registros:
            f.write(f"{cid}\t{len(seq)}\n")

    # 5. Reporte
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Secuencias extraídas:        {len(registros)}")
    print(f"Sin _protein.pdb:            {len(sin_pdb)}")
    print(f"Sin secuencia válida:        {len(sin_seq)}")
    if ids_grafo is not None:
        print(f"Carpetas sin grafo (omitidas): {len(sin_grafo)}")
        # complejos con grafo pero sin secuencia (importante detectarlos)
        ids_extraidos = {cid for cid, _ in registros}
        faltan = ids_grafo - ids_extraidos
        if faltan:
            print(f"\n⚠️ {len(faltan)} complejos con grafo pero SIN secuencia extraída:")
            for x in sorted(list(faltan))[:20]:
                print(f"    {x}")
            if len(faltan) > 20:
                print(f"    ... y {len(faltan)-20} más")
    print(f"\nFASTA guardado en: {FASTA_SALIDA}")
    print(f"Mapa guardado en:  {MAPA_SALIDA}")


if __name__ == "__main__":
    main()
