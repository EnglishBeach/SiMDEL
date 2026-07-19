"""Converter functions."""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd
from rdkit import Chem as rdChem

from simdel import _deps, _utils
from simdel._parsers import index_parser
from simdel._wrappers import gmx, plumed


@_deps.require(gmx)
def pdb2gro(pdb: Path, workdir: Path) -> Path:
    """Convert geometry .pdb to .gro file by gromacs editconf.

    :param pdb: Geometry .pdb file
    :param workdir: Workdir path
    :return: Converted .gro file
    """
    workdir.mkdir(parents=True, exist_ok=True)
    copied_pdb = workdir / pdb.name
    if copied_pdb != pdb:
        _utils.backup(copied_pdb)
        shutil.copy(pdb, copied_pdb)

    return gmx.editconf(
        workdir=workdir,
        geometry=copied_pdb,
        out_fname=f"{pdb.stem}.gro",
    )


@_deps.require(gmx)
def gro2pdb(gro: Path, workdir: Path) -> Path:
    """Convert geometry .gro to .pdb file by gromacs editconf.

    :param gro: Geometry .gro file
    :param workdir: Workdir path
    :return: Converted .pdb file
    """
    workdir.mkdir(parents=True, exist_ok=True)
    copied_gro = workdir / gro.name
    if copied_gro != gro:
        _utils.backup(copied_gro)
        shutil.copy(gro, copied_gro)

    return gmx.editconf(
        geometry=copied_gro,
        out_fname=f"{gro.stem}.pdb",
        workdir=workdir,
    )


def split_sdf(
    sdf: Path,
    workdir: Path,
) -> dict[str, Path]:
    """Split molecules .sdf file to separate molecule .sdf files with names (l00, l01...).

    :param sdf: Molecules .sdf file path
    :param workdir: Workdir path
    :return: Map: ligand name - .sdf file path
    """
    workdir.mkdir(parents=True, exist_ok=True)
    mols: list[rdChem.rdchem.Mol] = []
    with rdChem.SDMolSupplier(sdf.as_posix(), sanitize=True, removeHs=False) as sdf_system:
        mols = [rdChem.rdchem.Mol(sdf_mol) for sdf_mol in sdf_system]

    lig_paths = {}
    for i, ligand in enumerate(mols):
        lig_name: str = ligand.GetProp("_Name")
        lig_path = workdir / f"{i}.sdf"
        with rdChem.SDWriter(lig_path.as_posix()) as writer:
            writer.write(ligand)
        if lig_paths.get(lig_name):
            msg = "Only unique ligand names allowed in sdf"
            raise ValueError(msg)
        lig_paths[lig_name] = lig_path
    return lig_paths


def dump_index(index_file: Path, indexes: dict[str, pd.Series[bool]]) -> Path:
    """Dump selections in .ndx file.

    :param index_file: Output index .ndx file path
    :param indexes: Indexes dict
    :return: Index .ndx file
    """
    lines = []
    for name, selection in indexes.items():
        lines.extend(index_parser.dump_index(name=name, mask=selection))
    _utils.backup(index_file)
    index_file.write_text("\n".join(lines))
    return index_file


def load_index(index_file: Path, n_atoms: int) -> dict[str, pd.Series[bool]]:
    """Load index from file.

    :param index_file: Index .ndx file
    :param n_atoms: Total atoms in system
    :return: Index dict {str, mask}
    """
    return index_parser.parse_index(index=index_file, n_atoms=n_atoms)


def read_plumed(
    fes: Path,
) -> pd.DataFrame:
    """Read FES .dat file.

    :param fes: FES .dat file
    :return: pandas.DataFrame with additional plumed fields
    """
    return plumed.parse_plumed(fes)[0]
