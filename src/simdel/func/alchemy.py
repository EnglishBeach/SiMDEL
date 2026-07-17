"""Alchemy transformation functions."""

import itertools
import json
from pathlib import Path
import shutil

from simdel import chem
from simdel._misc import context, utils
from simdel._wrappers import gromacs, pmx

from . import converters


@context.require_pmx
def create_hybrids(
    workdir: Path,
    system_a: chem.System,
    system_b: chem.System,
) -> tuple[chem.System, chem.System]:
    """REQUIRE PMX!
    Align and create hybrid systems with one hybrid molecule topology, but different geometry
    System must contain only 1 molecule inside.

    :param workdir: Workdir path
    :param system_a: Reference system - A
    :param system_b: Probe system - B
    :return: Hybrid A, hybrid B systems
    """
    if len(system_a.molecules) != len(system_b.molecules) != 1:
        msg = (
            f"No 1 molecules in systems:\n"
            f"systemA: {system_a.molecules}\n"
            f"systemB: {system_b.molecules}\n"
        )
        raise ValueError(msg)
    suffixA, suffixB = "A", "B"
    workdir.mkdir(parents=True, exist_ok=True)

    nameA, ffA, itpA, geoA = _split_system_dump(
        system=system_a, workdir=workdir / "ligA", new_name="MOL"
    )
    nameB, ffB, itpB, geoB = _split_system_dump(
        system=system_b, workdir=workdir / "ligB", new_name="MOL"
    )

    aligned_dir = workdir / "aligned"
    aligned_dir.mkdir(exist_ok=True, parents=True)
    # TODO: extract
    align_files = pmx.align_ligands(
        workdir=aligned_dir,
        geometry_a=geoA,
        geometry_b=geoB,
    )

    hybrids_dir = workdir / "hybrids"
    hybrids_dir.mkdir(exist_ok=True, parents=True)

    hybrid_files = pmx.hybridize_ligands(
        workdir=hybrids_dir,
        topology_name="MOL",
        geometry_a=align_files.aligned_pdb_a,
        itp_a=itpA,
        ff_a=ffA,
        geometry_b=align_files.aligned_pdb_b,
        itp_b=itpB,
        ff_b=ffB,
        pair_ab=align_files.pair_ab,
    )
    geo_a = gromacs.editconf(
        workdir=hybrids_dir,
        geometry=hybrid_files.geometry_a,
        out_fname=f"{hybrid_files.geometry_a.stem}.gro",
    )
    hybridA = (
        chem.System.load(
            top=hybrid_files.topology,
            gro=geo_a,
        )
        .set_info(**dict(system_a.info))
        .rename_topologies({"MOL": f"{nameA}{suffixA}"})
        .rename(system_a.name)
    )

    geo_b = gromacs.editconf(
        workdir=hybrids_dir,
        geometry=hybrid_files.geometry_b,
        out_fname=f"{hybrid_files.geometry_b.stem}.gro",
    )
    hybridB = (
        chem.System.load(
            top=hybrid_files.topology,
            gro=geo_b,
        )
        .set_info(**dict(system_b.info))
        .rename_topologies({"MOL": f"{nameB}{suffixB}"})
        .rename(system_b.name)
    )
    return hybridA, hybridB


@context.require_mamba
def gen_alchemy_graph(workdir: Path, sdf_list: list[Path]) -> list[tuple[Path, Path]]:
    """REQUIRE MAMBA DEPENDENCIES!
    Generate pair transformations graph, uses LOMAP when n ligands >=6.

    :param workdir: Workdir path
    :param sdf_list: List of .sdf files (1 .sdf per molecule)
    :return: List of pairs (.sdf 1, .sdf 2)
    """
    import lomap  # type: ignore # noqa: PLC0415

    min_pairs_n_lomap_use = 6

    workdir.mkdir(parents=True, exist_ok=True)
    if len(sdf_list) < min_pairs_n_lomap_use:
        return list(itertools.combinations(sdf_list, 2))

    sdf_map = {}
    for sdf in sdf_list:
        shutil.copy(src=sdf, dst=workdir / f"{sdf.name}")
        sdf_map[sdf.stem] = sdf

    mol_db = lomap.DBMolecules(directory=workdir.as_posix(), radial=True)
    mol_db.build_matrices()
    nx_graph = mol_db.build_graph()
    pairs = []
    for s, e in nx_graph.edges:  # type: ignore
        sdf_a = Path(mol_db.dic_mapping[s]).stem
        sdf_b = Path(mol_db.dic_mapping[e]).stem

        pairs.append((sdf_map[sdf_a], sdf_map[sdf_b]))
    (workdir / "pairs.json").write_text(json.dumps(pairs))
    return pairs


def _split_system_dump(
    workdir: Path,
    system: chem.System,
    new_name: str,
) -> tuple[str, Path, Path, Path]:
    """Save system in separate files: .ff.itp, .itp, hack function for hybridization.
    Take FIRST molecule from system.

    :param workdir: Workdir path
    :param system: System
    :param new_name: Molecule new name
    :return: Molecule old name, .ff.itp, .itp file, .gro file paths
    """
    workdir.mkdir(parents=True, exist_ok=True)
    top_old = system.topology_map[system.molecules[0]]
    top = top_old.rename(new_name)
    s_files = system.save(workdir)

    ff = workdir / f"{top.name}.ff.itp"
    utils.backup(ff)
    ff.write_text("\n".join(system.forcefield.dump()))

    itp = workdir / f"{top.name}.itp"
    utils.backup(itp)
    itp.write_text("\n".join(top.dump()))
    pdb = converters.gro2pdb(gro=s_files.gro, workdir=workdir)
    return (top_old.name, ff, itp, pdb)
