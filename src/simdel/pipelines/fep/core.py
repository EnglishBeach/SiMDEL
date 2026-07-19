"""FEP simulation functions."""

# ruff: disable[D101, D103]
from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
from pydantic import BaseModel

from simdel import _deps, _utils, analyse, chem, func, traj
from simdel._wrappers import gmx, openff, pmx
from simdel.chem import DefaultFF, DefaultIon

from . import configs

mark_pipeline = _deps.require(pmx, gmx, openff)


class StateName:
    """Hybrid system name."""

    la = "ligand_A"
    lb = "ligand_B"
    ca = "complex_A"
    cb = "complex_B"


class PipelineResult(_utils.Table):
    ligand_a: str
    """Ligand A name"""

    ligand_b: str
    """Ligand A name"""

    ddG: pd.Series[float]
    """dG_complex - dG_ligand, equal dG(P + ligB) - dG(P + ligA) - difference
    in ligand activity in `kJ/mol`"""

    ddG_error: pd.Series[float]
    """Analytical error of ddG by s sum rule in `kJ^2/mol^2`"""

    ddG_berror: pd.Series[float]
    """Bootstrap error of ddG by find s of all samplings together in `kJ^2/mol^2`"""


def is_forward(system: chem.System) -> bool:
    """Get forward system or not: ca, la - forward, cb, lb - not forward.

    :param name: System name
    :return: Bool
    """
    return system.name in [StateName.la, StateName.ca]


# TODO: compress to context
def parametrize_protein(
    workdir: Path,
    raw_prot_path: Path,
    ff_type: chem.GromacsFF,
    compress: bool,
) -> chem.System:
    workdir.mkdir(exist_ok=True, parents=True)
    temp_dir = workdir / "temp"
    temp_dir.mkdir(exist_ok=True, parents=True)

    protein_copy = temp_dir / raw_prot_path.name
    shutil.copy(src=raw_prot_path, dst=protein_copy)

    protein = func.parametrize_protein(
        name=raw_prot_path.stem,
        geometry=raw_prot_path,
        workdir=temp_dir,
        ff=ff_type,
    )
    if compress:
        shutil.rmtree(temp_dir)

    protein.save(save_dir=workdir)
    return protein


def create_extras_map(
    workdir: Path,
    extra_sdf: list[Path],
) -> dict[str, Path]:
    workdir.mkdir(parents=True, exist_ok=True)

    sdf_map = {f"e{i:>02}": sdf for i, sdf in enumerate(extra_sdf)}
    (workdir / "names.json").write_text(json.dumps(sdf_map))
    return sdf_map


def parametrize_extra(
    workdir: Path,
    sdf: Path,
    name: str,
    fast: bool,
    compress: bool,
) -> chem.System:
    workdir.mkdir(parents=True, exist_ok=True)
    temp_dir = workdir / "temp"
    system = func.parametrize_small(
        sdf=sdf,
        ff=DefaultFF.openff210,
        name=name,
        workdir=temp_dir,
        fast=fast,
    )
    system.save(save_dir=workdir)

    if compress:
        shutil.rmtree(temp_dir)
        _utils.clear_backups(workdir)
    return system


def split_ligand_sdf(
    workdir: Path,
    raw_ligs_sdf: Path,
    pairs: list[tuple[str, str]],
) -> tuple[
    dict[str, Path],
    dict[str, str],
    list[tuple[str, str]],
]:
    workdir.mkdir(parents=True, exist_ok=True)

    ligand_map = func.split_sdf(sdf=raw_ligs_sdf, workdir=workdir / "sdf")

    sdf_map: dict[str, Path] = {}
    name_map: dict[str, str] = {}
    for i, (name, sdf) in enumerate(ligand_map.items()):
        key = f"l{i:>02}"
        sdf_map[key] = sdf
        name_map[key] = name

    (workdir / "keys.json").write_text(json.dumps(name_map))

    if pairs:
        name_map_ = {v: k for k, v in name_map.items()}
        pair_keys = [(name_map_[raw_a], name_map_[raw_b]) for raw_a, raw_b in pairs]
    else:
        pairs_sdf = func.gen_alchemy_graph(
            workdir=workdir / "graph", sdf_list=list(sdf_map.values())
        )
        sdf_map_ = {v: k for k, v in sdf_map.items()}
        pair_keys = [(sdf_map_[sdf_a], sdf_map_[sdf_b]) for sdf_a, sdf_b in pairs_sdf]

    (workdir / "pairs.json").write_text(json.dumps(pair_keys))

    used_keys = {i[0] for i in pair_keys} | {i[1] for i in pair_keys}
    sdf_map = {k: v for k, v in sdf_map.items() if k in used_keys}
    name_map = {k: v for k, v in name_map.items() if k in used_keys}
    return sdf_map, name_map, pair_keys


def parametrize_ligand(
    workdir: Path,
    sdf: Path,
    name: str,
    fast: bool,
    compress: bool,
) -> chem.System:
    workdir.mkdir(parents=True, exist_ok=True)
    temp_dir = workdir / "temp"
    system = func.parametrize_small(
        sdf=sdf,
        ff=DefaultFF.openff210,
        name=name,
        workdir=temp_dir,
        fast=fast,
    )
    system.save(save_dir=workdir)

    if compress:
        shutil.rmtree(temp_dir)
    return system


def create_hybrids(
    workdir: Path,
    system_a: chem.System,
    system_b: chem.System,
    compress: bool,
) -> tuple[chem.System, chem.System]:
    temp_dir = workdir / "temp"
    hybridA, hybridB = func.create_hybrids(
        system_a=system_a,
        system_b=system_b,
        workdir=temp_dir,
    )

    if compress:
        shutil.rmtree(temp_dir)

    hybridA.save(workdir / "A")
    hybridB.save(workdir / "B")
    return hybridA, hybridB


# TODO: refactor
def create_box(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    water_type: chem.WaterType,
    concentration: float,
    emtol: float,
    compress: bool,
) -> chem.System:
    temp_box_dir = workdir / "temp_box"
    box = func.create_box(
        system=system,
        box_type=chem.BoxType.Dodecahedron,
        box_distance=1.5,
        workdir=temp_box_dir,
    )
    # TODO: custom fep field
    temp_solvated_dir = workdir / "temp_solvated"
    ff_type = (
        box.info.ff_type
        if isinstance(box.info.ff_type, chem.GromacsFF)
        else DefaultFF.amber99sb_ildn
    )
    solvated_box = func.solvate(
        system=box,
        ff=ff_type,
        water_type=water_type,
        flexible_water=True,
        workdir=temp_solvated_dir,
    )

    temp_ionic_dir = workdir / "temp_ionic"
    ions_config = configs.Ions(forward=is_forward(system), emtol=emtol)
    ionic_box = func.add_ions(
        system=solvated_box,
        parameters=ions_config,
        concentration=concentration,
        positive_ion=DefaultIon.Na,
        negative_ion=DefaultIon.Cl,
        workdir=temp_ionic_dir,
    )
    temp_index_dir = workdir / "temp_index"
    indexed_box = ionic_box.set_indexes(
        **func.create_gromacs_indexes(system=ionic_box, workdir=temp_index_dir)
    )

    if compress:
        shutil.rmtree(temp_box_dir)
        shutil.rmtree(temp_solvated_dir)
        shutil.rmtree(temp_ionic_dir)

    indexed_box.save(workdir)
    return indexed_box


# TODO: refactor
def em(  # noqa: PLR0913
    workdir: Path,
    box: chem.System,
    emtol: float,
    n_mpi: int | None,
    n_omp: int | None,
    compress: bool,
) -> chem.System:
    mdp = configs.EM(forward=is_forward(box), emtol=emtol)
    system, trajectory, energy, _ = func.simulate(
        workdir=workdir,
        system=box,
        parameters=mdp,
        compress=compress,
        n_mpi=n_mpi,
        n_omp=n_omp,
    )
    if compress:
        if trajectory:
            trajectory.remove_file()
        if energy:
            energy.edr.unlink() if energy.edr else None
        (workdir / f"{system.name}.top").unlink()
        (workdir / f"{system.name}.ndx").unlink()
    return system


def set_rigid_water(
    workdir: Path,
    system: chem.System,
    compress: bool,
) -> chem.System:
    if not system.info.water_type:
        msg = f"System must be solvated: {system.name}"
        raise ValueError(msg)

    temp_dir = workdir / "temp"
    new_system = func.resolvate(
        system=system,
        # TODO: check
        flexible_water=False,
        water_type=system.info.water_type,
        workdir=temp_dir,
    )
    new_system.save(workdir)
    if compress:
        shutil.rmtree(temp_dir)
    return new_system


# TODO: refactor
def eq(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    time: float,
    T: float,
    n_frames: int,
    internal_steps: int,
    n_mpi: int | None,
    n_omp: int | None,
    compress: bool,
) -> tuple[chem.System, chem.Trajectory]:
    mdp = configs.EQ(
        forward=is_forward(system),
        time=time * 1000,
        T=T,
        ti_frames=n_frames,
        energy_checks=internal_steps,
    )
    simulation_system, trajectory, energy, _ = func.simulate(
        system=system,
        parameters=mdp,
        n_mpi=n_mpi,
        n_omp=n_omp,
        workdir=workdir,
        compress=compress,
    )
    if not trajectory:
        msg = "No trajectory after em stage"
        raise ValueError(msg)
    if compress:
        if energy and energy.edr:
            energy.edr.unlink()
        (workdir / f"{system.name}.top").unlink()
        (workdir / f"{system.name}.ndx").unlink()
    return simulation_system, trajectory


# TODO: refactor
def split(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    trajectory: chem.Trajectory,
    discard_time: float,
    end_time: float,
    n_frames: int,
    compress: bool,
) -> list[chem.System]:
    systems = traj.split(
        workdir=workdir,
        system=system,
        trajectory=trajectory,
        start_time=discard_time * 1000,
        end_time=end_time * 1000,
        segments=n_frames,
        compress=compress,
    )
    if compress:
        shutil.rmtree(workdir)
    return systems


class NeqResult(BaseModel):
    """NEQ stage result."""

    xvg: Path
    """Trajectory data .xvg file path."""
    edr: Path
    """Energy .edr file path."""


# TODO: refactor
def neq(  # noqa: PLR0913
    workdir: Path,
    frame_system: chem.System,
    time: float,
    T: float,
    n_mpi: int | None,
    n_omp: int | None,
    compress: bool,
) -> NeqResult:
    parameters = configs.NEQ(
        forward=is_forward(frame_system),
        time=time * 1000,
        T=T,
    )
    v = frame_system.topology_view
    clear_posres = chem.PositionRestraints.from_top_view(v)
    cleared_system = func.set_restraints(system=frame_system, restraints=clear_posres)

    system, trajectory, energy, _ = func.simulate(
        workdir=workdir,
        system=cleared_system,
        parameters=parameters,
        n_mpi=n_mpi,
        n_omp=n_omp,
        compress=compress,
    )
    if compress:
        if trajectory:
            trajectory.remove_file()

        (workdir / f"{system.name}.top").unlink()
        (workdir / f"{system.name}.ndx").unlink()

    if (not energy) or (not energy.xvg) or (not energy.edr):
        msg = "Trajectory data not found: .xvg, .edr files"
        raise RuntimeError(msg)

    return NeqResult(xvg=energy.xvg, edr=energy.edr)


class AnalyzeFEPOut(BaseModel):
    """Analyze thermodynamics transitions by BAR.

    - p - protein with extra molecules
    - La, Lb - ligands a/b states
    - pLa, pLb - complexes
    """

    ddG: float
    """Free energy site binding difference:
    ddG = dG(p + Lb -> pLb) - dG(c + La -> pLa)
    = dG_pL - dG_L."""

    dG_L: float
    """dG(La) - dG(Lb) in `kJ/mol`"""

    dG_pL: float
    """dG(pLa) - dG(pLb) in `kJ/mol`"""

    dG_L_error: float
    """dG(La) - dG(Lb) analytical error, s**2 = s1**2 + s2**2 in `kJ^2/mol^2`"""

    dG_pL_error: float
    """dG(pLa) - dG(pLb) analytical error, s**2 = s1**2 + s2**2 in `kJ^2/mol^2`"""

    dG_L_berror: float
    """dG(La) - dG(Lb) bootstrap error in `kJ^2/mol^2`"""

    dG_pL_berror: float
    """dG(pLa) - dG(pLb) bootstrap error in `kJ^2/mol^2`"""

    sampling_power: int
    """Points number"""


# TODO: refactor
def analyze_dG(  # noqa: PLR0913
    workdir: Path,
    T: int,
    samples: int,
    table_la: list[Path],
    table_lb: list[Path],
    table_ca: list[Path],
    table_cb: list[Path],
) -> AnalyzeFEPOut:
    ligand_result = analyse.analyze_dG(
        xvgs_a=table_la,
        xvgs_b=table_lb,
        temperature=T,
        samples=samples,
        workdir=workdir / "ligands",
    )
    complex_result = analyse.analyze_dG(
        xvgs_a=table_ca,
        xvgs_b=table_cb,
        temperature=T,
        samples=samples,
        workdir=workdir / "complex",
    )

    result = AnalyzeFEPOut(
        ddG=complex_result.dG - ligand_result.dG,
        dG_L=ligand_result.dG,
        dG_pL=complex_result.dG,
        dG_L_error=ligand_result.analytical_error,
        dG_pL_error=complex_result.analytical_error,
        dG_L_berror=ligand_result.bootstrap_error,
        dG_pL_berror=complex_result.bootstrap_error,
        sampling_power=samples,
    )
    with (workdir / "result_dG.json").open("w") as file:
        file.write(result.model_dump_json())
    return result
