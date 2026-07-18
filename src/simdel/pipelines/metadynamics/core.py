"""Metadynamics core functions and stages."""

# ruff: disable[D101, D103]
from __future__ import annotations

import json
from pathlib import Path
import shutil
import time as time_

from matplotlib import axes as plt_axes
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import interpolate

from simdel import _log, _utils, analyse, chem, func, traj
from simdel.pipelines import selections

from . import configs


class SiteResidue(BaseModel):
    chain: str
    residue_id: int
    residue_name: str


class PipelineResult(_utils.Table):
    dG: pd.Series[float]
    dG_error: pd.Series[float]


class MetadynamicsOut(BaseModel):
    cv: Path
    """CV .dat file path"""

    hills: Path
    """Hills .dat file path."""

    funnel: Path
    """Funnel potential .dat file path."""


class AnalyzeMetaOut(BaseModel):
    dG: float
    """Mean dG, in `kJ/mol`."""

    dG_error: float
    """Block standard error of the mean (BSE), in `kJ/mol`."""

    error_stable: bool
    """SEM stability."""

    dG_std: float
    """Standard error of dG. Not good for time series, in `kJ/mol`."""

    transitions_count: int
    """Transitions bind <-> unbind states."""

    transitions_frequency: float
    """Transitions bind <-> unbind states frequency, in `1/ps`."""

    convergence_rate: float
    """Stable mean dG plateau proportion."""

    v_convergence: float
    """Velocity of reach the plateau, in `1/ps`."""


def parametrize_protein(
    workdir: Path,
    raw_prot_path: Path,
    ff_type: chem.GromacsFF,
    water_type: chem.WaterType,
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
        water_type=water_type,
        fix_missing=True,
    ).rename("protein")

    if compress:
        shutil.rmtree(temp_dir)
        _utils.clear_backups(workdir)

    protein.save(save_dir=workdir)
    return protein


def split_ligand_sdf(
    workdir: Path,
    raw_ligs_sdf: Path,
) -> tuple[
    dict[str, Path],
    dict[str, str],
]:
    workdir.mkdir(parents=True, exist_ok=True)

    ligand_map = func.split_sdf(sdf=raw_ligs_sdf, workdir=workdir / "sdf")

    sdf_map = {}
    name_map = {}
    for i, (name, sdf) in enumerate(ligand_map.items()):
        key = f"l{i:>02}"
        sdf_map[key] = sdf
        name_map[key] = name

    (workdir / "names.json").write_text(json.dumps(name_map))
    return sdf_map, name_map


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
        ff=chem.DefaultFF.openff210,
        name=name,
        workdir=temp_dir,
        fast=fast,
    )
    system.save(save_dir=workdir)

    if compress:
        shutil.rmtree(temp_dir)
        _utils.clear_backups(workdir)
    return system


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
        ff=chem.DefaultFF.openff210,
        name=name,
        workdir=temp_dir,
        fast=fast,
    )
    system.save(save_dir=workdir)

    if compress:
        shutil.rmtree(temp_dir)
        _utils.clear_backups(workdir)
    return system


# TODO: refactor
def create_box(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    water_type: chem.WaterType,
    concentration: float,
    emtol: float,
    compress: bool,
    box_distance: float = 2,
) -> chem.System:
    workdir.mkdir(parents=True, exist_ok=True)

    temp_box = workdir / "temp_box"
    box = func.create_box(
        workdir=temp_box,
        system=system,
        box_type=chem.BoxType.Cubic,
        box_distance=box_distance,
        center=True,
    )

    # TODO: custom field
    temp_solvated = workdir / "temp_solvated"
    ff_type = (
        box.info.ff_type
        if isinstance(box.info.ff_type, chem.GromacsFF)
        else chem.DefaultFF.amber99sb_ildn
    )

    solvated_box = func.solvate(
        workdir=temp_solvated,
        system=box,
        ff=ff_type,
        water_type=water_type,
        flexible_water=True,
    )

    temp_ionic = workdir / "temp_ionic"
    ions_config = configs.Ions(emtol=emtol)
    ionic_box = func.add_ions(
        system=solvated_box,
        parameters=ions_config,
        concentration=concentration,
        positive_ion=chem.DefaultIon.Na,
        negative_ion=chem.DefaultIon.Cl,
        workdir=temp_ionic,
    )

    temp_index = workdir / "temp_index"
    indexed_box = ionic_box.set_indexes(
        **func.create_gromacs_indexes(system=ionic_box, workdir=temp_index)
    )

    indexed_box.save(workdir)
    if compress:
        shutil.rmtree(temp_box)
        shutil.rmtree(temp_solvated)
        shutil.rmtree(temp_ionic)
        shutil.rmtree(temp_index)
        _utils.clear_backups(workdir)
    return indexed_box


def site_search(
    workdir: Path,
    system: chem.System,
    ligand_name: str,
    site_residues: list[SiteResidue] | None = None,
) -> tuple[pd.Series[bool], pd.Series[bool], pd.Series[bool]]:
    workdir.mkdir(parents=True, exist_ok=True)

    view = system.geometry_view
    ligand_mask = view.molecule == ligand_name
    protein_mask = view.molecule.map(lambda x: "Protein" in x)  # type: ignore
    if site_residues:
        residues_list = []
        for site_res in site_residues:
            site_str = ":".join(str(i) for i in dict(site_res).values())
            residues_list.append(site_str)

        def f2(x: pd.Series) -> bool:
            return ":".join(str(i) for i in x) in residues_list

        site_mask = view.to_df()[["chain", "sequence", "residue"]].agg(f2, axis=1)
    else:
        site_mask = analyse.get_site_mask(
            view=view,
            ligand_mask=ligand_mask,
            protein_mask=protein_mask,
        )

    func.dump_index(
        index_file=workdir / "masks.ndx",
        indexes=dict(
            site=site_mask,
            ligand=ligand_mask,
            protein=protein_mask,
        ),
    )
    return site_mask, ligand_mask, protein_mask


# TODO: refactor
def minimize(  # noqa: PLR0913
    workdir: Path,
    box: chem.System,
    emtol: float,
    min_time: float,
    T: float,
    n_mpi: int | None,
    n_omp: int | None,
    compress: bool,
) -> chem.System:
    workdir.mkdir(parents=True, exist_ok=True)

    temp_posres = workdir / "temp_posres"
    view = box.topology_view
    posres = selections.set_constant_posres(
        view=view,
        selection=view.molecule.map(lambda x: "Protein" in x) * (view.mass > 1),  # type: ignore
        value=1000,
    )
    posres_system = func.set_restraints(system=box, restraints=posres)
    posres_system.save(temp_posres)

    temp_em = workdir / "temp_em"
    mdp = configs.EM(emtol=emtol)
    em_system, *_ = func.simulate(
        workdir=temp_em,
        system=posres_system,
        parameters=mdp,
        compress=compress,
        n_mpi=n_mpi,
        n_omp=n_omp,
    )

    temp_nvt = workdir / "temp_nvt"
    mdp = configs.NVT(time=min_time * 1000, T=T)
    nvt_system, *_ = func.simulate(
        workdir=temp_nvt,
        system=em_system,
        parameters=mdp,
        n_mpi=n_mpi,
        n_omp=n_omp,
        compress=compress,
    )

    temp_npt = workdir / "temp_npt"
    mdp = configs.NPT(time=min_time * 1000, T=T, frames=1, energy_checks=0)
    npt_system, *_ = func.simulate(
        workdir=temp_npt,
        system=nvt_system,
        parameters=mdp,
        n_mpi=n_mpi,
        n_omp=n_omp,
        compress=compress,
    )

    temp_clear_posres = workdir / "temp_clear_posres"
    view = npt_system.topology_view
    posres = chem.PositionRestraints.from_top_view(view)
    posres.clear()
    minimized_system = func.set_restraints(system=npt_system, restraints=posres)
    minimized_system.save(temp_clear_posres)

    if compress:
        shutil.rmtree(temp_posres)
        shutil.rmtree(temp_em)
        shutil.rmtree(temp_nvt)
        shutil.rmtree(temp_npt)
        shutil.rmtree(temp_clear_posres)
        _utils.clear_backups(workdir)
    return minimized_system


def set_rigid_water(
    workdir: Path,
    system: chem.System,
    compress: bool,
) -> chem.System:
    temp_rigid = workdir / "temp_rigid"
    if not system.info.water_type:
        msg = f"System must be solvated: {system.name}"
        raise ValueError(msg)
    resolved_system = func.resolvate(
        workdir=temp_rigid,
        system=system,
        flexible_water=False,
        water_type=system.info.water_type,
    )
    resolved_system.save(workdir)

    if compress:
        shutil.rmtree(temp_rigid)
    return resolved_system


# TODO: refactor
def npt(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    time: float,
    T: float,
    frames: int,
    n_mpi: int | None,
    n_omp: int | None,
    compress: bool,
) -> tuple[chem.System, chem.Trajectory]:
    workdir.mkdir(parents=True, exist_ok=True)

    mdp = configs.NPT(time=time * 1000, T=T, frames=frames, energy_checks=1)
    simulation_system, trajectory, energy, _ = func.simulate(
        system=system,
        parameters=mdp,
        n_mpi=n_mpi,
        n_omp=n_omp,
        workdir=workdir,
        compress=compress,
    )
    if compress:
        if energy and energy.edr:
            energy.edr.unlink()
        (workdir / f"{system.name}.top").unlink()
        (workdir / f"{system.name}.ndx").unlink()
        _utils.clear_backups(workdir)

    if not trajectory:
        msg = "No trajectory after npt"
        raise ValueError(msg)
    return simulation_system, trajectory


# TODO: refactor
# TODO: clusterize
def split_trajectory(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    trajectory: chem.Trajectory,
    discard_time: float,
    end_time: float,
    n_frames: int,
    compress: bool,
) -> list[chem.System]:
    workdir.mkdir(parents=True, exist_ok=True)

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
        _utils.clear_backups(workdir)

    return [system.rename(f"frame_{i}") for i, system in enumerate(systems)]


def create_funnels(
    workdir: Path,
    system: chem.System,
    site_mask: pd.Series[bool],
    protein_mask: pd.Series[bool],
    ligand_mask: pd.Series[bool],
) -> list[chem.Funnel]:
    cone_alpha = np.pi / 6
    nonH_mass = 2
    C_atom_num = 6
    workdir.mkdir(parents=True, exist_ok=True)

    view = system.geometry_view

    lig = view[ligand_mask & (view.mass > nonH_mass)]
    lig_xyz = np.stack([lig["x"], lig["y"], lig["z"]], axis=1)
    lig_center = lig_xyz.mean(axis=0)

    site = view[site_mask]
    site_center = np.stack([site["x"], site["y"], site["z"]], axis=1).mean(axis=0)

    prot = view[protein_mask]
    prot_xyz = np.stack([prot["x"], prot["y"], prot["z"]], axis=1)
    w = np.array(prot.ff_C.map(lambda x: x[0]) / (2 ** (1 / 6)))

    traces = _get_best_directions(center=site_center, xyz=prot_xyz, w=w)

    c_skeleton = site[site.atomic_number == C_atom_num]
    anchor_id = c_skeleton.ai.iloc[0]

    funnels: list[chem.Funnel] = []
    for i, trace in enumerate(traces):
        cone_center, cone_h = _get_cone_params(
            alpha=cone_alpha,
            center=site_center,
            trace=trace,
            xyz=lig_xyz,
        )
        binding_h = float((lig_center - cone_center) @ trace)

        max_h = analyse.find_box_distance(
            system=system,
            center_point=tuple(cone_center),
            direction_point=tuple(cone_center + trace),
        )
        funnel = chem.Funnel.create(
            center=cone_center,
            b=cone_center + trace,
            anchor_id=anchor_id,
            cone_angle=cone_alpha,
            cone_h=cone_h,
            tube_r=0.1,
            total_h=round(max_h * 0.9, 2),
            site_h=binding_h,
        )

        (workdir / f"funnel_{i}.json").write_text(funnel.dump())
        funnels.append(funnel)
    return funnels


# TODO: refactor
# TODO: -> none
def metadynamics(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    funnel: chem.Funnel,
    site_mask: pd.Series[bool],
    ligand_mask: pd.Series[bool],
    protein_mask: pd.Series[bool],
    time: float,
    T: float,
    n_mpi: int | None,
    n_omp: int | None,
    compress: bool,
    sigma: float,
    height: float,
    pace: int,
) -> MetadynamicsOut | None:
    workdir.mkdir(parents=True, exist_ok=True)

    reference_file = _get_reference_file(workdir=workdir, system=system, protein_mask=protein_mask)

    # TODO: may be parametrize
    config = configs.META(
        time=time * 1000,
        T=T,
        frames=100,
        energy_checks=1,
    )

    config_plumed = configs.Funnel(
        system=system,
        reference_file=reference_file,
        funnel=funnel,
        T=T,
        ligand_mask=ligand_mask,
        site_mask=site_mask,
        sigma=sigma,
        height=height,
        pace=pace,
    )
    for _ in range(2):
        try:
            _, _, energy, plumed_data = func.simulate(
                system=system,
                parameters=config,
                plumed_parameters=config_plumed,
                n_mpi=n_mpi,
                n_omp=n_omp,
                workdir=workdir,
                compress=compress,
            )
            if compress:
                if energy and energy.edr:
                    energy.edr.unlink()
                (workdir / f"{system.name}.top").unlink()
                (workdir / f"{system.name}.ndx").unlink()
                _utils.clear_backups(workdir)

            return MetadynamicsOut(
                cv=plumed_data["colvar"],
                hills=plumed_data["hills"],
                funnel=plumed_data["funnel"],
            )
        except Exception:  # noqa: PERF203
            time_.sleep(1)
    msg = "All trials are failed"
    _log.warning(msg)
    return None


# TODO: refactor
def analyze_meta(  # noqa: PLR0913
    workdir: Path,
    hills_file: Path,
    cv_file: Path,
    funnel_file: Path,
    T: float,
    funnel: chem.Funnel,
    fast: bool = False,
) -> AnalyzeMetaOut:

    max_stride = 500
    rtol = 0.3
    min_points = 2
    workdir.mkdir(parents=True, exist_ok=True)

    cv_df = func.read_plumed(cv_file)
    hills_df = func.read_plumed(hills_file)
    funnel_df = func.read_plumed(funnel_file)

    stride = 500 if len(hills_df) > max_stride else 1

    integrals = analyse.integrate_hills(
        workdir=workdir / "fes",
        hills=hills_file,
        T=T,
        dt=0.002 * stride,  # integration_step (=0.002 ps) * write_cv_step (=500 step)
        stride=stride,
    )

    # Find bind/unbind cv ranges
    t = np.array([fes.time for fes in integrals])
    if fast:
        error = 0.3
        unbind_h = cv_df["fps.lp"].max()
        b0, b1 = (funnel.site_h - error, funnel.site_h + error)
        shift = 0.1
        u0, u1 = (unbind_h - shift - error, unbind_h - shift + error)
    else:
        error = 0.3
        b0, b1 = (funnel.site_h - error, funnel.site_h + error)
        # TODO: link with create funnel function: var to mark unbound region (all wall regions)
        shift = 0.5
        u0, u1 = (funnel.h - shift - error, funnel.h - shift + error)
    dG = np.array(
        [
            analyse.calculate_funnel_dG(
                fes=fes,
                tube_r=funnel.tube_r,
                T=T,
                bound={"fps.lp": (b0, b1)},
                unbound={"fps.lp": (u0, u1)},
            )
            for fes in integrals
        ]
    )

    t_plateau, dG_mean, dG_error, error_relative = analyse.analyse_time_series(
        t=t,
        y=dG,
        tail_range=10_000,
        abs_error=10,
    )
    convergent_mark = error_relative < rtol

    # Find plateau standard error (incorrect), if not convergence - at all time line
    plateau_mask = t >= t_plateau if sum(t >= t_plateau) > min_points else t > 0
    dG_std = dG[plateau_mask].std()

    transitions_count = analyse.calculate_hysteresis_transitions(
        (cv_df["fps.lp"] - funnel.cone_h).to_numpy()
    )
    transitions_frequency = transitions_count / t[-1]

    # The fraction of a convergent trajectory
    convergence_rate = 1 - t_plateau / t[-1]

    result = AnalyzeMetaOut(
        dG=dG_mean,
        dG_error=dG_error,
        error_stable=convergent_mark,
        dG_std=dG_std,
        transitions_count=transitions_count,
        transitions_frequency=transitions_frequency,
        v_convergence=1 / t_plateau if convergent_mark else 0,
        convergence_rate=convergence_rate if convergent_mark else 0,
    )
    (workdir / "result.json").write_text(json.dumps(dict(result)))

    fes = integrals[-1]
    bind_mask = (b0 <= fes.df["fps.lp"]) & (fes.df["fps.lp"] <= b1)
    unbind_mask = (u0 <= fes.df["fps.lp"]) & (fes.df["fps.lp"] <= u1)

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(16, 9))
    conv_title = f"converges in {t_plateau}" if convergent_mark else "not converges"
    fig.suptitle(
        f"dG={dG_mean:.1f} ± {dG_error:.0f} $kJ/mol$: {conv_title}\n"
        f"Funnel transitions = {int(transitions_count)} ({transitions_frequency * 100:.1} in $ns$)"
    )
    (
        (ax_dG, ax_cv, ax_dGt),
        (ax_funnel, ax_bias, ax_hills),
    ) = axes

    _plot_free(
        ax=ax_dG,
        lp=fes.df["fps.lp"].to_numpy(),
        free=fes.df["free"].to_numpy(),
        bind_mask=bind_mask,
        unbind_mask=unbind_mask,
        dG_funnel=dG_mean,
    )
    _funnel_plot(
        ax=ax_funnel,
        funnel_lp=funnel_df["fps.lp"].to_numpy(),
        funnel_ld=funnel_df["fps.ld"].to_numpy(),
        funnel_bias=funnel_df["funnel.bias"].to_numpy(),
        cv_ld=cv_df["fps.ld"].to_numpy(),
        cv_lp=cv_df["fps.lp"].to_numpy(),
    )
    _cv_plot(
        ax=ax_cv,
        time=cv_df["time"].to_numpy(),
        ld=cv_df["fps.ld"].to_numpy(),
        lp=cv_df["fps.lp"].to_numpy(),
    )
    _bias_plot(
        ax=ax_bias,
        time=cv_df["time"].to_numpy() / 1000,
        funnel=cv_df["funnel.bias"].to_numpy(),
        lwall=cv_df["lwall.bias"].to_numpy(),
        uwall=cv_df["uwall.bias"].to_numpy(),
        metad=cv_df["metad.bias"].to_numpy(),
    )
    _dGt_plot(
        ax=ax_dGt,
        time=t / 1000,
        dG=dG,
        time_conv=t_plateau / 1000,
    )
    _hills_plot(
        ax=ax_hills,
        time=hills_df["time"].to_numpy() / 1000,
        hills=hills_df["height"].to_numpy(),
    )
    fig.tight_layout()
    plt.close()
    fig.savefig(workdir / "plots", dpi=300)
    return result


def _get_reference_file(
    workdir: Path,
    system: chem.System,
    protein_mask: pd.Series[bool],
) -> Path:
    C_atom_num = 6
    workdir.mkdir(exist_ok=True, parents=True)

    reference = func.extract_subsystem(system=system, index=protein_mask).rename("ref")
    system_files = reference.save(workdir)

    v = system.geometry_view
    df = v[v.atomic_number == C_atom_num].to_df()
    v2 = df.groupby("sequence", as_index=False).first()

    pdb = func.gro2pdb(gro=system_files.gro, workdir=workdir)

    text = pdb.read_text().split("\n")

    head = text[:4]
    atoms = list(v2.apply(_get_pdb_atoms, axis=1))
    end = [i for i in text[-3:] if "ATOM" not in i]

    _utils.backup(file=pdb)
    pdb.write_text("\n".join([*head, *atoms, *end]))
    return pdb


def _get_pdb_atoms(d: pd.Series) -> str:
    ai = d["ai"]
    atom_name = d["name"]
    icode = d["icode"] or " "
    residue = d["residue"]
    chain = d["chain"] or " "
    sequence = d["sequence"]
    x, y, z = d["x"] * 10, d["y"] * 10, d["z"] * 10
    xyz = f"{x:8.3f}{y:8.3f}{z:8.3f}"

    return (
        f"ATOM  {ai:>5} {atom_name:>4} {residue:>3} {chain:1}{sequence:4}{icode:1}   "
        f"{xyz}  1.00  0.00"
    )


# Funnel
def _create_fibonacci_traces(n_points: int) -> np.ndarray:
    """Generate n traces r=1 on sphere using fibonacci spiral.

    :param n_points: Traces number
    :return: Traces xyz coordinates
    """
    indices = np.arange(n_points, dtype=float)
    phi = np.arccos(1 - 2 * (indices + 0.5) / n_points)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.column_stack([x, y, z])


def _get_intersections(traces: np.ndarray, xyz: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Create intersections map around xyz points.
    Uses traces from center to points, calculate intersections with it,
    uses point radius. Only intersections, not tangent.

    :param traces: Traces to crate map
    :param xyz: Points xyz coordinates
    :param r: Any radius like scalar for each point
    :return: _description_
    """
    P = traces @ xyz.T
    X2 = (xyz**2).sum(axis=1)
    S2 = X2 - P**2

    in_spheres = (S2 - r**2) < 0
    in_direction = P > 0
    return np.sum(in_spheres * in_direction, axis=1)


def _to_spherical(
    xyz: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Convert Cartesian XYZ coordinates to spherical coordinates relative to a point:
    - r (radius): Distance from origin point
    - theta (θ): Polar angle (0 to π), angle from Z axis
    - phi (φ): Azimuthal angle (0 to 2π), angle in XY plane from X axis.

    :param xyz: Cartesian coordinates vector (or matrix):
        |x1 y1 z1|
        |x2 y2 z2|
    :param center: Spherical coordinates center, defaults to (0 0 0)
    :return: Spherical coordinates vector (or matrix):
    r1 theta1 phi1
    r2 theta2 phi2
    """
    shift = xyz - center

    r = np.linalg.norm(shift, axis=-1)

    cos_theta = np.divide(shift[:, 2], r, out=np.zeros_like(r), where=r != 0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    phi = np.arctan2(shift[:, 1], shift[:, 0])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.stack([r, theta, phi], axis=-1)


def _to_cartesian(
    spherical: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Convert spherical coordinates to Cartesian XYZ coordinates.

    :param spherical: Spherical coordinates vector (or matrix):
        |r1 theta1 phi1|
        |r2 theta2 phi2|
    :param center: Cartesian coordinates center, defaults to (0 0 0)
    :return: Cartesian coordinates xyz vector (or matrix)
    """
    r = spherical[:, 0]
    theta = spherical[:, 1]
    phi = spherical[:, 2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1) + center


def _get_best_directions(
    xyz: np.ndarray,
    w: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    best_peaks_rate = 0.85
    peaks_n = 500
    surface_traces = _create_fibonacci_traces(peaks_n)

    intersections = _get_intersections(traces=surface_traces, xyz=xyz - center, r=w)
    v = intersections.max() - intersections
    v = v / v.max()

    rtp = _to_spherical(xyz=surface_traces, center=np.array((0.0, 0.0, 0.0)))
    rtp[:, 0] = v

    t_per = np.pi
    p_per = 2 * np.pi

    r = rtp[:, 0]
    t = rtp[:, 1]
    p = rtp[:, 2]

    # t= 0-pi (polar)=> non-symmetric on boundary (anti-symmetric is correct here)
    r = np.concatenate([r, r, r])
    t = np.concatenate([t[::-1] - t_per, t, t[::-1] + t_per])
    p = np.concatenate([p, p, p])

    # p = 0-2pi (azimuthal)=> symmetric on boundary
    r = np.concatenate([r, r, r])
    t = np.concatenate([t, t, t])
    p = np.concatenate([p - p_per, p, p + p_per])

    rtp_padded = np.column_stack([r, t, p])

    inter = interpolate.LinearNDInterpolator(
        np.column_stack([rtp_padded[:, 1], rtp_padded[:, 2]]),
        rtp_padded[:, 0],
    )

    funnel_traces_xyz = _create_fibonacci_traces(40)
    rtp = _to_spherical(xyz=funnel_traces_xyz, center=np.array((0.0, 0.0, 0.0)))[:, 1:]
    r = inter(rtp)
    df = pd.DataFrame(
        dict(
            r=r,
            x=funnel_traces_xyz[:, 0],
            y=funnel_traces_xyz[:, 1],
            z=funnel_traces_xyz[:, 2],
        )
    )
    peaks = df.sort_values("r", ascending=False)
    peaks = peaks[peaks["r"] > best_peaks_rate]
    return peaks[["x", "y", "z"]].to_numpy()


def _get_cone_params(
    xyz: np.ndarray,
    center: np.ndarray,
    trace: np.ndarray,
    alpha: float = np.pi / 2,
) -> tuple[np.ndarray, float]:
    xyz = xyz - center
    proj = xyz @ trace

    hs = (np.sum(xyz**2, axis=1) - proj**2) ** (1 / 2)
    max_hs = (hs / np.tan(alpha)) + proj

    back_shift = abs(proj[proj < 0].min()) + 0.1 if len(proj[proj < 0]) else 0
    new_center = center - back_shift * trace

    max_h = round(max_hs.max() + 0.1 + back_shift, 2)
    return new_center, max_h


# Plots
# TODO: refactor
def _plot_free(  # noqa: PLR0913
    ax: plt_axes.Axes,
    lp: np.ndarray,
    free: np.ndarray,
    bind_mask: pd.Series[bool],
    unbind_mask: pd.Series[bool],
    dG_funnel: float,
):
    free = free - free.min()

    ax.plot(lp, free)

    max_y = max(free[bind_mask].max(), free[unbind_mask].max())

    # Bind
    x = lp[bind_mask]
    y = free[bind_mask]
    x0_b, xm_b, x1_b = x.min(), x.mean(), x.max()
    y0_b, ym_b, y1_b = y.min(), y.mean(), y.max()
    rect = matplotlib.patches.Rectangle(
        (x0_b, y0_b),
        x1_b - x0_b,
        y1_b - y0_b,
        fill=True,
        alpha=0.3,
        edgecolor="green",
        facecolor="green",
    )
    ax.add_patch(rect)
    ax.scatter(
        [xm_b],
        [ym_b],
        color="green",
        s=10,
        label=f"Среднее ({xm_b:.2f}, {ym_b:.2f})",
    )

    # Unbind
    x = lp[unbind_mask]
    y = free[unbind_mask]
    x0_u, xm_u, x1_u = x.min(), x.mean(), x.max()
    y0_u, ym_u, y1_u = y.min(), y.mean(), y.max()
    rect = matplotlib.patches.Rectangle(
        (x0_u, y0_u),
        x1_u - x0_u,
        y1_u - y0_u,
        fill=True,
        alpha=0.3,
        edgecolor="red",
        facecolor="red",
    )
    ax.add_patch(rect)
    ax.scatter([xm_u], [ym_u], color="red", s=10)

    ax.plot([xm_b, (xm_u + xm_b) / 2 * 1.1], [ym_b, ym_b], "k-", linewidth=1)
    ax.plot([xm_u, (xm_u + xm_b) / 2 * 0.9], [ym_u, ym_u], "k-", linewidth=1)
    ax.plot([(xm_u + xm_b) / 2, (xm_u + xm_b) / 2], [ym_b, ym_u], "k-", linewidth=1)
    ax.plot([xm_b, (xm_u + xm_b) / 2 * 1.1], [ym_b, ym_b], "k-", linewidth=1)
    ax.plot([xm_u, (xm_u + xm_b) / 2 * 0.9], [ym_u, ym_u], "k-", linewidth=1)
    ax.plot([(xm_u + xm_b) / 2, (xm_u + xm_b) / 2], [ym_b, ym_u], "k-", linewidth=1)
    ax.text(
        (xm_u + xm_b) / 2,
        (ym_b + ym_u) / 2,
        f"$ΔG = {round(dG_funnel, 2)} \\frac{{kJ}}{{mol}}$",
        ha="left",
        va="center",
        fontsize=12,
    )

    ax.set_title("Free")
    ax.set_ylim(0, max_y * 2)
    ax.set_xlabel("$Axis~[nm]$")
    ax.set_ylabel("$dG~[\\frac{{kJ}}{{mol}}]$")


# TODO: refactor
def _funnel_plot(  # noqa: PLR0913
    ax: plt_axes.Axes,
    funnel_lp: np.ndarray,
    funnel_ld: np.ndarray,
    funnel_bias: np.ndarray,
    cv_lp: np.ndarray,
    cv_ld: np.ndarray,
):
    funnel_bias = np.log10(funnel_bias + 1e-3)
    funnel_x, funnel_y = np.mgrid[
        funnel_lp.min() : funnel_lp.max() : 100j,
        funnel_ld.min() : funnel_ld.max() : 100j,
    ]
    funnel_grid = interpolate.griddata(
        points=(funnel_lp, funnel_ld),
        values=funnel_bias,
        xi=(funnel_x, funnel_y),
        method="linear",
    )

    ax.contourf(funnel_x, funnel_y, funnel_grid, 10, cmap="bwr", vmin=-3, vmax=3)
    ax.scatter(cv_lp, cv_ld, c="black", s=2, label="Trajectory")

    x0, y0 = cv_lp[0], cv_ld[0]
    dx = 0.1
    dy = dx / cv_lp.max() * cv_ld.max()
    rect = matplotlib.patches.Rectangle(
        (x0 - dx, y0 - dy),
        2 * dx,
        2 * dy,
        fill=True,
        alpha=0.5,
        edgecolor="white",
        facecolor="white",
        label="Start",
    )
    ax.add_patch(rect)

    ax.set_title("Funnel potential")
    ax.legend()
    ax.set_xlabel("$Axis~[nm]$")
    ax.set_ylabel("$R~[nm]$")


def _cv_plot(
    ax: plt_axes.Axes,
    time: np.ndarray,
    ld: np.ndarray,
    lp: np.ndarray,
):
    time = time / 1000

    ax.plot(time[::100], ld[::100], c="red", label="R")
    ax.plot(time[::100], lp[::100], c="blue", label="Axis")

    ax.set_title("CV")
    y_max = max(ld.max(), lp.max())
    ax.set_xlim(0, time.max())
    ax.set_ylim(0, y_max * 1.1)
    ax.set_xlabel("$Time~[ns]$")
    ax.set_ylabel("$Distance~[nm]$")
    ax.legend()

    # ax2 = ax.twinx()


# TODO: refactor
def _bias_plot(  # noqa: PLR0913
    ax: plt_axes.Axes,
    time: np.ndarray,
    funnel: np.ndarray,
    lwall: np.ndarray,
    uwall: np.ndarray,
    metad: np.ndarray,
):
    time = time / 1000

    x_b, y1_b, w = _window_bin(time, funnel, 200)
    ax.bar(x_b, y1_b, width=w, label="funnel", color="green", alpha=0.3)

    x_b, y2_b, w = _window_bin(time, lwall, 200)
    ax.bar(x_b, y2_b, width=w, label="low wall", color="red", alpha=0.3)

    x_b, y3_b, w = _window_bin(time, uwall, 200)
    ax.bar(x_b, y3_b, width=w, label="up wall", color="blue", alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(time, metad, label="meta", color="black")

    cv_vals = np.concatenate([funnel, lwall, uwall])
    cv_vals = cv_vals[cv_vals != 0]
    y2_max = float(np.percentile(cv_vals, 99.0) * 2) if len(cv_vals) else 1

    ax.set_title("Bias")
    ax.set_xlim(0, time.max())
    ax.set_xlabel("$Time~[ns]$")
    ax.set_ylim(0, y2_max)
    ax.set_ylabel("Bias")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")


def _dGt_plot(ax: plt_axes.Axes, time: np.ndarray, dG: np.ndarray, time_conv: float):
    ax.plot(time, dG, c="black", label="dG")
    ax.plot([time_conv, time_conv], [dG.min(), dG.max()], label="Convergence time")

    ax.legend()
    ax.set_title("dG(t)")
    ax.set_xlabel("$Time~[ns]$")
    ax.set_ylabel("$dG~[\\frac{{kJ}}{{mol}}]$")


def _hills_plot(
    ax: plt_axes.Axes,
    time: np.ndarray,
    hills: np.ndarray,
):
    ax.plot(time, hills, label="Hills")

    ax.set_title("Hills")
    ax.set_yscale("log")
    ax.set_xlabel("$Time~[ns]$")
    ax.set_ylabel("$Hill height$")
    ax.legend()


def _window_bin(x, y, n):
    bin_edges = np.linspace(x.min(), x.max(), n + 1)
    x_b = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_indices = np.digitize(x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n - 1)

    y_b = np.full(n, -np.inf)
    np.maximum.at(y_b, bin_indices, y)
    y_b = np.where(np.isneginf(y_b), 0, y_b)

    return x_b, y_b, x.max() / n
