"""Trajectory manipulation functions module."""

from __future__ import annotations

import enum
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from simdel import _deps, chem, func
from simdel._wrappers import gmx


class PBCAlgorithm(enum.Enum):
    """Periodic boundary condition."""

    Molecule = "mol"
    """Put the center of mass of molecules in the box."""

    Residue = "res"
    """Put the center of mass of residues in the box."""

    Atom = "atom"
    """Put all the atoms in the box."""

    NoJump = "nojump"
    """Check if atoms jump across the box and then puts them back.
    Ensures a continuous trajectory but molecules may diffuse out of the box."""

    Cluster = "cluster"
    """Clusters all the atoms in the selected index such that
    they are all closest to the center of mass of the cluster."""

    Whole = "whole"
    """Only makes broken molecules whole."""


class BoxType(enum.Enum):
    """Unit cell representation for PBCType."""

    Rectangle = "rect"
    """Ordinary brick shape."""

    Triclinic = "tric"
    """Triclinic unit cell."""

    Compact = "compact"
    """Puts all atoms at the closest distance from the center of the box."""


class CenterLocation(enum.Enum):
    """Centering location."""

    DiagonalHalf = "rect"
    """Half of the box diagonal."""

    EdgeHalf = "tric"
    """Half of the sum of the box vectors."""

    Zero = "zero"
    """Zero."""


class FitType(enum.Enum):
    """Fitting type."""

    Trans = "translation"
    """XYZ translation fitting."""

    TransXY = "transxy"
    """XY plane translation fitting."""

    RotTrans = "rot+trans"
    """XYZ translation fitting + rotation fitting."""

    RotTransXY = "rotxy+transxy"
    """XY plane translation fitting + XY plane rotation."""

    Progressive = "progressive"
    """Progressive fitting."""


def extract_cords(
    system: chem.System,
    trajectory: chem.Trajectory,
    atom_idxs: list[int],
) -> np.ndarray:
    """Return a slice with atoms of interest from the trajectory array.

    :param system: Extract coordinates matrix
    :param trajectory: Trajectory to analyze
    :param atom_idxs: List with atom indexes in the trajectory
    :return: np.ndarray trj[N_frames, N_atoms, 3], where the last array dimension
    corresponds to particular axis (X, Y, Z)
    """
    traj = trajectory.get_mdtraj(system=system)
    xyz = traj.xyz
    if xyz is not None:
        return xyz[:, atom_idxs, :]
    msg = "Trajectory has not xyz"
    raise ValueError(msg)


# TODO: refactor
@_deps.require(gmx)
def split(  # noqa: PLR0913
    system: chem.System,
    trajectory: chem.Trajectory,
    workdir: Path,
    start_time: float | None = None,
    end_time: float | None = None,
    segments: int | None = None,
    every_n: int | None = None,
    name: str = "",
    compress: bool = True,
) -> list[chem.System]:
    """Split trajectory to separate geometry .gro files,
    take frames from .xtc or .trr files (find in this order) by gmx trjconv.

    :param system: Trajectory's system
    :param trajectory: Trajectory
    :param workdir: Workdir path
    :param start_time: Start time, in `ps`, defaults to None.
    :param end_time: End time, in `ps`, defaults to None.
    :param segments: Segments number, defaults to None
    :param every_n: Split frequency, in `steps`, defaults to None.
    :param name: Out name, defaults to trajectory name
    :param compress: Compress internal temporary files, defaults to True
    :return: Frame systems list
    """
    temp_dir = workdir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    system_dump = system.save(temp_dir)
    # trajectory = chem.copy(destination_dir=workdir)

    if segments == 1:
        msg = "Trajectory can be splitted to several segments not 1"
        raise ValueError(msg)

    index_file = func.dump_index(
        index_file=workdir / "system.ndx",
        indexes=dict(index=~system.geometry_view.name.isna()),
    )

    frames = gmx.trjconv(
        workdir=workdir,
        reference=system_dump.gro,
        trajectory=trajectory.file,
        index=index_file,
        start=start_time,
        end=end_time,
        separate=True,
        out_name=name or system.name,
    )
    if segments and not every_n:
        n = np.linspace(0, len(frames) - 1, segments)
        frames = [frames[int(i)] for i in n]
    elif every_n and not segments:
        n = range(0, len(frames) - 1, every_n)
        frames = [frames[i] for i in n]
    elif segments and every_n:
        msg = "Setting segments and every_n a both is not allowed"
        raise ValueError(msg)

    systems = [
        chem.System.load(
            top=system_dump.top,
            gro=frame,
        )
        .set_info(**dict(system.info))
        .set_indexes(**system.index)
        .rename(f"{system.name}_{i}")
        for i, frame in enumerate(frames)
    ]
    if compress:
        shutil.rmtree(temp_dir)
    return systems


# TODO: refactor
@_deps.require(gmx)
def extract_frame(  # noqa: PLR0913
    system: chem.System,
    trajectory: chem.Trajectory,
    workdir: Path,
    time: float,
    name: str = "",
    compress: bool = True,
) -> chem.System:
    """Extract 1 frame from trajectory by gmx trjconv.

    :param system: Trajectory's system
    :param trajectory: Trajectory
    :param workdir: Workdir path
    :param time: Time near frame (time <= frame time)
    :param name: Frame system name, defaults to ""
    :param compress: Compress internal temporary files, defaults to True
    :return: System frame
    """
    temp_dir = workdir / "temp"
    workdir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    system_dump = system.save(temp_dir)
    # trajectory = chem.copy(destination_dir=workdir)

    index_file = func.dump_index(
        index_file=workdir / "system.ndx",
        indexes=dict(index=~system.geometry_view.name.isna()),
    )
    dt_ = trajectory.dt
    frames = trajectory.frames

    t = (time // dt_ + (time % dt_ > 0)) * dt_
    t = round(t, 3)
    if t > round(dt_ * (frames - 1), 3):
        msg = "Time > trajectory length"
        raise ValueError(msg)

    frames = gmx.trjconv(
        workdir=workdir,
        reference=system_dump.gro,
        trajectory=trajectory.file,
        index=index_file,
        start=t,
        end=t,
        separate=True,
        out_name=name or system.name,
    )

    extracted_system = (
        chem.System.load(
            top=system_dump.top,
            gro=frames[0],
        )
        .set_info(**dict(system.info))
        .set_indexes(**system.index)
    )

    if compress:
        shutil.rmtree(temp_dir)
    return extracted_system


# def extract_subtraj(
#     system: chem.System,
#     trajectory: trajectory_.Trajectory,
#     index: pd.Series,
#     workdir: Path,
#     name: str = "",
#     compress: bool = True,
# ) -> tuple[chem.System, trajectory_.Trajectory]:
#     temp_dir = workdir / "temp"
#     workdir.mkdir(parents=True, exist_ok=True)
#     temp_dir.mkdir(parents=True, exist_ok=True)

#     name = name if name else f"{chem.name}_sub"

#     system_dump = system.save(temp_dir)
#     # trajectory = chem.copy(destination_dir=temp_dir)

#     traj, *_ = gromacs.trjconv(
#         workdir=workdir,
#         trajectory=chem.traj,
#         reference=system_dump.gro,
#         index=simulations.dump_index(index_file=workdir / "extract_index.ndx", indexes=dict(out=in
# dex)),
#         groups=["out"],
#         out_name=name,
#     )

#     if compress:
#         shutil.rmtree(temp_dir)
#     return (
#         chem.System.load(),
#         trajectory_.chem.load(system=system, traj=traj),
#     )


# TODO: refactor
def fix(  # noqa: PLR0913
    system: chem.System,
    trajectory: chem.Trajectory,
    workdir: Path,
    pbc_algorithm: PBCAlgorithm | None = None,
    box_type: BoxType | None = None,
    box_size: tuple[float, float, float] | None = None,
    cluster_target: pd.Series | None = None,
    center_target: pd.Series | None = None,
    center_location: CenterLocation | None = None,
    name: str = "",
    compress: bool = True,
) -> chem.Trajectory:
    """Change periodic boundary conditions, unit cell sizes and center system by gmx trjconv.

    :param system: Trajectory's system
    :param trajectory: Trajectory
    :param workdir: Workdir path
    :param pbc_algorithm: Periodic boundary conditions calculation algorithm, defaults to None
    :param box_type: Unit cell representation for PBCType, defaults to None
    :param box_size: Unit cell type, defaults to None
    :param cluster_target: Clustering target index, defaults to None
    :param center_target: Center target index, need to set center_location, defaults to None
    :param center_location: Centering location, defaults to None
    :param name: New trajectory name, defaults to ""
    :param compress: Compress internal temporary files, defaults to True
    :return: Fixed trajectory
    """
    temp_dir = workdir / "temp"
    workdir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    name = name or f"{trajectory.name}_boxed"

    if (
        pbc_algorithm in [PBCAlgorithm.Whole, PBCAlgorithm.Cluster, PBCAlgorithm.NoJump]
        and box_type
    ):
        msg = f"Can't set box type when boundary type = {pbc_algorithm.value}"
        raise ValueError(msg)

    if box_type == BoxType.Compact and pbc_algorithm:
        msg = "Compact box type is not compatible with any pbc"
        raise ValueError(msg)

    if box_size and not box_type:
        msg = "Box size is set, but box type is not set"
        raise ValueError(msg)

    if pbc_algorithm == PBCAlgorithm.Cluster and cluster_target is None:
        msg = "Set cluster index"
        raise ValueError(msg)

    if (cluster_target is not None) and (pbc_algorithm != PBCAlgorithm.Cluster):
        msg = (
            f"Cluster index is set, but pbc_algorithm = "
            f"{pbc_algorithm.value if pbc_algorithm else None}."
        )
        raise ValueError(msg)

    system_dump = system.save(temp_dir)
    # trajectory = chem.copy(destination_dir=temp_dir)

    v = system.geometry_view
    index = dict(out=~v.name.isna())
    groups = ["out"]

    if center_target is not None:
        x: dict[str, pd.Series[bool]] = dict(center=center_target)
        index |= x
        groups.insert(0, "center")

    if pbc_algorithm == PBCAlgorithm.Cluster and cluster_target is not None:
        x: dict[str, pd.Series[bool]] = dict(cluster=cluster_target)
        index |= x
        groups.insert(0, "cluster")

    mdp = temp_dir / "mdp.mdp"
    mdp.write_text("")
    grompp_out = gmx.grompp(
        geometry=system_dump.gro,
        posres_geometry=system_dump.gro,
        top=system_dump.top,
        workdir=temp_dir,
        mdp=mdp,
        out_name="proxy",
    )

    traj, *_ = gmx.trjconv(
        workdir=workdir,
        trajectory=trajectory.file,
        reference=grompp_out.tpr,
        pbc_type=pbc_algorithm.value if pbc_algorithm else None,
        unit_representation=box_type.value if box_type else None,
        box=box_size or None,
        center_atoms=center_target is not None,
        center_box=center_location.value if center_location else None,
        index=func.dump_index(index_file=workdir / "fix_index.ndx", indexes=index),
        groups=groups,
        out_name=name,
    )

    if compress:
        shutil.rmtree(temp_dir)
    return chem.Trajectory(
        file=traj,
        dt=trajectory.dt,
        frames=trajectory.frames,
    )


# TODO: refactor
def fit(  # noqa: PLR0913
    system: chem.System,
    trajectory: chem.Trajectory,
    workdir: Path,
    fit_target: pd.Series,
    fit_type: FitType,
    center_target: pd.Series | None = None,
    center_location: CenterLocation | None = None,
    name: str = "",
    compress: bool = True,
) -> chem.Trajectory:
    """Fit trajectory to the target by gmx trjconv.

    :param system: Target system,
    :param trajectory: Trajectory
    :param workdir: Workdir path
    :param fit_target: Fit target index
    :param fit_type: Fit type
    :param center_target: Center target index, need to set center_location, defaults to None
    :param center_location: Centering location, defaults to None
    :param name: Fitted trajectory name, defaults to ""
    :param compress: Compress internal temporary files, defaults to True
    :return: Fitted trajectory
    """
    temp_dir = workdir / "temp"
    workdir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    name = name or f"{trajectory.name}_fitted"

    system_dump = system.save(temp_dir)
    # trajectory = chem.copy(destination_dir=temp_dir)

    index = dict(
        out=~system.geometry_view.name.isna(),
        fit=fit_target,
    )
    groups = ["fit", "out"]

    if center_target is not None:
        x: dict[str, pd.Series[bool]] = dict(center=center_target)
        index |= x
        groups.insert(1, "center")

    traj, *_ = gmx.trjconv(
        workdir=workdir,
        trajectory=trajectory.file,
        reference=system_dump.gro,
        fit_type=fit_type.value,
        center_atoms=center_target is not None,
        center_box=center_location.value if center_location else None,
        index=func.dump_index(index_file=workdir / "fit_index.ndx", indexes=index),
        groups=groups,
        out_name=name,
    )

    if compress:
        shutil.rmtree(temp_dir)
    return chem.Trajectory(
        file=traj,
        dt=trajectory.dt,
        frames=trajectory.frames,
    )


# TODO: can use shift and trans a both?
# TODO: refactor
def shift_on_vector(  # noqa: PLR0913
    system: chem.System,
    trajectory: chem.Trajectory,
    workdir: Path,
    shift: tuple[float, float, float] = (0, 0, 0),
    trans: bool = True,
    center_target: pd.Series | None = None,
    center_location: CenterLocation | None = None,
    name: str = "",
    compress: bool = True,
) -> chem.Trajectory:
    """Shift trajectory on vector by gmx trjconv.

    :param system: Target system,
    :param trajectory: Trajectory
    :param workdir: Workdir path
    :param shift: Shift vector, defaults to (0, 0, 0)
    :param trans: Use `-trans` (True) or `-shift` option, defaults to True
    :param center_target: Center target index, need to set center_location, defaults to None
    :param center_location: Centering location, defaults to None
    :param name: Shifted trajectory name, defaults to ""
    :param compress: Compress internal temporary files, defaults to True
    :return: Shifted trajectory
    """
    temp_dir = workdir / "temp"
    workdir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    name = name or f"{trajectory.name}_shifted"

    chemdump = system.save(temp_dir)
    trajectory = trajectory.copy(destination_dir=temp_dir)

    index = dict(
        out=~system.geometry_view.name.isna(),
    )
    groups = ["out"]

    if center_target is not None:
        x: dict[str, pd.Series[bool]] = dict(center=center_target)
        index |= x
        groups = ["center"] + groups

    traj, *_ = gmx.trjconv(
        workdir=workdir,
        trajectory=trajectory.file,
        reference=chemdump.gro,
        index=func.dump_index(index_file=workdir / "shift_index.ndx", indexes=index),
        groups=groups,
        out_name=name,
        trans=shift if trans else None,
        shift=shift if not trans else None,
        center_atoms=center_target is not None,
        center_box=center_location.value if center_location else None,
    )

    if compress:
        shutil.rmtree(temp_dir)
    return chem.Trajectory(
        file=traj,
        dt=trajectory.dt,
        frames=trajectory.frames,
    )
