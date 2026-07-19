"""GROMACS function wrappers."""

from pathlib import Path

from simdel import _utils

_utils.run("gromacs check", ["which gmx"])


class GromppOut(_utils.PathContainer):
    """Grompp output file path container."""

    tpr: Path
    """`-o` Binary .tpr file path"""

    top: Path
    """Processed or `-pp` merged topology .top file path"""

    full_mdp: Path
    """`-po` Full MD parameters .mdp file path"""


# TODO: add system charge value
def grompp(  # noqa: PLR0913
    workdir: Path,
    geometry: Path,
    top: Path,
    mdp: Path,
    out_name: str,
    posres_geometry: Path | None = None,
    index: Path | None = None,
    merge: bool = False,
    maxwarn: int = 0,
) -> GromppOut:
    """Preprocess topology, geometry. Uses gmx grompp
    Can create merged topology from all *.itp files included in .top file if `merge_name` set.

    :param workdir: Workdir path
    :param geometry: `-c` Geometry .pdb/.gro file path
    :param top: `-p` Topology .top file path
    :param mdp: `-f` Simulation config .mdp file path
    :param out_name: Name for output files
    :param posres_geometry: `-r` Geometry to apply restraints, defaults to None
    :param index: `-n` Index .ndx file path, defaults to None
    :param merge: `-pp` Merge topology to .top file, defaults to False
    :param maxwarn: `-maxwarn` Max warning during GROMACS simulation, defaults to 0
    :return: Path container
    """
    tpr = workdir / f"{out_name}.tpr"
    full_mdp = workdir / f"{out_name}_{mdp.name}"
    command = [
        "gmx grompp",
        "-nocopyright",
        f"-c '{geometry.resolve()}'",
        f"-p '{top.resolve()}'",
        f"-f '{mdp.resolve()}'",
        f"-o '{tpr.resolve()}'",
        f"-po '{full_mdp.resolve()}'",
    ]

    if maxwarn:
        command.append(f"-maxwarn {maxwarn}")

    if posres_geometry:
        command.append(f"-r '{posres_geometry.resolve()}'")

    if index:
        command.append(f"-n '{index.resolve()}'")

    if merge:
        out_top = workdir / f"merge_{top.name}"
        command.append(f"-pp '{out_top.resolve()}'")
    else:
        out_top = top

    _utils.run(
        command=command,
        title=f"preprocess {geometry.name}, {top.name}",
        workdir=workdir,
    )
    return GromppOut(tpr=tpr, full_mdp=full_mdp, top=out_top)


class Pdb2gmxOut(_utils.PathContainer):
    """Pdb2gmx protein output file paths container."""

    gro: Path
    """`-o` Geometry .gro file path."""

    top: Path
    """`-p` System topology .top file path."""

    pr: Path | None
    """`-i` Positional restraints .pr.itp file path."""

    itps: list[Path] = []
    """`-p` Molecules topology .itp file paths."""


# TODO: refactor
def pdb2gmx(  # noqa: PLR0913
    workdir: Path,
    ff_paths: list[Path],
    geometry: Path,
    out_name: str,
    ff_name: str,
    water_index: int,
    posres_value: float | None = None,
    reset_H: bool = False,
    fix_missing: bool = False,
    merge: bool = False,
) -> Pdb2gmxOut:
    """Parametrize protein and save .top .pdb files. Uses gmx pdb2gmx.

    :param workdir: Workdir path
    :param geometry: `-f` Path to protein .pdb/.gro file
    :param out_name: Name for output files
    :param ff_name: `-ff` Protein forcefield
    :param water_index: Water group number if system contains water
    :param posres_value: `-posrefc` Value positional restrictions to heavy atoms, defaults to None
    :param reset_H: `-ignh` Reset H atoms to fix their positions, defaults to False
    :param fix_missing: `-missing` Fix missing atoms and chain caps, defaults to False
    :param merge: `-merge all` Merge all molecules into 1 molecule, defaults to False
    :return: Path container
    """
    out_geometry = workdir / f"{out_name}.gro"
    out_top = workdir / f"{out_name}.top"
    out_posres = workdir / f"{out_name}.pr.itp"

    command_start = "${GMXLIB:+${GMXLIB}:}"
    path_strings = ":".join({i.resolve().parent.as_posix() for i in ff_paths})
    command = [
        f'export GMXLIB="{command_start}{path_strings}"',
        "&&",
        f"echo {water_index} | gmx pdb2gmx",
        "-nocopyright",
        f"-f '{geometry.resolve()}'",
        f"-ff {ff_name}",
        "-water select",
        f"-o '{out_geometry.resolve()}'",
        f"-p '{out_top.resolve()}'",
        f"-i '{out_posres.resolve()}'",
    ]
    if merge:
        command.append("-merge all")

    if posres_value:
        command.append(f"-posrefc {posres_value}")

    if reset_H:
        command.append("-ignh")

    if fix_missing:
        command.append("-missing")

    _utils.run(
        command=command,
        title=f"pdb2gmx {geometry.name}",
        workdir=workdir,
    )
    itps = [i for i in workdir.iterdir() if i.suffix == ".itp" and i != out_posres]
    return Pdb2gmxOut(gro=out_geometry, top=out_top, pr=out_posres, itps=itps)


# TODO: refactor
def editconf(  # noqa: PLR0913
    workdir: Path,
    geometry: Path,
    out_fname: str,
    box_type: str | None = None,
    box_distance: float | None = None,
    center: bool | None = None,
) -> Path:
    """Add box to geometry .pdb/.gro using by gmx editconf.

    :param workdir: Workdir path
    :param geometry: `-f` Geometry .pdb/.gro file path
    :param out_fname: Box .pdb file name
    :param box_type: `-bt` Box type, defaults to None
    :param box_distance: `-d` Gap between protein and wall, in `nm`, defaults to None
    :param center: `-c/-noc` Center command
    :return: `-o` Box geometry .gro file path
    """
    out_box = workdir / out_fname
    command = [
        "gmx editconf",
        "-nocopyright",
        f"-f '{geometry.resolve()}'",
        f"-o '{out_box.resolve()}'",
    ]

    if box_type:
        command.append(f"-bt {box_type}")

    if box_distance:
        command.append(f"-d {box_distance}")

    if center:
        command.append("-c")
    elif center is False:
        command.append("-noc")

    _utils.run(
        command=command,
        title=f"editconf {geometry.name}",
        workdir=workdir,
    )
    if not out_box.exists():
        msg = f"Out box does not exist: {out_box}"
        raise FileExistsError(msg)
    return out_box


class SolvateOut(_utils.PathContainer):
    """Solvate output file paths container."""

    gro: Path
    """`-o` Geometry .gro file path."""

    top: Path
    """`-p` Solvated topology (overwritten) .top file path."""


def solvate(
    workdir: Path,
    geometry: Path,
    top: Path,
    out_name: str,
    solvent_geometry: str,
) -> SolvateOut:
    """Solvate system with box using by gmx solvate.

    :param workdir: Workdir path
    :param geometry: `-cs` Geometry .pdb/.gro file path
    :param top: `-p` Topology .top file path, wil be backed up and overwritten
    :param out_name: Solvated system files name
    :param solvent_geometry: `-cs` Solvent geometry
    :return: Path container
    """
    gro = workdir / f"{out_name}.gro"
    command = [
        "gmx solvate",
        "-nocopyright",
        f"-cp '{geometry.resolve()}'",
        f"-p '{top.resolve()}'",
        f"-cs '{solvent_geometry}'",
        f"-o '{gro.resolve()}'",
    ]

    _utils.run(
        command=command,
        title=f"solvate {top.name}",
        workdir=workdir,
    )
    return SolvateOut(gro=gro, top=top.replace(workdir / f"{out_name}.top"))


class GenionOut(_utils.PathContainer):
    """Generate ions output file path container."""

    gro: Path
    """`-o` Box with ions geometry .gro file path."""

    top: Path
    """`-p` Box with ions topology .top file path."""


# TODO: refactor
def genion(  # noqa: PLR0913
    workdir: Path,
    tpr: Path,
    top: Path,
    sol_index: Path,
    positive_ion: str,
    positive_charge: int,
    negative_ion: str,
    negative_charge: int,
    concentration: float,
    neutral: bool,
    out_name: str,
) -> GenionOut:
    """Add ions in solvated box using by gmx genion.

    :param workdir: Workdir path
    :param tpr: `-s` Processed system binary file
    :param top: `-p` Topology .top file path
    :param sol_index: `-n` Index. ndx file path
    :param positive_ion: `-pname` Positive ion type
    :param positive_charge: `-pq` Positive ion charge
    :param negative_ion: `-nname` Negative ion type
    :param negative_charge: `-nq` Negative ion charge
    :param concentration: `-conc` Ions concentration in mol/l
    :param out_name: Solvated system files name
    :return: Path container
    """
    ionic_gro = workdir / f"{out_name}.gro"

    command = [
        "gmx genion",
        "-nocopyright",
        f"-s '{tpr.resolve()}'",
        f"-p '{top.resolve()}'",
        f"-n '{sol_index.resolve()}'",
        f"-pname {positive_ion}",
        f"-pq {positive_charge}",
        f"-nname {negative_ion}",
        f"-nq {negative_charge}",
        f"-conc {concentration}",
        f"-o '{ionic_gro.resolve()}'",
    ]
    if neutral:
        command.append("-neutral")

    _utils.run(
        command=command,
        title=f"genion {top.name}",
        workdir=workdir,
    )
    return GenionOut(gro=ionic_gro, top=top)


# TODO: refactor
def trjconv(  # noqa: PLR0913
    workdir: Path,
    trajectory: Path,
    reference: Path,
    index: Path,
    out_name: str,
    dt: float | None = None,
    start: float | None = None,
    end: float | None = None,
    separate: bool = False,
    pbc_type: str | None = None,
    unit_representation: str | None = None,
    center_atoms: bool = False,
    center_box: str | None = None,
    box: tuple[float, float, float] | None = None,
    fit_type: str | None = None,
    trans: tuple[float, float, float] | None = None,
    shift: tuple[float, float, float] | None = None,
    write_velocity: bool = False,
    write_force: bool = False,
    groups: list[str] | None = None,
) -> list[Path]:
    """Convert trajectory to frames, shifting/rotating etc by gmx trjconv.
    Index must contain only 1 group.

    :param workdir: Workdir path
    :param trajectory: `-f` Trajectory .xtc/.trr file
    :param reference: `-s` Geometry .pdb/.gro/.tpr file
    :param index: `-n` Index .ndx file
    :param out_name: `-o` Frames names, out -> out1, out2 ...
    :param dt: `-dt` Time delta for split
    :param start: `-b` Start time to make frames in `ps`, defaults to None (trajectory start)
    :param end: `-e` End time to make frames in `ps`, defaults to None (trajectory end)
    :param separate: `-sep` Separate trajectory to frames, defaults to False
    :param fit: `-fit` Fit molecule in frame to geometry, defaults to None
    :param pbc: `-pbc` PBC treatment, defaults to None
    :param ur: `-ur` Unit-cell representation, defaults to None
    :param center: `-center` Center atoms in box, defaults to False
    :param boxcenter: `-boxcenter` Center for -pbc and -center, defaults to None
    :param vel: `-vel` Read and write velocities if possible, defaults to False
    :param force: `-force` Read and write velocities if possible, defaults to False
    :return: Transformed system file path(s)
    """
    groups = groups or []
    out_path = workdir / f"{out_name}.gro" if separate else workdir / f"{out_name}.xtc"

    command = [
        "gmx trjconv",
        "-nocopyright",
        f"-f '{trajectory.resolve()}'",
        f"-s '{reference.resolve()}'",
        f"-n '{index.resolve()}'",
        f"-o '{out_path.resolve()}'",
    ]

    if dt is not None:
        command.append(f"-dt {dt}")

    if start is not None:
        command.append(f"-b {start}")

    if end is not None:
        command.append(f"-e {end}")

    if separate:
        command.append("-sep")

    if pbc_type:
        command.append(f"-pbc {pbc_type}")

    if unit_representation:
        command.append(f"-ur {unit_representation}")

    if center_atoms:
        command.append("-center")

    if center_box:
        command.append(f"-boxcenter {center_box}")

    if box:
        command.append(f"-box {' '.join(str(i) for i in box)}")

    if fit_type:
        command.append(f"-fit {fit_type}")

    if trans:
        command.append(f"-trans {' '.join(str(i) for i in trans)}")

    if shift:
        command.append(f"-shift {' '.join(str(i) for i in shift)}")

    if write_velocity and (not center_box):
        msg = "Center box must be set to write velocity"
        raise ValueError(msg)
    elif write_velocity:
        command.append("-vel")

    if write_force and (not center_box):
        msg = "Center box must be set to write forces"
        raise ValueError(msg)
    elif write_force:
        command.append("-force")

    if groups:
        groups_text = "\n".join(groups)
        command.append(f"<< EOF\n{groups_text}\nEOF")

    _utils.run(
        command=command,
        title=f"trjconv {trajectory.name}",
        workdir=workdir,
    )
    if separate:
        files = (i for i in workdir.iterdir() if (out_name in i.stem) and "#" not in i.stem)
        return sorted((i for i in files), key=lambda x: int(x.stem.replace(out_name, "")))
    return [workdir / f"{out_name}.xtc"]


class MDRunOut(_utils.PathContainer):
    """MDRun output file path container."""

    gro: Path
    """Geometry .gro file path."""

    edr: Path | None
    """Optional energy .edr file path."""

    trr: Path | None
    """Optional full precision trajectory .trr file path."""

    xtc: Path | None
    """Optional compressed trajectory .xtc file path."""

    xvg: Path | None
    """Optional analyze data .xvg file path."""

    cpt: Path | None
    """Simulation mdrun checkpoint .cpt file path."""

    log: Path
    """Log .log file path."""


# TODO: refactor
def mdrun(  # noqa: PLR0913
    workdir: Path,
    tpr: Path,
    out_name: str,
    plumed: Path | None = None,
    n_mpi: int | None = None,
    n_omp: int | None = None,
) -> MDRunOut:
    """Run simulation.

    :param workdir: Workdir path
    :param tpr: `-s` Processed system binary file
    :param out_name: `-deffnm` Output files name
    :param n_mpi: `-ntmpi` Number of thread-MPI ranks, defaults to None
    :param n_omp: `-ntomp` Number of OpenMP threads per MPI rank, defaults to None
    :return: Path container
    """
    command = [
        "gmx mdrun",
        "-nocopyright",
        f"-s '{tpr.resolve()}'",
        f"-deffnm {out_name}",
    ]

    if n_mpi:
        command.append(f"-ntmpi {n_mpi}")

    if n_omp:
        command.append(f"-ntomp {n_omp}")

    if plumed:
        command.append(f"-plumed {plumed.resolve()}")

    _utils.run(
        command=command,
        title=f"mdrun {out_name}",
        workdir=workdir,
    )
    return MDRunOut(
        gro=workdir / f"{out_name}.gro",
        edr=workdir / f"{out_name}.edr",
        trr=workdir / f"{out_name}.trr",
        xtc=workdir / f"{out_name}.xtc",
        xvg=workdir / f"{out_name}.xvg",
        cpt=workdir / f"{out_name}.cpt",
        log=workdir / f"{out_name}.log",
    )


def make_ndx(
    workdir: Path,
    geometry: Path,
    out_name: str,
    groups: list[str],
) -> Path:
    """Generate all indexes.

    :param workdir: Workdir path
    :param geometry: `-f` Geometry .gro file path
    :param out_name: `-o` Output name
    :return: Index .ndx file path
    """
    index_name = workdir / f"{out_name}.ndx"
    command = [
        "gmx make_ndx",
        "-nocopyright",
        f"-f '{geometry.resolve()}'",
        f"-o '{index_name.resolve()}'",
    ]

    groups = groups.copy()
    groups.append("q")
    selections_text = "\n".join(groups)
    command.append(f"<< EOF\n{selections_text}\nEOF")

    _utils.run(
        command=command,
        title=f"make_ndx {geometry.name}",
        workdir=workdir,
    )
    return index_name
