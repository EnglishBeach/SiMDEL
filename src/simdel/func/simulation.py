"""High level assembly system and simulation functions."""

from __future__ import annotations

import gzip
from pathlib import Path
import re
import shutil

from frozendict import frozendict
import pandas as pd

from simdel import _utils, chem, sim
from simdel._wrappers import gromacs


# TODO: refactor parameters+plumed_parameters-> simulator
# TODO: refactor n_mpi,n_omp,compress
@_utils.require(gromacs)
def simulate(  # noqa: PLR0913
    system: chem.System,
    parameters: sim.GromacsSimulator,
    workdir: Path,
    plumed_parameters: sim.PlumedMDP | None = None,
    n_mpi: int | None = None,
    n_omp: int | None = None,
    compress: bool = True,
) -> tuple[chem.System, chem.Trajectory | None, chem.EnergyDump | None, dict[str, Path]]:
    """Run simulation stage configured by MDP config.
    Can set positional restraints for all - float, for exact topologies - dict.

    :param system: System to simulate
    :param parameters: MD simulation parameters
    :param workdir: Workdir path
    :param plumed_parameters: PLUMED simulation parameters, defaults to None
    :param n_mpi: N mpi for gromacs
    :param n_omp: N omp for gromacs
    :param compress_log: Compress internal log file, defaults to True
    :return: System and trajectory
    """
    _check_groups(parameters=parameters, indexes=system.index)
    workdir.mkdir(parents=True, exist_ok=True)

    system_files = system.save(workdir)
    preprocessed = gromacs.grompp(
        workdir=workdir,
        geometry=system_files.gro,
        top=system_files.top,
        posres_geometry=system_files.gro,
        mdp=parameters.save(workdir),
        index=system_files.index,
        out_name=system.name,
        maxwarn=5 if not _utils.STRICT else 0,
    )

    run_files = gromacs.mdrun(
        workdir=workdir,
        tpr=preprocessed.tpr,
        plumed=plumed_parameters.save(workdir) if plumed_parameters else None,
        out_name=system.name,
        n_mpi=n_mpi,
        n_omp=n_omp,
    )

    simulated_system = (
        chem.System.load(
            top=system_files.top,
            gro=run_files.gro,
        )
        .set_info(
            ff_type=system.info.ff_type,
            water_type=system.info.water_type,
            water_flexibility=system.info.water_flexibility,
        )
        .set_indexes(**system.index)
    )

    # TODO: may be save only trr file always
    if run_files.xtc:
        trajectory = chem.Trajectory(
            file=run_files.xtc,
            dt=parameters.dt or 0.001,
            frames=parameters.nsteps or 0,
        )
        run_files.trr.unlink() if run_files.trr else None
    elif run_files.trr:
        trajectory = chem.Trajectory(
            file=run_files.trr,
            dt=parameters.dt or 0.001,
            frames=parameters.nsteps or 0,
        )
    else:
        trajectory = None

    # TODO: write only edr+evg or none
    if run_files.edr or run_files.xvg:
        energy = chem.EnergyDump(
            edr=run_files.edr,
            xvg=run_files.xvg,
        )
    else:
        energy = None

    plumed_data = plumed_parameters.get_data(workdir) if plumed_parameters else {}

    # TODO: compress -> context
    if compress:
        _compress(
            workdir=workdir,
            full_mdp_name=f"{system.name}_{parameters.name}.mdp",
            cpt_file=run_files.cpt,
            tpr_file=preprocessed.tpr,
            log_file=run_files.log,
        )

    return simulated_system, trajectory, energy, plumed_data


def _compress(
    workdir: Path,
    full_mdp_name: str,
    cpt_file: Path | None,
    tpr_file: Path,
    log_file: Path,
):
    """Compress files after simulation.

    :param workdir: Workdir path
    :param full_mdp_name: Gromacs generated mdp name
    :param cpt_file: Checkpoint .cpt file path
    :param tpr_file: Binary system .tpr file path
    :param log_file: Log .log file path
    """
    [i.unlink() for i in workdir.iterdir() if re.match(r"\#.*\#", i.stem)]
    (workdir / full_mdp_name).unlink()
    cpt_file.unlink() if cpt_file else None
    tpr_file.unlink()

    zip_file = workdir / f"{log_file.stem}.zip"
    with log_file.open("rb") as f_in, gzip.open(zip_file, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    log_file.unlink()


def _check_groups(parameters: sim.GromacsSimulator, indexes: frozendict[str, pd.Series[bool]]):
    """Check selections in MDP and system selections.

    :param parameters: MDP object
    :param selections: System selection dict
    """
    index_set = set(indexes)
    tc_grps = set(parameters.tc_grps)
    compressed_x_grps = set(parameters.compressed_x_grps)

    if any((i - index_set) != set() for i in [tc_grps, compressed_x_grps]):
        msg = (
            f"Selections not fully contain all mdp groups:\n"
            f"selections: {' '.join(index_set)}\n"
            f"tc_grps: {' '.join(tc_grps)}\n"
            f"compressed_x_grps: {' '.join(compressed_x_grps)}"
        )
        raise ValueError(msg)
