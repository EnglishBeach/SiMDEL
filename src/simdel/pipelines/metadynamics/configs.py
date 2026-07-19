"""Metadynamics mdp configs."""

# ruff: disable[D103]
from __future__ import annotations

from pathlib import Path

import pandas as pd

from simdel import _utils, chem, sim
from simdel.pipelines import selections


def Ions(emtol: float) -> sim.GromacsSimulator:
    return sim.GromacsSimulator(
        name="ions",
        integrator=sim.Integrator.Steep,
        nsteps=-1,
        emtol=emtol,
        emstep=0.001,
        nstenergy=0,
        constraints=sim.Constraints.HBonds,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
    )


def EM(emtol: float) -> sim.GromacsSimulator:
    return sim.GromacsSimulator(
        name="em",
        integrator=sim.Integrator.Steep,
        nsteps=-1,
        emtol=emtol,
        emstep=0.001,
        nstenergy=0,
        constraints=sim.Constraints.HBonds,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
    )


def NVT(time: float, T: float) -> sim.GromacsSimulator:
    dt = 0.002
    nsteps = int(time / dt)
    tc_grps = [selections.SelectionGroups.system]
    return sim.GromacsSimulator(
        name="nvt",
        integrator=sim.Integrator.MD,
        nsteps=nsteps,
        dt=dt,
        nstenergy=500,
        nstlog=500,
        nstxout_compressed=500,
        compressed_x_grps=tc_grps,
        constraints=sim.Constraints.HBonds,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
        tcoupl=sim.TCoupl.VRescale,
        tc_grps=tc_grps,
        tau_t=[1.0],
        ref_t=[T],
        gen_vel=True,
        gen_temp=T,
    )


def NPT(time: float, T: float, energy_checks: int, frames: int) -> sim.GromacsSimulator:
    dt = 0.002
    nsteps = int(time / dt)
    write_dt = nsteps // ((energy_checks + 1) * frames)
    return sim.GromacsSimulator(
        name="npt",
        integrator=sim.Integrator.MD,
        nsteps=nsteps,
        tinit=0.0,
        dt=dt,
        continuation=True,
        nstenergy=0,
        nstcalcenergy=write_dt,
        nstxout_compressed=write_dt,
        constraints=sim.Constraints.HBonds,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
        pcoupl=sim.PCoupl.CR,
        compressibility=[4.5e-05],
        ref_p=[1.0],
        refcoord_scaling=sim.RefCoordScaling.COM,
        tcoupl=sim.TCoupl.VRescale,
        tc_grps=[selections.SelectionGroups.system],
        tau_t=[1.0],
        ref_t=[T],
    )


def META(time: float, T: float, energy_checks: int, frames: int) -> sim.GromacsSimulator:
    dt = 0.002
    nsteps = int(time / dt)
    write_dt = nsteps // ((energy_checks + 1) * frames)
    return sim.GromacsSimulator(
        name="meta",
        integrator=sim.Integrator.MD,
        nsteps=nsteps,
        tinit=0.0,
        dt=dt,
        continuation=True,
        nstenergy=write_dt,
        nstcalcenergy=write_dt,
        nstxout_compressed=write_dt,
        constraints=sim.Constraints.HBonds,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
        pcoupl=sim.PCoupl.CR,
        pcoupltype=sim.PCouplType.Isotropic,
        compressibility=[4.5e-05],
        ref_p=[1.0],
        refcoord_scaling=sim.RefCoordScaling.COM,
        tcoupl=sim.TCoupl.VRescale,
        tc_grps=[selections.SelectionGroups.system],
        tau_t=[1.0],
        ref_t=[T],
    )


# TODO: refactor
def Funnel(  # noqa: PLR0913
    funnel: chem.Funnel,
    T: float,
    system: chem.System,
    site_mask: pd.Series[bool],
    ligand_mask: pd.Series[bool],
    reference_file: Path,
    sigma: float = 0.2,
    height: float = 2,
    pace: int = 500,
) -> sim.PlumedMDP:
    stride = 500
    v = system.geometry_view
    site_str = _make_str(v[site_mask])
    ligand_str = _make_str(v[ligand_mask])

    a = funnel.center
    b = funnel.center + funnel.vector
    h = round(funnel.h, 2)
    cone_h = round(funnel.cone_h, 2)
    cone_angle = round(funnel.cone_angle, 2)
    tube_r = round(funnel.tube_r, 2)

    points_str = ",".join([str(round(i, 3)) for i in [*a, *b]])
    delta = 0.5
    # GRID_WSTRIDE=250000
    text = f"""WHOLEMOLECULES ENTITY0={site_str},{funnel.anchor},{ligand_str}

site: COM ATOMS={site_str}
lig: COM ATOMS={ligand_str}

fps: FUNNEL_PS ...
    LIGAND=lig
    REFERENCE={reference_file.name} ANCHOR={funnel.anchor}
    POINTS={points_str}
...

funnel: FUNNEL ...
    ARG=fps.lp,fps.ld
    ZCC={cone_h} ALPHA={cone_angle} RCYL={tube_r}
    MINS={-delta} MAXS={h + delta} KAPPA=35000
    NBINS=500 NBINZ=500
    FILE=funnel.dat
...

metad: METAD ...
    ARG=fps.lp
    SIGMA={sigma} HEIGHT={height} PACE={pace}
    TEMP={T}
    GRID_MIN={-delta} GRID_MAX={h + delta} GRID_BIN=100
    CALC_RCT
    RCT_USTRIDE=100
    BIASFACTOR=20
    FILE=hills.dat
...

lwall: LOWER_WALLS ARG=fps.lp AT=0 KAPPA=35000 EXP=2 OFFSET=0
uwall: UPPER_WALLS ARG=fps.lp AT={h} KAPPA=35000 EXP=2 OFFSET=0

PRINT STRIDE={stride} ARG=* FILE=colvar.dat"""
    return sim.PlumedMDP(
        name="metadynamics",
        dt=stride * 0.002,
        text=text,
        other_files=[reference_file],
        out_data=["colvar", "funnel", "hills"],
    )


def _make_str(masked: _utils.Table):
    return ",".join(str(i) for i in masked.index + 1)
