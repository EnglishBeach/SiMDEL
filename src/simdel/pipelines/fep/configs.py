"""FEP mdp configs."""

from simdel import sim
from simdel.pipelines import selections


def Ions(
    forward: bool,
    emtol: float,
) -> sim.GromacsSimulator:
    """Generate config .mdp for FEP pipeline's add-ions stage.

    :param forward: True for A stage, False - for B
    :param emtol: Min value for maximum force to stop simulation in `kJ/(mol*nm)`
    :return: MDP
    """
    return sim.GromacsSimulator(
        name="ions",
        integrator=sim.Integrator.Steep,
        nsteps=-1,
        emtol=emtol,
        emstep=0.001,
        nstenergy=0,
        nstcheckpoint=0,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
        constraints=sim.Constraints.HBonds,
        free_energy=sim.FreeEnergy.YES,
        init_lambda=0 if forward else 1.0,
        delta_lambda=0.0,
        sc_alpha=0.3,
        sc_sigma=0.25,
        sc_power=1,
        sc_coul=True,
    )


def EM(
    forward: bool,
    emtol: float,
) -> sim.GromacsSimulator:
    """EM parameters.

    :param forward: True for A stage, False - for B
    :param emtol: Min value for maximum force to stop simulation in `kJ/(mol*nm)`
    :return: MDP
    """
    return sim.GromacsSimulator(
        name="em",
        integrator=sim.Integrator.Steep,
        nsteps=-1,
        emtol=emtol,
        emstep=0.001,
        nstenergy=0,
        nstcheckpoint=0,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
        constraints=sim.Constraints.HBonds,
        free_energy=sim.FreeEnergy.YES,
        init_lambda=0 if forward else 1.0,
        delta_lambda=0.0,
        sc_coul=True,
        sc_alpha=0.3,
        sc_sigma=0.25,
    )


# TODO: refactor
def EQ(  # noqa: PLR0913
    forward: bool,
    time: float,
    T: float,
    ti_frames: int,
    energy_checks: int,
    dt: float = 0.002,
) -> sim.GromacsSimulator:
    """EQ parameters.

    :param forward: True for A stage, False - for B
    :param time: EQ stage time in `ps`
    :param temperature: Temperature to simulate in `K`
    :param ti_frames: Number of transition frames to calculate
    frequency of energy state saving in trajectory file
    :param energy_checks: Number os steps between frames to save to check energy
    :param dt: Timestep in `ps`
    :return: MDP
    """
    nsteps = int(time / dt)
    write_dt = nsteps // ((energy_checks + 1) * ti_frames)
    return sim.GromacsSimulator(
        name="eq",
        integrator=sim.Integrator.SD,
        nsteps=nsteps,
        tinit=0.0,
        dt=dt,
        nstenergy=0,
        nstcheckpoint=0,
        nstcalcenergy=100,
        nstlog=write_dt * 10,
        nstxout_compressed=write_dt,
        nstdhdl=100,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
        tc_grps=[selections.BaseSelectionGroups.system],
        tau_t=[2.0],
        ref_t=[T],
        pcoupl=sim.PCoupl.PR,
        pcoupltype=sim.PCouplType.Isotropic,
        compressibility=[4.6e-05],
        ref_p=[1.0],
        constraints=sim.Constraints.HBonds,
        constraint_algorithm=sim.ConstraintAlgorithm.LINCS,
        continuation=True,
        lincs_iter=2,
        free_energy=sim.FreeEnergy.YES,
        init_lambda=0 if forward else 1.0,
        delta_lambda=0,
        sc_alpha=0.3,
        sc_sigma=0.25,
        sc_power=1,
        sc_coul=True,
    )


def NEQ(
    forward: bool,
    time: float,
    T: float,
    dt: float = 0.002,
) -> sim.GromacsSimulator:
    """NEQ parameters.

    :param forward: True for A stage, False - for B
    :param time: NEQ stage time in `ps`
    :param temperature: Temperature to simulate in `K`
    :param dt: Timestep in `ps`
    :return: MDP
    """
    nsteps = int(time / dt)
    return sim.GromacsSimulator(
        name="neq",
        integrator=sim.Integrator.SD,
        nsteps=nsteps,
        tinit=0.0,
        dt=dt,
        nstcalcenergy=1,
        nstenergy=min(5000, nsteps // 50),
        nstlog=1000,
        nstcheckpoint=0,
        nstdhdl=1,
        coulombtype=sim.CoulombType.PME,
        dispcorr=sim.DispCorr.EnerPres,
        fourierspacing=0.125,
        # tcouple ignored
        tc_grps=[selections.BaseSelectionGroups.system],
        tau_t=[2.0],
        ref_t=[T],
        gen_vel=False,
        pcoupl=sim.PCoupl.PR,
        pcoupltype=sim.PCouplType.Isotropic,
        tau_p=5.0,
        compressibility=[4.6e-05],
        ref_p=[1.0],
        constraints=sim.Constraints.HBonds,
        constraint_algorithm=sim.ConstraintAlgorithm.LINCS,
        continuation=True,
        lincs_iter=2,
        free_energy=sim.FreeEnergy.YES,
        init_lambda=0 if forward else 1.0,
        delta_lambda=(1 / nsteps) if forward else (-1 / nsteps),
        sc_alpha=0.3,
        sc_sigma=0.25,
        sc_power=1,
        sc_coul=True,
    )
