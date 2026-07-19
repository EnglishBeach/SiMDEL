"""MD pipeline configs."""

from simdel import sim
from simdel.pipelines import selections


def add_ions_mdp(emtol: float = 100, emstep: float = 0.01) -> sim.GromacsSimulator:
    """Generate config .mdp for MD add ions stage.

    :param emtol: Min value for maximum force to stop simulation, in `kJ/(mol*nm)`
    :param emstep: Initial step-size for energy minimization, in `nm`
    :return: Add ions mdp
    """
    return sim.GromacsSimulator(
        name="ions",
        # Run control
        integrator=sim.Integrator.Steep,
        # TI
        emtol=emtol,
        emstep=emstep,
        nsteps=-1,
        # Neighbor searching
        nstlist=1,
        cutoff_scheme=sim.CutoffScheme.Verlet,
        pbc=sim.PBC.XYZ,
        # Electrostatics
        coulombtype=sim.CoulombType.Cutoff,
        rcoulomb=1.2,
        rvdw=1.2,
    )


def em_mdp(emtol: float = 100, emstep: float = 0.001) -> sim.GromacsSimulator:
    """Generate config .mdp for MD EM stage.

    :param emtol: Min value for maximum force to stop simulation, in `kJ/(mol*nm)`
    :param emstep: Initial step-size for energy minimization, in `nm`
    :return: EM mdp
    """
    return sim.GromacsSimulator(
        name="em",
        # Run control
        integrator=sim.Integrator.Steep,
        # EM
        emtol=emtol,
        emstep=emstep,
        # TI
        nsteps=-1,
        # Neighbor searching
        nstlist=1,
        cutoff_scheme=sim.CutoffScheme.Verlet,
        pbc=sim.PBC.XYZ,
        # Electrostatics
        coulombtype=sim.CoulombType.PME,
        rcoulomb=1.2,
        # Van der Waals
        dispcorr=sim.DispCorr.EnerPres,
        rvdw=1.2,
        # Ewald summing
        pme_order=4,
        fourierspacing=0.16,
    )


def nvt_mdp(
    nsteps: int = 50000,
    dt: float = 0.002,
    T: float = 298,
    tc_grps: list[str] | None = None,
) -> sim.GromacsSimulator:
    """Generate config .mdp for MD NVT stage.

    :param nsteps: Maximum number of steps to integrate or minimize, -1 -infinite time, in `steps`
    :param dt: Timestep, in `ps`
    :param T: Target temperature, in `K`
    :param tc_grps: Molecule groups to set temperature
    :return: NVT mdp
    """
    tc_grps = tc_grps or [
        selections.BaseSelectionGroups.protein,
        selections.BaseSelectionGroups.nonProtein,
    ]
    return sim.GromacsSimulator(
        name="nvt",
        # Run control
        integrator=sim.Integrator.MD,
        # TI
        nsteps=nsteps,
        dt=dt,
        # Output control
        nstxout=0,
        nstvout=0,
        nstfout=0,
        nstenergy=500,
        nstlog=500,
        nstxout_compressed=500,
        compressed_x_grps=[selections.BaseSelectionGroups.system],
        # Neighbor searching
        nstlist=20,
        cutoff_scheme=sim.CutoffScheme.Verlet,
        pbc=sim.PBC.XYZ,
        # Electrostatics
        coulombtype=sim.CoulombType.PME,
        rcoulomb=1.2,
        # Van der Waals
        dispcorr=sim.DispCorr.EnerPres,
        rvdw=1.2,
        # Ewald summing
        pme_order=4,
        fourierspacing=0.16,
        # Temperature coupling
        tcoupl=sim.TCoupl.VRescale,
        tc_grps=tc_grps,
        tau_t=[0.1] * len(tc_grps),
        ref_t=[T] * len(tc_grps),
        # Velocity generation
        gen_vel=True,
        gen_temp=T,
        gen_seed=-1,
        # Bonds
        continuation=False,
        constraint_algorithm=sim.ConstraintAlgorithm.LINCS,
        constraints=sim.Constraints.HBonds,
        lincs_iter=1,
        lincs_order=4,
    )


def npt_mdp(
    nsteps: int = 50000,
    dt: float = 0.002,
    T: float = 298,
    tc_grps: list[str] | None = None,
    P: float = 1,
) -> sim.GromacsSimulator:
    """Generate config .mdp for MD NPT stage.

    :param nsteps: Maximum number of steps to integrate or minimize, -1 -infinite time, in `steps`
    :param dt: Timestep, in `ps`
    :param T: Target temperature in `K`
    :param tc_grps: Molecule groups to set temperature
    :param P: Target pressure
    :return: NPT mdp
    """
    tc_grps = tc_grps or [
        selections.BaseSelectionGroups.protein,
        selections.BaseSelectionGroups.nonProtein,
    ]
    return sim.GromacsSimulator(
        name="npt",
        # Run control
        integrator=sim.Integrator.MD,
        # TI
        nsteps=nsteps,
        dt=dt,
        # Output control
        nstxout=0,
        nstvout=0,
        nstfout=0,
        nstenergy=500,
        nstlog=500,
        nstxout_compressed=500,
        compressed_x_grps=[selections.BaseSelectionGroups.system],
        # Neighbor searching
        nstlist=20,
        cutoff_scheme=sim.CutoffScheme.Verlet,
        pbc=sim.PBC.XYZ,
        # Electrostatics
        coulombtype=sim.CoulombType.PME,
        rcoulomb=1.2,
        # Van der Waals
        dispcorr=sim.DispCorr.EnerPres,
        rvdw=1.2,
        # Ewald summing
        pme_order=4,
        fourierspacing=0.16,
        # Temperature coupling
        tcoupl=sim.TCoupl.VRescale,
        tc_grps=tc_grps,
        tau_t=[0.1] * len(tc_grps),
        ref_t=[T] * len(tc_grps),
        # Temperature coupling
        pcoupl=sim.PCoupl.PR,
        pcoupltype=sim.PCouplType.Isotropic,
        tau_p=2.0,
        ref_p=[P],
        compressibility=[4.5e-5],
        refcoord_scaling=sim.RefCoordScaling.COM,
        # Velocity generation
        gen_vel=False,
        # Bonds
        continuation=True,
        constraint_algorithm=sim.ConstraintAlgorithm.LINCS,
        constraints=sim.Constraints.HBonds,
        lincs_iter=1,
        lincs_order=4,
    )


def product_mdp(
    nsteps: int = 50000,
    dt: float = 0.002,
    T: float = 298,
    tc_grps: list[str] | None = None,
    P: float = 1,
) -> sim.GromacsSimulator:
    """Generate config .mdp for MD MD stage.

    :param nsteps: Maximum number of steps to integrate or minimize, -1 -infinite time, in `steps`
    :param dt: Timestep, in `ps`
    :param T: Target temperature in `K`
    :param tc_grps: Molecule groups to set temperature
    :param P: Target pressure
    :return: NPT mdp
    """
    tc_grps = tc_grps or [
        selections.BaseSelectionGroups.protein,
        selections.BaseSelectionGroups.nonProtein,
    ]
    return sim.GromacsSimulator(
        name="npt",
        # Run control
        integrator=sim.Integrator.MD,
        # TI
        nsteps=nsteps,
        dt=dt,
        # Output control
        nstxout=0,
        nstvout=0,
        nstfout=0,
        nstenergy=1000,
        nstlog=1000,
        nstxout_compressed=1000,
        compressed_x_grps=[selections.BaseSelectionGroups.system],
        # Bonds
        continuation=True,
        constraint_algorithm=sim.ConstraintAlgorithm.LINCS,
        constraints=sim.Constraints.HBonds,
        lincs_iter=1,
        lincs_order=4,
        # Neighbor searching
        nstlist=20,
        cutoff_scheme=sim.CutoffScheme.Verlet,
        pbc=sim.PBC.XYZ,
        # Van der Waals
        dispcorr=sim.DispCorr.EnerPres,
        rvdw=1.2,
        # Electrostatics
        coulombtype=sim.CoulombType.PME,
        rcoulomb=1.2,
        # Ewald summing
        pme_order=4,
        fourierspacing=0.16,
        # Temperature coupling
        tcoupl=sim.TCoupl.VRescale,
        tc_grps=tc_grps,
        tau_t=[0.1] * len(tc_grps),
        ref_t=[T] * len(tc_grps),
        # Temperature coupling
        pcoupl=sim.PCoupl.PR,
        pcoupltype=sim.PCouplType.Isotropic,
        tau_p=2.0,
        ref_p=[P],
        compressibility=[4.5e-5],
    )
