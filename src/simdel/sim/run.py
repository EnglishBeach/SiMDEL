"""Molecular dynamic parameters for Run control, motion remove, Langevin dynamics,
energy minimization, shell dynamics, particle insertion, output control.
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

import pydantic

from . import core_mdp


class Integrator(enum.Enum):
    """Time (dynamic) or energy minimization integrator."""

    # Dynamic integrators (Dynamic)
    MD = "md"
    """A leap-frog algorithm for integrating Newton's equations of motion."""

    MDVV = "md-vv"
    """A velocity Verlet algorithm for integrating Newton's equations of motion,
    more accurate then `MD` integrator, slowly."""

    MDVVAvek = "md-vv-avek"
    """Similar MDVV, the kinetic energy is determined as the average of
    the two half step kinetic energies, more accurate.

    - `NoseHoover and/or PR` => increase in computational cost"""

    SD = "sd"
    """An accurate and efficient leap-frog stochastic dynamics integrator.

    > Controlled by:
    - `ref_t` - reference T
    - `tc_grps` - T coupling groups
    - `tau_t` - inverse friction constant for each group
    - `tcoupl`, `nsttcouple` - ignored
    - `ld-seed` - random seed"""

    BD = "bd"
    """An Euler integrator for Brownian or position Langevin dynamics.

    - `bd_fric > 0` => velocity=force/`bd_fric` and friction coefficient = `bd_fric`
    - `bd_fric = 0` => friction coefficient = mass / `tau_t`
    > Controlled by:
    - `ref_t` - random thermal noise
    - `ld-seed` - random seed"""

    # Minimization 'integrators' (EM)
    Steep = "steep"
    """A steepest descent algorithm for energy minimization.

    > Controlled by:
    - `emstep` - maximum step size
    - `emtol` - tolerance"""

    CG = "cg"
    """A conjugate gradient algorithm for energy minimization, more efficient to use nstcgsteep,
    GROMACS should be compiled in double precision.

    > Controlled by:
    - `emtol` - tolerance
    - `nstcgsteep` - frequency of steepest descent usage during calculation"""

    LBFGS = "l-bfgs"
    """A quasi-Newtonian algorithm for energy minimization according to
    the low-memory Broyden-Fletcher-Goldfarb-Shanno approach,
    faster than CG, it is not parallelized."""

    # Tests
    NM = "nm"
    """Normal mode analysis is performed on the structure in the tpr file,
    GROMACS should be compiled in double precision."""

    Tpi = "tpi"
    """Test particle insertion, the test particle - the last molecule in the topology.
    For charged molecules, using PME with a fine grid is most accurate and also efficient,
    since the potential in the system only needs to be calculated once per frame.
    No trajectory or energy file is written.
    Use trajectory (should not contain the molecule to be inserted) => must use mdrun -rerun.

    - `nstlist > 1] => `nstlist` insertions are performed in a sphere
    > Controlled by:
    - `nsteps` - frequency of performed insertions
    - `ld_seed` - random seed
    - `ref_t` - temperature for the Boltzmann weighting
    - `rtpi` - sphere radius"""

    Tpic = "tpic"
    """Test particle insertion into a predefined cavity location, same as for `Tpi`,
    the molecule to be inserted should be centered at 0,0,0.

    > Controlled by:
    - `rtpi` - sphere radius"""

    # QM/MM
    # TODO: fix doc
    MiMiC = "mimic"
    """MiMiC QM/MM coupling to run hybrid molecular dynamics.

    > Controlled by:
    - t-coupling, p-coupling, timestep, number os steps - ignored
    - PME, cut-off ... - work as usual
    - `QMMM_groups` - define QM atoms"""


# TODO: add descriptions
class MTSForce(enum.Enum):
    """Force type."""

    LongrangeNonbonded = "longrange-nonbonded"
    Nonbonded = "nonbonded"
    Pair = "pair"
    Dihedral = "dihedral"
    Angle = "angle"
    Pull = "pull"
    Awh = "awh"


class GroupRunControl(pydantic.BaseModel):
    """Time integration/energy minimization parameters."""

    integrator: Integrator
    """Integration method type."""

    tinit: float | None = None
    """Start time, in `ps`. (0)

    - ACTIVE IF dynamic"""

    dt: float | None = None
    """Integration timestep, in `ps`. (0.001)

    - ACTIVE IF dynamic"""

    nsteps: int | None = None
    """Maximum number of steps to integrate or minimize, in `steps`. (0)

    - `= -1` => infinite"""

    init_step: int | None = None
    """The starting step, in `steps`. (0)

    => simulation step = `tinit` + `dt` * (`init_step` + i)
    => lambda = `init_lambda` + `delta_lambda` * (`init_step` + i)"""

    # TODO: may be depends on groups
    simulation_part: int | None = None
    """A simulation can consist of multiple parts, each of which has a part number. (1)"""


class GroupMultiTimeStepping(pydantic.BaseModel):
    """Multiple timing-stepping integrator parameters."""

    mts: core_mdp.bool_yn | None = None
    """Use a multiple timing-stepping integrator to evaluate some forces. (False)

    - ACTIVE IF intergator = `MD`.
    - `False` => evaluate all forces at every timestep
    > Controlled by:
    - `mts_level2_forces`,`mts_level2_factor` - configure"""

    mts_levels: int | None = None
    """(2)"""

    mts_level2_forces: MTSForce | None = None
    """A list of one or more force groups that will be evaluated only
    every `mts_level2_factor` steps. (LongrangeNonbonded)

    > Controlled by:
    - `mts_level2_factor` - frequency of evaluate forces"""

    mts_level2_factor: int | None = None
    """Interval for computing the forces in level 2 of the multiple time-stepping scheme
    In `steps`. (2)"""

    mass_repartition_factor: float | None = None
    """Scales the masses of the lightest atoms in the system by this factor to the mass mMin. (1)

    - `all atoms masses < mMin` => mMin set to atoms
    - light atom is bound to another => mass(light) - mass change
    - there is no bound => warning
    - more 1 atom bound => error
    - `bound atom <= mMin` => error"""


class CommMode(enum.Enum):
    """Center of mass motion removal type."""

    Linear = "Linear"
    """Remove center of mass translational velocity."""

    Angular = "Angular"
    """Remove center of mass translational and rotational velocity around the center of mass."""

    LAC = "Linear-acceleration-correction"
    """Remove center of mass translational velocity, use when expected on the center of mass moving.

    > Controlled by:
    - `nstcomm` - frequency of corrections steps"""

    NONE = "None"
    """No restriction on the center of mass motion."""


class GroupMotionRemove(pydantic.BaseModel):
    """Correct velocities to remove all system moving."""

    comm_mode: CommMode | None = None
    """Center of mass motion removal type. (Linear)"""

    nstcomm: int | None = None
    """Number of steps for center of mass motion removal, in `steps`. (100)"""

    comm_grps: list[str] = []
    """Group(s) for center of mass motion removal. ([])

    - [] => all system"""


class GroupLangevinDynamics(pydantic.BaseModel):
    """Langevin integration parameters."""

    bd_fric: float | None = None
    """Brownian dynamics friction coefficient, in `amu/ps`. (0)

    - `= 0` => mass/ tau-t"""

    ld_seed: int | None = None
    """Initialize random generator for thermal noise for stochastic and Brownian dynamics. (-1)

    - `= -1` => random seed"""


class GroupEnergyMinimization(pydantic.BaseModel):
    """Parameters only for energy minimization."""

    emtol: float | None = None
    """Min value for maximum force to stop simulation, in `kJ/(mol*nm)`. (10)"""

    emstep: float | None = None
    """Initial step-size for energy minimization, in `nm`. (0.01)"""

    nstcgsteep: int | None = None
    """Frequency of performing 1 steepest descent step for CG, in `steps`. (1000)"""

    nbfgscorr: int | None = None
    """Number of correction steps to use for L-BFGS minimization.
    More accurate, slower, in `steps`.
    """


class GroupShell(pydantic.BaseModel):
    """Energy minimization for shell or flexible constraints."""

    niter: int | None = None
    """Maximum number of iterations for optimizing the shell positions
    and the flexible constraints. (20)"""

    fcstep: int | None = None
    """Step size for optimizing the flexible constraints, in `ps^2`. (0)

    Should be chosen as mu/(d2V/dq2):
    mu - reduced mass
    d2V/dq2 - derivative of the potential in the constraint direction"""


class GroupParticleInsertion(pydantic.BaseModel):
    """Parameter for test particle insertion simulations."""

    rtpi: float | None = None
    """The test particle insertion radius, in `nm`. (0.05)"""


class GroupOutControl(pydantic.BaseModel):
    """Output parameters."""

    nstxout: int | None = None
    """Frequency for writing coords (x) in trajectory .trr file, in `steps`. (0)"""

    nstvout: int | None = None
    """Frequency for writing velocities (v) in trajectory .trr file, in `steps`. (0)"""

    nstfout: int | None = None
    """Frequency for writing forces (f) in trajectory .trr file, in `steps`. (0)"""

    nstlog: int | None = None
    """Frequency for writing energies to .log file, in `steps`. (1000)"""

    nstcalcenergy: int | None = None
    """Frequency for calculating the energies, in `steps`. (100)

    - ACTIVE IF dynamic"""

    nstenergy: int | None = None
    """Frequency for writing energies to energy .edr file. = `nstcalcenergy`.
    First and last enerdies always are written, in `steps`. (1000)"""

    nstxout_compressed: int | None = None
    """Frequency for writing coords (x) in compressed trajectory .xtc file, in `steps`. (0)"""

    compressed_x_precision: float | None = None
    """Fprecision for compressed trajectory file, in `digits`. (1000)"""

    compressed_x_grps: list[str] = []
    """Group(s) to write to the compressed trajectory .xtc file. ([])

    - ACTIVE IF `nstxout_compressed` > 0
    - `= []` => all system"""

    energygrps: list[str] = []
    """Group(s) for which to write to write short-ranged non-bonded potential energies
    to the energy file. ([])

    - ACTIVE IF GPU is off
    - `= []` => all system"""

    nstcheckpoint: int | None = None
    """Frequency for checkpointing, helps to continue after crashes, in `steps`. ()"""
