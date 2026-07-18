"""Molecular dynamic parameters for TCouplings, PCouplings, Annealing, Velocity generation.
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

from pydantic import BaseModel

from . import core_mdp


class EnsembleTemperatureSetting(enum.Enum):
    """Temperature setting type."""

    Auto = "auto"
    """Automatic select `ensemble_temperature_setting`.

    - `all atoms t-coupled` => `ensemble_temperature_setting` = `Constant`"""

    Constant = "constant"
    """Constant ensemble temperature, is required for certain sampling algorithms.

    > Controlled by:
    - `ensemble_temperature` - system temperature"""

    Variable = "variable"
    """The system ensemble temperature is set dynamically during the simulation,
    for tempering or annealing."""

    NotAvailable = "not-available"
    """The system has no ensemble temperature."""


class TCoupl(enum.Enum):
    """Temperature coupling type."""

    NO = "no"
    """No temperature coupling."""

    Berendsen = "berendsen"
    """Temperature coupling with a Berendsen-thermostat.

    > Controlled by:
    - `ref_t` - thermostat temperature
    - `tc_grps` - groups can be coupled separately
    - `tau_t` - thermostat time constant"""

    NoseHoover = "nose-hoover"
    """Temperature coupling using a Nose-Hoover extended ensemble.

    > Controlled by:
    - `ref_t` - thermostat temperature
    - `tc_grps` - groups can be coupled separately
    - `tau_t` - temperature fluctuations at equilibrium"""

    Andersen = "andersen"
    """Temperature coupling by randomizing a fraction
    of the particle velocities at each timestep.

    - ACTIVE IF `cutoff_scheme` = `Verlet` or no constraints
    > Controlled by:
    - `ref_t` - thermostat temperature
    - `tc_grps` - groups can be coupled separately
    - `tau_t` - average time between randomization of each molecule"""

    AndersenMassive = "andersen-massive"
    """Temperature coupling by randomizing velocities of
    all particles at infrequent timesteps.

    > Controlled by:
    - `ref_t` - thermostat temperature
    - `tc_grps` - groups can be coupled separately
    - `tau_t` - time between randomization of all molecules"""

    VRescale = "v-rescale"
    """Temperature coupling using velocity rescaling with a stochastic term.

    - ACTIVE CORRECTLY IF `tau_t` = 0
    > Controlled by:
    - `tau_t` - thermostat time constant
    - `ld-seed` - random seed"""


class GroupTemperatureCoupling(BaseModel):
    """Temperature coupling parameters."""

    ensemble_temperature_setting: EnsembleTemperatureSetting | None = None
    """Temperature setting type. (Auto)"""

    ensemble_temperature: float | None = None
    """The ensemble temperature for the system, in `K`. (-1)

    - ACTIVE IF `ensemble_temperature_setting` = `Constant`
    - `= 1` => temperature copied from thermal bath temperature"""

    tcoupl: TCoupl | None = None
    """Temperature coupling type. (NO)"""

    nsttcouple: int | None = None
    """The interval between steps that couple the temperature, in `steps`. (-1)

    - `= -1` => 100 `steps`
    - `= 1` => for `Verlet` integrator"""

    nh_chain_length: int | None = None
    """The number of chained Nose-Hoover thermostats for velocity Verlet integrators.
    Data for NH chain variable not write in energy .edr file. (10)

    - `integrator`= `MD` => 1
    > Controlled by:
    - `print-nose-hoover-chain-variable` - write thermostats data to energy .edr file"""

    print_nose_hoover_chain_variables: core_mdp.bool_yn | None = None
    """Store all positions and velocities of the Nose-Hoover chain in the energy file. (False)"""

    tc_grps: list[str] = []
    """Groups to couple to separate temperature baths. ([])

    - `= []` => all system."""

    tau_t: list[float] = []
    """Time constant for coupling (one for each group in tc-grps), in `ps`. ([])

    - `= -1` => no temperature coupling"""

    ref_t: list[float] = []
    """Reference temperature for couplin, for each group in `tc_grps`, in `K`. ([])"""


class PCoupl(enum.Enum):
    """Pressure coupling type."""

    NO = "no"
    """No pressure coupling. Fixed box size."""

    Berendsen = "Berendsen"
    """Exponential relaxation pressure coupling,does not yield a correct thermodynamic ensemble.

    > Controlled by:
    - `tau_p` - time constant
    - `nstpcouple` - steps to scale box"""

    CR = "C-rescale"
    """Exponential relaxation pressure coupling with time constant,
    including a stochastic term to enforce correct volume fluctuations.

    > Controlled by:
    - `tau_p` - time constant"""

    PR = "Parrinello-Rahman"
    """Extended-ensemble pressure coupling where
    the box vectors are subject to an equation of motion.
    May causes big P fluctuation.

    > Controlled by:
    - `tau-p` - period of pressure fluctuations at equilibrium"""

    MTTK = "MTTK"
    """Martyna-Tuckerman-Tobias-Klein implementation.

    - ACTIVE IF `integrator`= `md_vv` or `md_vv_avek`
    > Controlled by:
    - `tau_p` - time constant"""


class PCouplType(enum.Enum):
    """Isotropy type of the pressure coupling."""

    Isotropic = "isotropic"
    """Isotropic pressure coupling with time constant, `compressibility`, `ref_p` for each group.

    > Controlled by:
    - `tau_p` - time constant"""

    Semiisotropic = "semiisotropic"
    """Pressure coupling which is isotropic in the x and y direction, but different in the z.
    2 values: `compressibility`, `ref_p` to each group, in x/y, z directions."""

    Anisotropic = "anisotropic"
    """Pressure coupling for xx, yy, zz, xy/yx, xz/zx, yz/zy directions,
    can lead to extreme deformation of the simulation box.

    > Controlled by:
    - 6 values: `compressibility`, `ref_p` for each group
    - `off-diagonal compressibilities = 0` => rectangular box will stay rectangular"""

    SurfaceTension = "surface-tension"
    """Surface tension coupling for surfaces parallel to the xy-plane,
    normal pressure coupling for the z-direction. (must be accurate, if = 0 => constant box height)

    > Controlled by:
    - 2 values `ref_p` - surface tension times the number of surfaces (`bar*nm`), z-pressure (`bar`)
    - 2 values `compressibility` - compressibility in x/y, z-direction compressibility"""


class RefCoordScaling(enum.Enum):
    """Scale type for reference coordinates."""

    NO = "no"
    """The reference coordinates for position restraints are not modified."""

    All = "all"
    """The reference coordinates are scaled with the scaling matrix of the pressure coupling."""

    COM = "com"
    """Scale the center of mass of the reference coordinates with
    the scaling matrix of the pressure coupling."""


class GroupPressureCoupling(BaseModel):
    """Pressure coupling parameters."""

    pcoupl: PCoupl | None = None
    """Pressure coupling type. (NO)"""

    pcoupltype: PCouplType | None = None
    """Isotropy type of the pressure coupling. (Isotropic)"""

    nstpcouple: int | None = None
    """The interval between steps that couple the pressure, in `steps`. (-1)

    - `= -1` => 100
    - `integrator = Verlet` => `nsttcouple` = 1"""

    tau_p: float | None = None
    """Time constant for pressure coupling, in `ps`. (5)"""

    compressibility: list[float] = []
    """The compressibility, for water 1 atm/300 K = 4.5e-5
    number of values depends of `pcoupltype`, in `1/bar`."""

    ref_p: list[float] = []
    """The reference pressure for coupling,number of values depends of `pcoupltype`.
    In `bar`. ()"""

    refcoord_scaling: RefCoordScaling | None = None
    """Scale type for reference coordinates. (NO)"""


class Annealing(enum.Enum):
    """Type of annealing for each temperature group."""

    NO = "no"
    """No simulated annealing - just couple to reference temperature."""

    Single = "single"
    """A single sequence of annealing points.
    - `time > last point` => temperature will be coupled to this constant value after the annealing.
    """

    Periodic = "periodic"
    """The annealing will start over at the first reference point onces
    the last reference time is reached. Repeated until the simulation ends."""


class GroupAnnealing(BaseModel):
    """Annealing parameters.

    For example:
     - tc_grps = Water Protein
     - annealing = single periodic
     - annealing-npoints = 3 4
     - annealing-time = 0 3 6 0 2 4 6
     - annealing-temp = 298 280 270 298 320 320 298
    =>
     - water: 298K -(at 3ps)-> 280K  -(from 3ps to 6ps)-> 270K  -> constant 270K
     - protein: 298K -(at 2ps)-> 320K  -> 320K (constant until 4ps) \
-(from 4 to 6ps)-> 298K -> repeat
    """

    annealing: list[Annealing] = []
    """Type of annealing for each temperature group. (No)"""

    annealing_npoints: list[float] = []
    """List with the number of annealing reference points used for each temperature group,
    equal the number of `tc_grps`, in `K`. ([])

    - `0 for group` => annealing is off for group, the number of entries should"""

    annealing_time: list[int] = []
    """Times at the annealing points for each group,
    len(`annealing_time`) = len(`annealing_npoints`), in `ps`. ([])

    - `annealing = periodic` => times will be used modulo the last value"""

    annealing_temp: list[float] = []
    """Temperatures at the annealing points for each group,
    len(`annealing_time`) = len(`annealing_npoints`), in `K`. ([])"""


class GroupVelocityGeneration(BaseModel):
    """Generate velocities parameters."""

    gen_vel: core_mdp.bool_yn | None = None
    """Generate velocities according to a Maxwell distribution at temperature. (False)

    - ACTIVE IF `integrator` = `MD`
    - no velocities in input file => velocities are set to zero
    > Controlled by:
    - `gen_temp` - generated temperature,
    - `gen_seed` - seed to generate velocities"""

    gen_temp: float | None = None
    """Temperature for Maxwell distribution, in `K`. (300)"""

    gen_seed: int | None = None
    """Initialize random generator for random velocities. (-1)

    - `gen_seed` = -1 => random seed"""
