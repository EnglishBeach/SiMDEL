"""Molecular dynamic parameters for AWH, free energy calculations, soft core functions,
dH/dL calculations, expanded ensemble calculations, Simulated Tempering.
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

from pydantic import BaseModel

from . import core_mdp


class AWHPotential(enum.Enum):
    """Type of bias potential used in AWH."""

    Convolved = "convolved"
    """The applied biasing potential is the convolution of the bias function
    and a set of harmonic umbrella potentials.
    - ACTIVE IF free energy labmda uses as AWH reaction coordinate
    > Controlled by:
    `awh1_dim1_force_constant` - force constant of each umbrella"""

    Umbrella = "umbrella"
    """The potential bias is applied by controlling the position of
    an harmonic potential using Monte-Carlo sampling.
    > Controlled by:
    - `awh1_dim1_force_constant` - force constant
    - `awh_nstsample` - frequency os sampling"""


class AWHGrowth(enum.Enum):
    """Growth of maximum convergence rate function type."""

    ExpLinear = "exp-linear"
    """Each bias keeps a reference weight histogram for the coordinate samples,
    magnitude of the bias function and free energy estimate updates.
    The initial stage is typically necessary for efficient convergence when
    starting a new simulation where high free energy
    barriers have not yet been flattened by the bias.

    > Controlled by:
    - `awh_nstsample` - growth rate"""

    Linear = "linear"
    """Only linear stage.
    This may be useful if there is a priori knowledge which
    eliminates the need for an initial stage."""


class AWHTarget(enum.Enum):
    """Target distribution type for AWH."""

    Constant = "constant"
    """The bias is tuned towards a constant (uniform) coordinate distribution in
    the defined sampling interval.

    > Controlled by:
    - `awh1_dimx_start` - interval start
    - `awh1_dimx_end` - interval end"""

    # TODO: error in gromacs docs

    Cutoff = "cutoff"
    """Similar to `Constant`, but the target distribution is proportional to
    1/(1 + exp(F-?), F - free energy relative to the estimated global minimum.

    Smooth switch:
    flat target distribution where free energy < the cut-off ->
    Boltzmann distribution where free energy > the cut-off"""

    Boltzmann = "boltzmann"
    """The target distribution is a Boltzmann distribtution with a scaled beta.

    > Controlled by:
    - `awh1_target_beta_scaling` - beta (inverse temperature) factor"""

    LocalBoltzmann = "local-boltzmann"
    """Similar to `Boltzmann`, but convergence is inherently local.

    - ACTIVE IF `awhx_growth`=`Linear`
    > Controlled by:
    - `awh1_target_beta_scaling` - beta (inverse temperature) factor"""


class AWHShareGroup(enum.Enum):
    """Share AWH calculations type."""

    Zero = "0"
    """Do not share the bias."""

    Positive = "positive"
    """Share the bias and PMF estimates between simulations.

    - ACTIVE IF biases mush have same indexes and  `awh_share_multisim` = True"""


class AWHDimCoordProvider(enum.Enum):
    """Coordinate provider for AWH."""

    Pull = "pull"
    """The pull module is providing the reaction coordinate for this dimension,
    AWH and pull must have same `mts_level`."""

    FEPLambda = "fep-lambda"
    """The free energy lambda state is the reaction coordinate for this dimension.

    - ACTIVE IF `delta_lambda` is off and `awh_potential` = `Umbrella`"""


class GroupAWH(BaseModel):
    """Adaptive biasing parameters
    Parameters for accelerated weight histogram (AWH) method used for
    free energy calculations and enhanced sampling.
    """

    awh: core_mdp.bool_yn | None = None
    """Enable AWH accelerated sampling. (False)

    - ACTIVE IF `ensemble_temperature_setting` = `Constant`
    and `pull_coord_geometry` != `DirectionPeriodic` or 'Transformation`
    for external:
    - `pull_coord1_type`=`external_potential` and `pull_coord_potential_provider` = `awh`"""

    # TODO: default
    awh_potential: list[AWHPotential] = []
    """Type of bias potential for each bias."""

    awh_share_multisim: core_mdp.bool_yn | None = None
    """Share share biases across simulations. (False)

    - ACTIVE IF `awh1_share_group` > 0 and gmx mdrun -multidir
    and simulations have same AWH settings"""

    awh_seed: int | None = None
    """Random seed for Monte-Carlo sampling the umbrella position. (-1)

    - ACTIVE IF `awh_potential` = `Umbrella`
    - `= -1` => random"""

    awh_nstout: int | None = None
    """Frequency for writing AWH data to energy file, in `steps`. (100000)

    - `!= 0` => multiply `nstenergy`"""

    awh_nstsample: int | None = None
    """Frequency for sampling coordinate values, in `steps`. (10)"""

    awh_nsamples_update: int | None = None
    """Number of samples per update of the AWH energy. (10)"""

    awh_nbias: list[int] = []
    """The number of biases, each acting on its own coordinate. (1)"""

    awh1_error_init: list[float] = []
    """Estimated initial average error of the PMF for this bias,
    determine the initial biasing rate, in `kJ/mol`. (10)"""

    # TODO: default
    awh1_growth: AWHGrowth | None = None
    """Growth of maximum convergence rate function type."""

    awh1_growth_factor: list[float] = []
    """The growth factor during the exponential phase.

    - ACTIVE IF with `awhx_growth`=`ExpLinear` and > 0 ([2])"""

    # TODO: default
    awh1_equilibrate_histogram: core_mdp.bool_yn | None = None
    """Before entering the initial stage, make sure the histogram of
    sampled weights is following the target distribution closely enough.

    - ACTIVE IF with `awhx_growth`=`ExpLinear`"""

    awh1_target: list[AWHTarget] = []
    """Target distribution type for each bias."""

    awh1_target_beta_scaling: float | None = None
    """Beta scaling factor. (0)

    - ACTIVE IF `awhx_target` =`Boltzmann`/`LocalBoltzmann`"""

    awh1_target_cutoff: float | None = None
    """Cut-off for `Cutoff` awh target, in `kJ/mol`. (0)

    - ACTIVE IF`awhx_target` = `Cutoff`"""

    awh1_user_data: core_mdp.bool_yn | None = None
    """Initialize the PMF and target distribution with user provided data.
    `awhinit.xvg` (awhinit1.xvg for multiple biases)- user data file."""

    # TODO: default
    awh1_share_group: AWHShareGroup | None = None
    """Share AWH calculations type."""

    awh1_target_metric_scaling: core_mdp.bool_yn | None = None
    """Scale the target distribution based on the AWH friction metric. (False)"""

    awh1_target_metric_scaling_limit: float | None = None
    """The upper limit of scaling, relative to the average, should be > 1. (10)

    - ACTIVE IF `awh1_target_metric_scaling` is on"""

    awh1_ndim: int | None = None
    """Number of dimensions of the coordinate, each dimension maps to 1 pull coordinate. (1)"""

    awh1_dim1_coord_provider: AWHDimCoordProvider | None = None
    """Coordinate provider for AWH."""

    awh1_dim1_coord_index: int | None = None
    """Index of the pull coordinate defining this coordinate dimension. (-1)"""

    # TODO: not all potentials
    awh_dim1_force_constant: list[float] = []
    """Force constant for the (convolved) umbrella potential(s) along this coordinate dimension.
    In `kJ/(mol*nm^2)` or `kJ/(mol*rad2)`. ([])"""

    awh_dim1_start: list[float] = []
    """Start value of the sampling interval along this dimension, in `nm` or `deg`. ([0])

    > Controlled by:
    - `pull_coord1_geometry` - range of allowed value"""

    awh_dim1_end: list[float] = []
    """End value defining the sampling interval, in `nm` or `deg`. ([0])"""

    awh_dim1_diffusion: list[float] = []
    """Estimated diffusion constant for this coordinate dimension
    determining the initial biasing rate, in `nm2/ps` or `rad2/ps` or `1/ps`. ([1e-5])"""

    # TODO: sampling interval length to var
    awh_dim1_cover_diameter: list[float] = []
    """Diameter that needs to be sampled by a single simulation around
    a coordinate value before the point is considered covered in the initial stage.
    In `nm` or `deg`. ([0])

    - `> 0` => or each covering there is a continuous transition of this diameter across
    each coordinate value
    - `= 0` => covering occurs as soon as the simulations have sampled the whole interval,
    not guarantee transitions across free energy barriers,
    - `> sampling interval length` => covering occurs when a single simulation has
    independently sampled the whole interval"""


class FreeEnergy(enum.Enum):
    """Free energy colculations type."""

    NO = "no"
    """Only use topology A."""

    YES = "yes"
    """Interpolate between topology A (lambda=0) to topology B (lambda=1).

    > Controlled by:
    - `dhdl_derivatives` - write d(Hamiltonian)/d(lambda)
    - `calc_lambda_neighbors` - write d(Hamiltonian)/d(lambda)
    - `sc-alpha` > 0 - soft-core potentials are used for the LJ and Coulomb interactions"""

    Expanded = "Expanded"
    """Alchemical state becomes a dynamic variable,
    allowing jumping between different Hamiltonians."""


class GroupFreeEnergy(BaseModel):
    """Free energy parameters."""

    free_energy: FreeEnergy | None = None
    """Interpolate between topology A - lambda=0 to topology B - lambda=1. (NO)

    - False => Only use topology A"""

    init_lambda: float | None = None
    """Starting value for lambda, range=[0,1]. (-1)

    - ACTIVE IF only for slow growth - `delta-lambda` != 0
    - active => lambda = `init_lambda` + `delta_lambda` * (`init_step` + i)
    - `delta-lambda = 0` => init_lambda should be off"""

    delta_lambda: float | None = None
    """Increment per time step for lambda (0)"""

    init_lambda_state: int | None = None
    """Specifies which column of the lambda vector should be used:
    `coul_lambdas` - 0, `vdw_lambdas` - 1, `bonded_lambdas` - ..., `restraint_lambdas`,
    `mass_lambdas`, `temperature_lambdas`, `fep_lambdas`. (-1)"""

    fep_lambdas: list[float] = []
    """Zero, one or more lambda values for which Delta H values will be
    determined and written, range=[0,1]. ([])

    > Controlled by:
    - `nstdhdl` - `steps` to write Delta H to dhdl.xvg"""

    coul_lambdas: list[float] = []
    """Zero, one or more lambda values for which Delta H values will be
    determined and written, for electrostatic interactions. ([])

    > Controlled by:
    - `nstdhdl` - `steps` to write Delta H to dhdl.xvg"""

    vdw_lambdas: list[float] = []
    """Zero, one or more lambda values for which Delta H values will be
    determined and written, for the van der Waals interactions. ([])

    > Controlled by:
    - `nstdhdl` - `steps` to write Delta H to dhdl.xvg"""

    bonded_lambdas: list[float] = []
    """Zero, one or more lambda values for which Delta H values will be
    determined and written, for bonded interactions. ([])

    > Controlled by:
    - `nstdhdl` - `steps` to write Delta H to dhdl.xvg"""

    restraint_lambdas: list[float] = []
    """Zero, one or more lambda values for which Delta H values will be
    determined and written, for restraint interactions. ([])

    > Controlled by:
    - `nstdhdl` - `steps` to write Delta H to dhdl.xvg"""

    mass_lambdas: list[float] = []
    """Zero, one or more lambda values for which Delta H values will be
    determined and written, for particle masses. ([])

    > Controlled by:
    - `nstdhdl` - `steps` to write Delta H to dhdl.xvg"""

    temperature_lambdas: list[float] = []
    """Zero, one or more lambda values for which Delta H values will be
    determined and written, for temperatures, only for simulated tempering.

    > Controlled by:
    - `nstdhdl` - `steps` to write Delta H to dhdl.xvg"""

    calc_lambda_neighbors: int | None = None
    """Controls the number of lambda values for which DH values will be
    calculated and written out. (1)

    - ACTIVE IF `init_lambda_state`
    - `> 0` => limit the number of lambda points calculated to only
    the nth neighbors of `init_lambda_state`
    - `= -1` => all lambda points will be written out"""


class ScFunction(enum.Enum):
    """Soft-core function type."""

    Beutler = "beutler"
    """Beutler soft-core function."""

    Gapsys = "gapsys"
    """Gapsys soft-core function."""


class GroupSoftFunction(BaseModel):
    """Soft-core function parameters."""

    sc_function: ScFunction | None = None
    """Soft-core function type. (Beutler)"""

    sc_alpha: float | None = None
    """Soft-core alpha parameter. (0)

    - ACTIVE IF `sc_function` = `Beutler`
    - `= 0` => linear interpolation of the LJ and Coulomb interactions"""

    sc_r_power: int | None = None
    """Power 6 for the radial term in the soft-core equation. (6)

    - ACTIVE IF `sc_function` = `Beutler`"""

    sc_coul: core_mdp.bool_yn | None = None
    """Apply the soft-core free energy interaction transformation to
    the Columbic interaction of a molecule. (False)

    - ACTIVE IF multiple lambda components and `sc_function` = `Beutler`
    - False => no apply soft core interactions
    - `sc-alpha = 0` => turn off soft core interactions"""

    sc_power: int | None = None
    """Power for lambda in the soft-core function, only 1, 2. (1)

    - ACTIVE IF `sc_function` = `Beutler`"""

    sc_sigma: float | None = None
    """Soft-core sigma for particles which have a C6 or C12 parameter, in `nm` (0.3)"""

    sc_gapsys_scale_linpoint_lj: float | None = None
    """Parameter alphaLJ, softness of the van der Waals interactions by
    scaling the point for linearizing the vdw force. (0.85)

    - ACTIVE IF `sc_function` = `Gapsys`
    - `= 0` => standard hard-core Van der Waals interactions"""

    sc_gapsys_scale_linpoint_q: float | None = None
    """Parameters alphaQ parameter of the softness of the Coulombic interactions.
    In `nm/e2`. (0.3)

    - ACTIVE IF `sc_function` = `Gapsys`"""

    sc_gapsys_sigma_lj: float | None = None
    """Soft-core sigma for particles which have a C6 or C12 parameter equal to zero. (0.3)

    - ACTIVE IF `sc_function` = `Gapsys`"""


class CoupleLambda(enum.Enum):
    """VDW and Coulomb interactions switch."""

    VDWQ = "vdw-q"
    """All interactions are on at lambda=0."""

    VDW = "vdw"
    """Coulomb is off (charges = zero) at lambda=0."""

    Q = "q"
    """VDW is at lambda=0.

    - ACTIVE IF use soft-core interactions"""

    NONE = "None"
    """VDW and Coulomb are off at lambda=0.

    - ACTIVE IF use soft-core interactions"""


class GroupFreeEnergyCouple(BaseModel):
    """Coupling for free energy calculation parameters."""

    couple_moltype: str | None = None
    """Supply a molecule type (as defined in the topology) for
    calculating solvation or coupling free energies.

    - ACTIVE IF `free_energy` is on and VDW/charges in this molecule type is on/off
    > Controlled by:
    - `couple_lambda0`
    - `couple_lambda1`"""

    # TODO: defaults
    couple_lambda0: CoupleLambda | None = None
    """VDW and Coulomb interactions switch for lambda=0 (VDWQ)."""

    # TODO: defaults
    couple_lambda1: CoupleLambda | None = None
    """VDW and Coulomb interactions switch for lambda=1 (VDWQ)."""

    couple_intramol: core_mdp.bool_yn | None = None
    """Intra-molecular intra-molecular VDW and Coulomb interactions are also turned on/off.
    Useful for partitioning free-energies of relatively large molecules.
    The 1-4 pair interactions are not turned off. (False)

    - False => All intra-molecular non-bonded interactions for
    moleculetype `couple_moltype` are replaced by exclusions and explicit pair interactions.
    - In this manner the decoupled state of the molecule corresponds to
    the proper vacuum state without periodicity effects."""


class DHDLPrintEnergy(enum.Enum):
    """Include either the total or the potential energy in the dhdl file."""

    NO = "no"
    """No write."""

    Potential = "potential"
    """Useful in case one is using mdrun -rerun to generate the dhdl.xvg."""

    Total = "total"
    """Write total energy in dhdl file."""


class GroupDHDL(BaseModel):
    """Writing dH/dl parameters."""

    nstdhdl: int | None = None
    """Interval for writing dH/dlambda and possibly Delta H to dhdl.xvg,
    should be a multiple of `nstcalcenergy`. (100)"""

    dhdl_derivatives: core_mdp.bool_yn | None = None
    """Write derivatives of the Hamiltonian with respect to lambda. (True)

    > Controlled by:
    `nstdhdl` - frequency for DHDL write"""

    # TODO: default
    dhdl_print_energy: DHDLPrintEnergy | None = None
    """Include either the total or the potential energy in the dhdl file."""

    # TODO: default
    separate_dhdl_file: core_mdp.bool_yn | None = None
    """The free energy values that are calculated are written out to a separate file
    with the default name dhdl.xvg

    - False => the free energy values are written out to the energy output file"""

    dh_hist_size: int | None = None
    """Specifies the size of the histogram into which the Delta H values
    and the derivative dH/dl values are binned, and written to ener.edr. (0)

    - `= 0` => no write
    > Controlled by:
    - `nstenergy` - frequency for histogram writing"""

    # TODO: energy units from gromcas docs is kJ?
    dh_hist_spacing: float | None = None
    """Specifies the bin width of the histograms.
    This size limits the accuracy with which free energies can be calculated, in `kJ`. (0.1)"""


class LMCStats(enum.Enum):
    """Sample weight method."""

    NO = "no"
    """No Monte Carlo in state space is performed."""

    MetropolisTransition = "metropolis-transition"
    """Uses the Metropolis weights to update the expanded ensemble weight of each state.
    Min(1,exp(-(beta_new*u_new - beta_old*u_old))."""

    BarkerTransition = "barker-transition"
    """Uses the Barker transition critera to update the expanded ensemble weight of each state i,
    defined by exp(-beta_new*u_new)/(exp(-beta_new*u_new)+exp(-beta_old*u_old))."""

    WangLandau = "wang-landau"
    """Uses the Wang-Landau algorithm (in state space, not energy space) to
    update the expanded ensemble weights."""

    MinVariance = "min-variance"
    """Uses the minimum variance updating method of Escobedo to
    update the expanded ensemble weights.
    Weights will not be the free energies, but will rather emphasize states
    that need more sampling to give even uncertainty."""


class LMCMCMove(enum.Enum):
    """Sample states method."""

    NO = "no"
    """No Monte Carlo in state space is performed."""

    MetropolisTransition = "metropolis-transition"
    """Randomly chooses a new state up or down, uses the Metropolis criteria to
    decide whether to accept or reject: Min(1,exp(-(beta_new*u_new - beta_old*u_old))."""

    BarkerTransition = "barker-transition"
    """Randomly chooses a new state up or down,  uses the Barker transition criteria to
    decide whether to accept or reject:
    exp(-beta_new*u_new)/(exp(-beta_new*u_new)+exp(-beta_old*u_old))."""

    Gibbs = "gibbs"
    """Uses the conditional weights of the state given the coordinate
    (exp(-beta_i*u_i) / sum_k * exp(beta_i*u_i) to decide which state to move to."""

    MetropolizedGibbs = "metropolized-gibbs"
    """Uses the conditional weights of the state given the coordinate
    (exp(-beta_i*u_i) / sum_k * exp(beta_i*u_i) to decide which state to move to,
    EXCLUDING the current state, then uses a rejection step to ensure detailed balance.
    Always more efficient that Gibbs, though only marginally so in many situations,
    such as when only the nearest neighbors have decent phase space overlap."""


class LMXWeightsEquil(enum.Enum):
    """Equilibration ensemble weights type."""

    NO = "no"
    """Expanded ensemble weights continue to be updated throughout the simulation."""

    YES = "yes"
    """The input expanded ensemble weights are treated as equilibrated,
    and are not updated throughout the simulation."""

    WLDelta = "wl-delta"
    """Expanded ensemble weight updating is stopped when:
    the Wang-Landau incrementor falls below this value."""

    NumberAllLambda = "number-all-lambda"
    """Expanded ensemble weight updating is stopped when:
    number of samples at all of the lambda states > this value."""

    NumberSteps = "number-steps"
    """Expanded ensemble weight updating is stopped when:
    number of steps is greater > the level specified by this value."""

    NumberSamples = "number-samples"
    """Expanded ensemble weight updating is stopped when:
    number of total samples across all lambda states is > level specified by this value."""

    CountRatio = "count-ratio"
    """Expanded ensemble weight updating is stopped when:
    ratio of samples at the least sampled lambda state and
    most sampled lambda state > this value."""


class GroupExpandedEnsembleCalculations(BaseModel):
    """Coordinates and the thermodynamic ensemble are treated as configuration variables
    that can be sampled over.
    """

    # TODO: default
    nstexpanded: int | None = None
    """The number of integration steps beween attempted moves changing
    the system Hamiltonian in expanded ensemble simulations.  In `steps`. ()

    - active => `nstexpanded` = `nstcalcenergy`"""

    # TODO: default
    lmc_stats: LMCStats | None = None
    """Sample weight method. ()"""

    # TODO: default
    lmc_mc_move: LMCMCMove | None = None
    """Sample states method. ()"""

    # TODO: only for monte carlo?
    lmc_seed: int | None = None
    """Random seed to use for Monte Carlo moves in state space. (-1)

    - `= -1` => random"""

    # TODO: default
    # TODO: only for monte carlo?
    mc_temperature: float | None = None
    """Temperature used for acceptance/rejection for Monte Carlo moves.

    > Controlled by:
    - `ref_t` - temperature if `mc_temperature` is off"""

    wl_ratio: float | None = None
    """The cutoff for the histogram of state occupancies to be reset,
    free energy incrementor to be changed from delta to delta * `wl_scale`. (0.8)"""

    wl_scale: float | None = None
    """Each time the histogram is considered flat, range = [0,1]. (0.8)

    > Controlled by:
    - `wl_scale` - multiply Wang-Landau incrementor for the free energies"""

    init_wl_delta: float | None = None
    """The initial value of the Wang-Landau incrementor in kT. (1)"""

    wl_oneovert: core_mdp.bool_yn | None = None
    """Incrementor becomes less than 1/N, where N is the number of samples collected,
    when the Wang-Lambda incrementor is set to 1/N, decreasing every step and
    `wl_ratio` is ignored, but the weights will still stop updating
    when the equilibration criteria set in `lmc_weights_equil` is achieved. (False)"""

    lmc_repeats: int | None = None
    """Controls the number of times that each Monte Carlo swap type is performed each iteration,
    not need to change. (1)"""

    lmc_gibbsdelta: int | None = None
    """Limit Gibbs sampling to selected numbers of neighboring states.

    - `> 0` => `lmc_gibbsdelta` means that only states plus or minus `lmc_gibbsdelta`
    are considered in exchanges up and down
    - `= -1` => all states are considered"""

    lmc_forced_nstart: int | None = None
    """Force initial state space sampling to generate weights.
    If lmc-forced-nstart is sufficiently long, then the weights will be close to correct.
    In most cases, it is probably better to simply run the
    standard weight equilibration algorithms. (0)"""

    nst_transition_matrix: int | None = None
    """Frequency of outputting the expanded ensemble transition matrix.

    - `= -1` => print only the end of the simulation (-1)"""

    symmetrized_transition_matrix: core_mdp.bool_yn | None = None
    """Whether to symmetrize the empirical transition matrix.
    In the infinite limit the matrix will be symmetric,
    but will diverge with statistical noise for short timescale. (False)"""

    mininum_var_min: int | None = None
    """Minimum number of samples that each state,
    (lmc-stats for larger number of samples). (100)"""

    init_lambda_weights: list[float] = []
    """The initial weights used for the expanded ensemble states,
    length must match the lambda vector lengths, in `kT`. (zeros vector)"""

    # TODO: default
    lmc_weights_equil: LMXWeightsEquil | None = None
    """Equilibration ensemble weights type."""


class SimulatedTemperingScaling(enum.Enum):
    """Controls the way that the temperatures at intermediate lambdas are calculated from
    the temperature-lambdas part of the lambda vector.
    """

    Linear = "linear"
    """Linearly interpolates the temperatures using the values of temperature-lambdas.
    A nonlinear set of temperatures can always be implemented with uneven spacing in lambda."""

    Geometric = "geometric"
    """Interpolates temperatures geometrically between `sim_temp_low` and `sim_temp_high`.
    This should give roughly equal exchange for constant heat capacity."""

    Exponential = "exponential"
    """Interpolates temperatures exponentially between sim-temp-low and sim-temp-high."""


class GroupSimulatedTempering(BaseModel):
    """Simulated tempering parameters groups."""

    simulated_tempering: core_mdp.bool_yn | None = None
    """Turn on simulated tempering is implemented as expanded ensemble sampling with
    different temperatures instead of different Hamiltonians. (False)"""

    sim_temp_low: float | None = None
    """Low temperature for simulated tempering, in `K`. (300)"""

    sim_temp_high: float | None = None
    """High temperature for simulated tempering, in `K`. (300)"""

    simulated_tempering_scaling: SimulatedTemperingScaling | None = None
    """Tempering type between lambdas. (Geometric)"""
