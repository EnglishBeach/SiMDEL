"""Molecular dynamic parameters for Non equilibrium MD simulations, Electric Fields,
Computational Electrophysiology, Density Guided Simulations.
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

import pydantic

from . import core_mdp


class GroupNonEquilibriumMD(pydantic.BaseModel):
    """Using acceleration to atom groups."""

    acc_grps: list[str] = []
    """Groups for constant acceleration. Kinetic energy of the center of mass of
    accelerated groups contributes to the kinetic energy and temperature of the system. ()

    > Controlled by:
    - `accelerate` - acceleration constants groups"""

    accelerate: list[float] = []
    """Acceleration vector: x, y, z for each group, in `nm/ps2`. (0)"""

    freezegrps: list[str] = []
    """Groups that are to be frozen. Virial and pressure are usually
    not meaningful when frozen atoms are present. ()

    > Controlled by:
    `freezedim` - dimension(s) the freezing applies"""

    # TODO: defaults
    freezedim: list[core_mdp.bool_YN] = []
    """Dimensions for which groups in `freezegrps` should be frozen,
    specify Y or N for X, Y and Z and for each group. ()"""

    cos_acceleration: float | None = None
    """The amplitude of the acceleration profile for calculating the viscosity, in `nm/ps2`. (0)"""

    deform: list[float] = []
    """The velocities of deformation for the box elements: a(x) b(y) c(z) b(x) c(x) c(y).
    In `nm/ps`. (0 0 0 0 0 0)"""

    # TODO: default
    deform_init_flow: core_mdp.bool_yn | None = None
    """Add a velocity profile corresponding to the box deformation to the initial velocities

    - ACTIVE IF `deform` is on"""


class GroupElectricFields(pydantic.BaseModel):
    """Modifying forcefield."""

    electric_field_x: str | None = None
    """Specify an electric field that optionally can be alternati/pulsed, for x coordinate."""

    electric_field_y: str | None = None
    """Specify an electric field that optionally can be alternating/pulsed, for y coordinate."""

    electric_field_z: str | None = None
    """Specify an electric field that optionally can be alternating/pulsed, for z coordinate."""


class Swapcoords(enum.Enum):
    """Ion/water position exchanges type."""

    NO = "no"
    """Do not enable ion/water position exchanges."""

    X = "X"
    """Allow for ion/water position exchanges along x direction."""

    Y = "Y"
    """Allow for ion/water position exchanges along x direction."""

    Z = "Z"
    """Allow for ion/water position exchanges along x direction."""


class GroupComputationalElectrophysiology(pydantic.BaseModel):
    """Computational Electrophysiology simulation parameters."""

    # TODO: default (may be No)
    swapcoords: Swapcoords | None = None
    """Ion/water position exchanges type."""

    swap_frequency: int | None = None
    """The swap attempt frequency, (time steps the ion counts per compartment)
    are determined and exchanges made if necessary, in `steps`. (1)"""

    split_group0: str | None = None
    """Name of the index group of the membrane-embedded part of channel #0."""

    split_group1: str | None = None
    """Channel #1 defines the position of the other compartment boundary."""

    massw_split0: core_mdp.bool_yn | None = None
    """Defines whether or not mass-weighting is used to calculate the split group center.
    In channel #0. (False)

    - `True` => use the center of mass
    - `False` => use the geometrical center"""

    massw_split1: core_mdp.bool_yn | None = None
    """Defines whether or not mass-weighting is used to calculate the split group center.
    In channel #1. (False)

    - `True` => use the center of mass
    - `False` => use the geometrical center"""

    solvent_group: str | None = None
    """Name of the index group of solvent molecules."""

    coupl_steps: int | None = None
    """Average the number of ions per compartment over these many swap attempt steps.
    In `steps`. (10)"""

    iontypes: int | None = None
    """The number of different ion types to be controlled. (1)"""

    iontype0_name: str | None = None
    """Name of the first ion type."""

    iontype0_in_A: int | None = None
    """Requested number of ions of type 0 in compartment A. (-1)

    - `= -1` => use the number of ions as found in time step 0"""

    iontype0_in_B: int | None = None
    """Requested number of ions of type 0 in compartment B. (-1)

    - `= -1` => use the number of ions as found in time step 0."""

    bulk_offsetA: float | None = None
    """Offset of the first swap layer from the compartment A midplane. (0)

    - `= 0` => ion/water exchanges happen between layers at maximum distance
    (= bulk concentration) to the split group layers"""

    bulk_offsetB: float | None = None
    """Offset of the first swap layer from the compartment B midplane. (0)

    - `= 0` => ion/water exchanges happen between layers at maximum distance (= bulk concentration)
    to the split group layers"""

    threshold: int | None = None
    """Only swap ions if threshold difference to requested count is reached. (1)"""

    cyl0_r: float | None = None
    """Radius of the split cylinder #0, in `nm`. (2)"""

    cyl0_up: float | None = None
    """Upper extension of the split cylinder #0, in `nm`. (1)"""

    cyl0_down: float | None = None
    """Lower extension of the split cylinder #0, in `nm`. (1)"""

    cyl1_r: float | None = None
    """Radius of the split cylinder #1, in `nm`. (2)"""

    cyl1_up: float | None = None
    """Upper extension of the split cylinder #1, in `nm`. (1)"""

    cyl1_down: float | None = None
    """Lower extension of the split cylinder #1, in `nm`. (1)"""


class DensityGuidedSimulationSimilarityMeasure(enum.Enum):
    """Similarity measure between the density that is calculated from
    the atom positions and the reference density.
    """

    InnerProduct = "inner-product"
    """Takes the sum of the product of reference density and simulated density voxel values."""

    RelativeEntropy = "relative-entropy"
    """Uses the negative relative entropy (or Kullback-Leibler divergence) between
    reference density and simulated density as similarity measure.

    > Controlled by:
    - `density<0` => values are ignored"""

    CrossCorrelation = "cross-correlation"
    """Uses the Pearson correlation coefficient between
    reference density and simulated density as similarity measure."""


class DensityGuidedSimulationAtomSpreadingWeight(enum.Enum):
    """Determines the multiplication factor for the Gaussian kernel
    when spreading atoms on the grid.
    """

    Unity = "unity"
    """Every atom in the density fitting group is assigned the same unit factor."""

    Mass = "mass"
    """Atoms contribute to the simulated density proportional to their mass."""

    Charge = "charge"
    """Atoms contribute to the simulated density proportional to their charge."""


class GroupDensityGuidedSimulations(pydantic.BaseModel):
    """Additional forces that are derived from three-dimensional densities parameters."""

    density_guided_simulation_active: core_mdp.bool_yn | None = None
    """Activate density-guided simulations. (False)"""

    density_guided_simulation_group: str | None = None
    """The atoms that are subject to the forces from the density-guided simulation and
    contribute to the simulated density. (protein)"""

    density_guided_simulation_similarity_measure: (
        DensityGuidedSimulationSimilarityMeasure | None
    ) = None
    """Similarity measure between the density that is calculated from
    the atom positions and the reference density. (InnerProduct)"""

    density_guided_simulation_atom_spreading_weight: (
        DensityGuidedSimulationAtomSpreadingWeight | None
    ) = None
    """Determines the multiplication factor for the Gaussian kernel
    when spreading atoms on the grid. (Unity)"""

    density_guided_simulation_force_constant: float | None = None
    """The scaling factor for density_guided simulation forces, in `kJ/mol`. (1e+9)"""

    density_guided_simulation_gaussian_transform_spreading_width: float | None = None
    """The Gaussian RMS width for the spread kernel for the simulated density, in `nm`. (0.2)"""

    density_guided_simulation_gaussian_transform_spreading_range_in_multiples_of_width: (
        float | None
    ) = None
    """The range after which the gaussian is cut off in multiples of
    the Gaussian RMS width described above. (4)"""

    density_guided_simulation_reference_density_filename: str | None = None
    """Reference density file name (reference.mrc)."""

    density_guided_simulation_nst: int | None = None
    """Interval in steps at which the density fitting forces are evaluated and applied.
    In `steps`. (1)"""

    density_guided_simulation_normalize_densities: core_mdp.bool_tf | None = None
    """Normalize the sum of density voxel values to one for
    the reference density as well as the simulated density. (True)"""

    density_guided_simulation_adaptive_force_scaling: core_mdp.bool_tf | None = None
    """Adaptive force scaling type (False)."""

    density_guided_simulation_adaptive_force_scaling_time_constant: float | None = None
    """Couple force constant to increase in similarity with
    reference density with this time constant, in `ps`. (4)"""

    density_guided_simulation_shift_vector: list[float] = []
    """Add this vector to all atoms in the density_guided_simulation_group before
    calculating forces and energies for density_guided_simulations, in `nm`. (0 0 0)"""

    density_guided_simulation_transformation_matrix: list[float] = []
    """Multiply all atoms with this matrix in the density_guided_simulation_group before
    calculating forces and energies for density_guided_simulations.
    For rotation of f the density_guided atom group around the z_axis by a degrees:
    'cos a, _sin a, 0, sin a, cos a, 0,0,0,1'. (1 0 0 0 1 0 0 0 1)"""
