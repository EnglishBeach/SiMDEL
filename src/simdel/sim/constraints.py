"""Molecular dynamic parameters for Bonds, Walls, COM, Enforced rotation.
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

from pydantic import BaseModel

from . import core_mdp


class Constraints(enum.Enum):
    """Constraints type, which bonds in the topology
    will be converted to rigid holonomic constraints,
    [ settels section ] not affected by this keyword.
    """

    NONE = "none"
    """No constraints except for those defined explicitly in the topology."""

    HBonds = "h-bonds"
    """Convert the bonds with H-atoms to constraints."""

    AllBonds = "all-bonds"
    """Convert all bonds to constraints."""

    HAngles = "h-angles"
    """Convert all bonds to constraints and convert the angles that
    involve H-atoms to bond-constraints."""

    AllAngles = "all-angles"
    """Convert all bonds to constraints and all angles to bond-constraints."""


class ConstraintAlgorithm(enum.Enum):
    """Method to calculate constraints."""

    LINCS = "LINCS"
    """LINear Constraint Solver, with domain decomposition
    the parallel version P-LINCS is used.

    - ACTIVE IF no coupled angle constraints
    - `bond rotates > lincs_warnangle` in one step => warning
    > Controlled by:
    - `lincs_order` - accuracy
    - `lincs_iter` - iterations number of correction to compensate for
    lengthening due to rotation
    - `nstlog` - root mean square relative constraint deviation is printed to the log"""

    SHAKE = "SHAKE"
    """SHAKE is slightly slower and less stable than LINCS.

    - ACTIVE IF no angle constraints and no EM
    > Controlled by:
     - `shake_tol` - relative tolerance"""


class GroupBonds(BaseModel):
    """Constraints parameters."""

    constraints: Constraints | None = None
    """Constraints type (NONE)."""

    constraint_algorithm: ConstraintAlgorithm | None = None
    """Method to calculate constraints (LINKS)."""

    continuation: core_mdp.bool_yn | None = None
    """Do not apply constraints to the start configuration and do not reset shells. (False)

    - `False` => apply constraints to the start configuration and reset shells."""

    shake_tol: float | None = None
    """Relative tolerance for SHAKE (0.0001)."""

    lincs_order: int | None = None
    """Highest order in the expansion of the constraint coupling matrix. (4)

    - `= 4` => normal MD
    - `= 6` => large time-steps with virtual sites or BD
    - `>= 8` => accurate EM"""

    lincs_iter: int | None = None
    """Number of iterations to correct for rotational lengthening in LINCS. (1)"""

    lincs_warnangle: float | None = None
    """Maximum angle that a bond can rotate before LINCS will complain. (30)"""

    morse: core_mdp.bool_yn | None = None
    """Bonds are represented by a Morse potential. (False)

    - False => bonds are represented by a harmonic potential"""


class WallType(enum.Enum):
    """Integrating potential behind the wall type."""

    W93 = "9-3"
    """LJ integrated over the volume behind the wall: 9-3 potential."""

    W104 = "10-4"
    """LJ integrated over the wall surface: 10-4 potential."""

    W126 = "12-6"
    """Direct LJ potential with the z distance from the wall."""


class GroupWalls(BaseModel):
    """Walls set parameters."""

    nwall: int | None = None
    """Wall positions type. (0)

    - ACTIVE IF `pbc` = `xy`
    - `active` => `comm_mode` = `no` in z direction
    - `= 1` => wall at z=0
    - `= 2` => wall at z = z-box, can use `p_coupl`, `Ewald`"""

    wall_atomtype: str | None = None
    """The atom type name in the force field for each wall.
    Allows for independent tuning of the interaction of each atomtype with the walls

    - ACTIVE IF special wall atom type in the topology is set"""

    # TODO: default
    wall_type: WallType | None = None
    """Integrating potential behind the wall type."""

    table: str | None = None
    """User defined potentials indexed with the z distance from the wall."""

    wall_r_linpot: float | None = None
    """Below this distance from wall the potential is continued linearly, the force is constant.
    In `nm`. (-1)

    - > 0 => useful for equilibration when some atoms are beyond a wall
    - <= 0 => error"""

    # TODO: default
    wall_density: float | None = None
    """The number density of the atoms for each wall, in `nm^-3/nm^-2`.

    - ACTIVE IF `wall_atomtype` = 'W93` or `W104`"""

    wall_ewald_zfac: float | None = None
    """The scaling factor for the third box vector for Ewald.The empty layer in the box serves
    to decrease the unphysical Coulomb interaction between periodic images (3)
    - ACTIVE IF `Ewald` and `nwall` = 2 and `ewald_geometry` = 'D3c`."""


class PullCoordType(enum.Enum):
    """Center of mass pulling type."""

    Umbrella = "umbrella"
    """Center of mass pulling using an umbrella potential between
    the reference group and one or more groups."""

    Constraint = "constraint"
    """Center of mass pulling using a constraint between
    the reference group and one or more groups,
    a rigid constraint is applied instead of a harmonic potential.

    - ACTIVE IF `mts` is off"""

    ConstantForce = "constant-force"
    """Center of mass pulling using a linear potential and therefore a constant force.

    - active => `pull_coordx_init`, `pull_coordx_rate` ignored"""

    FlatBottom = "flat-bottom"
    """At distances above `pull_coordx_init` a harmonic potential is applied,
    otherwise no potential is applied."""

    FlatBottomHigh = "flat-bottom-high"
    """At distances below `pull_coord1_init` a harmonic potential is applied,
    otherwise no potential is applied."""

    ExternalPotential = "external-potential"
    """An external potential that needs to be provided by another module."""


class PullCoordGeometry(enum.Enum):
    """Geometry type to pull COM."""

    Distance = "distance"
    """Pull along the vector connecting the two groups.

    > Controlled by:
    - `pull_coord1_dim` - group components"""

    Direction = "direction"
    """Pull in the direction.

    > Controlled by:
    - `pull_coord1_vec` - direction"""

    DirectionPeriodic = "direction-periodic"
    """Like `Direction`, but does not apply periodic box vector corrections to keep
    the distance within half the box length.

    - ACTIVE IF box geometry is fixed"""

    DirectionRelative = "direction-relative"
    """Like `Direction`, but the pull vector is the vector that points from
    the COM of a third to the COM of a fourth pull group.

    => 4 groups for `pull_coord1_groups`."""

    Cylinder = "cylinder"
    """Designed for pulling with respect to a layer where
    the reference COM is given by a local cylindrical part of the reference group.

    - ACTIVE IF constnt pullig is off
    - active => `pull_cylinder_r` < box_size/2
    > Controlled by:
    - `pull_coord1_vec` - direction
    - first from `pull_coord1_groups` - selecting cylinder
    - `pull_cylinder_r` - cylinder radius"""

    Angle = "angle"
    """Pull along an angle defined by four groups.

    - ACTIVE IF 4 COM groups set"""

    AngleAxis = "angle-axis"
    """Like `Angle`, but the second vector is given by `pull_coord1_vec`.

    - ACTIVE IF 2 COM groups set"""

    Dihedral = "dihedral"
    """Pull along a dihedral angle defined by six groups.

    - ACTIVE IF 6 COM groups and 3 vectors: 1-2 connecting between COM groups, 3-4, 5-6"""

    Transformation = "transformation"
    """Transforms other pull coordinates using a mathematical expression
    pull index must be > other pull indexes.

    > Controlled by:
    `pull_coord1_expression` - mathematical expression"""


class GroupCOMPulling(BaseModel):
    """Center of mass pulling parameters."""

    pull: core_mdp.bool_yn | None = None
    """Center of mass pulling will be applied on 1 or more groups
    using 1 or more pull coordinates. (False)"""

    pull_cylinder_r: float | None = None
    """Radius of the cylinder, in `nm`. (1.5)

    - ACTIVE IF `pull_coord1_geometry` = `Cylinder`"""

    pull_constr_tol: float | None = None
    """Relative constraint tolerance for constraint pulling. (1e-6)"""

    pull_print_com: core_mdp.bool_yn | None = None
    """Print the COM of all groups for all pull coordinates. (False)"""

    pull_print_ref_value: core_mdp.bool_yn | None = None
    """Print the reference value for each pull coordinate. (False)"""

    pull_print_components: core_mdp.bool_yn | None = None
    """Print the distance and Cartesian components.

    > Controlled by:
    `pull_coord1_dim` - components to print"""

    pull_nstxout: int | None = None
    """Frequency for writing out the COMs of all the pull group, in `steps`. (50)"""

    pull_nstfout: int | None = None
    """Frequency for writing out the force of all the pulled group, in `steps`. (50)"""

    pull_pbc_ref_prev_step_com: core_mdp.bool_yn | None = None
    """Use the COM of the previous step as reference for the treatment of
    periodic boundary conditions.

    > Controlled by:
    - `pull_group1_pbcatom` - reference atom to init reference which should be
    located centrally in the group"""

    pull_xout_average: core_mdp.bool_yn | None = None
    """Write the average coordinates (since last output) for all the pulled groups. (False)

    - False => write the instantaneous coordinates for all the pulled groups
    > Controlled by:
    - `pull_ngroups` - groups"""

    pull_fout_average: core_mdp.bool_yn | None = None
    """Write the average force (since last output) for all the pulled groups. (False)

    - False => write the instantaneous forces for all the pulled groups
    > Controlled by:
    `pull_ngroups` - groups"""

    pull_ngroups: int | None = None
    """The number of pull groups, not including the absolute reference group. (1)"""

    pull_ncoords: int | None = None
    """The number of pull coordinates. (1)"""

    # TODO: in mdp pull_group1_name, pull_group2_name...
    pull_group1_name: str | None = None
    """The name of the pull group, is looked up in the index file or in
    the default groups to obtain the atoms involved."""

    # TODO: misunderstanding desc
    pull_group1_weights: int | None = None
    """Relative weights which are multiplied with
    the masses of the atoms to give the total weight for the COM. ()

    - `= 0` => all 1
    - `= number of atoms`
    - `else` => error"""

    pull_group1_pbcatom: int | None = None
    """The reference atom for the treatment of periodic boundary conditions inside the group.
    This option is only important when the diameter of the pull group is larger
    than half the shortest box vector.
    For determining the COM, all atoms in the group are put at their periodic image
    which is closest to `pull_group1_pbcatom`.

    - ACTIVE IF `pull_coord1_geometry` is off
    - 0 => middle atom, only safe for small groups"""

    pull_coord1_type: PullCoordType | None = None
    """Center of mass pulling type. (Umbrella)"""

    pull_coord1_potential_provider: str | None = None
    """The name of the external module that provides the potential. ()

    - ACTIVE IF `pull_coord1_type` is external-potential"""

    # TODO: default
    pull_coord1_geometry: PullCoordGeometry | None = None
    """Geometry type to pull COM."""

    # TODO: default
    pull_coord1_expression: str | None = None
    """Mathematical expression to transform pull coordinates of lower indices to a new one.
    pull-coord1 -> x1, pull-coord2 -> x2, time -> t, angles in radiants.

    - ACTIVE IF `pull_coord1_geometry` = `Transformation`"""

    pull_coord1_dx: float | None = None
    """Size of finite difference to use in numerical derivation
    of the pull coordinate with respect to other pull coordinates. (1e-9)

    - ACTIVE IF `pull_coord1_geometry` = `Transformation`"""

    pull_coord1_groups: list[int] = []
    """The group indices on which this pull coordinate will operate.

    - `first index = 0` => `pull_coord1_origin` is active, system is no longer translation invariant
    > Controlled by:
    - `pull_coord1_geometry` - set n groups"""

    pull_coord1_dim: list[core_mdp.bool_YN] = []
    """Selects the dimensions that this pull coordinate acts on,
    Y Y N results in a distance in the x/y plane. (Y Y Y)

    - `pull_print_components = pull_coord1_start = True` =>  write to the output files
    - `pull_coord1_geometry = pull_coord1_geometry = Distance` => write cartesian
    components set to Y contribute to the distance
    - `pull_coord1_geometry != Distance` => all dimensions with non-zero entries
    in `pull_coord1_vec` should be = Y"""

    pull_coord1_origin: list[float] = []
    """The pull reference position for use with an absolute reference. (0 0 0)"""

    pull_coord1_vec: list[float] = []
    """The pull direction. (0 0 0)"""

    # TODO: default
    pull_coord1_start: core_mdp.bool_yn | None = None
    """Add the COM distance of the starting conformation to `pull_coord1_init`"""

    pull_coord1_init: float | None = None
    """The reference distance or reference angle at t=0, in `nm` or `deg`. (0)"""

    pull_coord1_rate: float | None = None
    """The rate of change of the reference position or reference angle, in `nm/ps` or `deg/ps`."""

    pull_coord1_k: float | None = None
    """The force constant.`pull_coord1_type`.
    In `kJ/(mol*nm2)` or `kJ/(mol*rad2)` or `kJ/(mol*nm)` or `kJ/(mol*rad)`. (0)

    - `Umbrela` => harmonic force constant
    - 'Constant` => force constant of the linear potential (negative force constant)"""

    pull_coord1_kB: float | None = None
    """As `pull_coord1_k` to state B.

    - ACTIVE IF `free_energy` is on"""


# TODO: desc
class RotType(enum.Enum):
    """Type of rotation group."""

    Iso = "iso"
    """Isotropic rotation."""

    IsoPF = "iso-pf"
    """Isotropic rotation with potential-free."""

    PM = "pm"
    """Potential-mean rotation."""

    PMPF = "pm-pf"
    """Potential-mean rotation with potential-free."""

    RM = "rm"
    """Potential-mean rotation with radius."""

    RMPF = "rm-pf"
    """Potential-mean rotation with potential-free and radius."""

    RM2 = "rm2"

    RM2PF = "rm2-pf"

    Flex = "flex"
    """Flexible axis rotation."""

    FlexTarget = "flex-t"
    """Flexible axis rotation with target."""

    Flex2 = "flex2"

    Flex2Target = "flex2-t"


class RotFitMethod(enum.Enum):
    """Fitting method when determining the actual angle of a rotation group."""

    RMSD = "rmsd"
    Norm = "norm"
    Potential = "potential"


class GroupEnforcedRotation(BaseModel):
    """Enforced rotation parameters
    Parameters for enforcing rotation on specific groups of atoms
    using various rotation types and fitting methods.
    """

    rotation: core_mdp.bool_yn | None = None
    """Apply the rotation potential. (False)

    > Controlled by:
    `rot_type0` -  rotation type for group
    `rot_group` - atoms group"""

    rot_ngroups: int | None = None
    """Number of rotation groups. (1)"""

    rot_group0: str | None = None
    """Name of rotation group 0 in the index file."""

    rot_type0: RotType | None = None
    """Rotation type."""

    rot_mass0: core_mdp.bool_yn | None = None
    """Use mass weighted rotation group positions. (False)"""

    rot_vec0: list[float] = []
    """Rotation vector components - x,y,z. (1 0 0)"""

    rot_pivot0: list[float] = []
    """Pivot point for the potentials, in `nm`. (0 0 0)

    - ACTIVE IF `rot_type0` = `Iso`/`PM`/`RM`/`RM2`"""

    rot_rate0: float | None = None
    """Reference rotation rate, in `deg/ps`. (0)"""

    rot_k0: float | None = None
    """Force constant, in `kJ/(mol*nm2)`. (0)"""

    rot_slab_dist0: float | None = None
    """Slab distance, in `nm`. (1.5)

    - ACTIVE IF flexible axis rotation type"""

    rot_min_gauss0: float | None = None
    """Minimum value (cutoff) of Gaussian function for the force to be evaluated. (0.001)

    - ACTIVE IF flexible axis rotation type"""

    rot_eps0: float | None = None
    """Value of additive constant epsilon, in `nm2`. (0.0001)

    - ACTIVE IF `rot_typex` = `RM2*`/`Flex2*`"""

    rot_fit_method0: RotFitMethod | None = None
    """Fitting method when determining the actual angle of a rotation group. (RMSD)"""

    rot_potfit_nsteps0: int | None = None
    """Number of angular positions around the reference angle
    for which the rotation potential is evaluated. (21)

    - ACTIVE IF `rot_fit_methodx` = `Potential`"""

    rot_potfit_step0: float | None = None
    """Distance in degrees between two angular positions. (0.25)

    - ACTIVE IF `rot_fit_methodx` = `Potential`"""

    # TODO: what file
    rot_nstrout: int | None = None
    """Frequency for writing rotation data  to output file:
    angles, torque and the rotation potential energy, in `steps`. (100)"""

    rot_nstsout: int | None = None
    """Frequency for per-slab data:
    flexible axis potentials, i.e. angles, torques and slab centers, in `steps` (1000)"""
