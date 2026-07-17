"""Molecular dynamic parameters for Neighbor searching, Electrostatics, VanDerWaals interactions,
Tables, Potential mesh (PME).
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

import pydantic

from . import core_mdp


class CutoffScheme(enum.Enum):
    """Fair interaction type."""

    Verlet = "Verlet"
    """Generate a pair list with buffering.
    - `verlet_buffer_tolerance > 0` => set buffer size based on
    - `verlet_buffer_tolerance < 0` and `rlist` => set buffer size based on `rlist`"""


class PBC(enum.Enum):
    """Periodic boundary condition types."""

    XYZ = "xyz"
    """Use periodic boundary conditions in all directions"""

    NO = "no"
    """Use no periodic boundary conditions, ignore the box.
    - `all cut-offs and nstlist = 0` => simulate without cut-offs"""

    XY = "xy"
    """Use periodic boundary conditions in x and y directions only,
    can be used in combination with walls"""


class GroupNeighborSearching(pydantic.BaseModel):
    """Nonbonded interactions detecting buffer parameters."""

    cutoff_scheme: CutoffScheme | None = None
    """Fair interaction type. (Verlet)"""

    nstlist: int | None = None
    """Interval between steps that update the neighbor list, >= 0, in `steps`. (10)

    - ACTIVE IF dynamic
    - dynamic and `verlet_buffer_tolerance` set and `nstlist`is min => `nstlist` increased
    - `= 0` => neighbor list is only constructed once"""

    pbc: PBC | None = None
    """Periodic boundary condition types. (xyz)"""

    periodic_molecules: core_mdp.bool_yn | None = None
    """For systems with molecules that couple to themselves through the periodic boundary condition.
    False => molecules are finite, faster PBC algorithm. (False)"""

    verlet_buffer_tolerance: float | None = None
    """Maximum allowed error for pair interactions per particle
    caused by the Verlet buffer, which indirectly sets `rlist`, in `kJ/(mol*ps)`. (0.005)

    - ACTIVE IF dynamic
    - `= -1` => no automatic set of buffer
    - `EM is off` => buffer = 0.05*`verlet_buffer_tolerance`
    - `NVE, T !=0` => buffer = 0.1*`verlet_buffer_tolerance`"""

    verlet_buffer_pressure_tolerance: float | None = None
    """Maximum tolerated error in the average pressure due to missing Lennard-Jones
    interactions of particle pairs that are not in the pair list, in `bar`. (0.5)

    - ACTIVE IF dynamic, `verlet_buffer_tolerance` > 0"""

    rlist: float | None = None
    """Cut-off distance for the short-range neighbor interaction, sets automatically.
    ACTIVE IF `verlet_buffer_tolerance`/`verlet_buffer_pressure_tolerance` is off.
    In `nm`. (1)

    - em => max(cut-off)+5%"""


class CoulombType(enum.Enum):
    """Coulomb electrostatics type."""

    Cutoff = "cut-off"
    """Plain cut-off with pair list radius `rlist` and Coulomb cut-off rcoulomb,
    must `rlist` >= `rcoulomb`."""

    Ewald = "Ewald"
    """Classical Ewald sum electrostatics for Coulomb interactions, must `rcoulomb` = `rlist`.

    > Controlled by:
    - `fourierspacing` - the highest magnitude of wave vectors
    - `ewald_rtol` - relative accuracy of direct/reciprocal space"""

    PME = "PME"
    """Fast smooth Particle-Mesh Ewald (SPME) for Coulomb interactions.

    > Controlled by:
    - `fourierspacing` - grid dimensions
    - `pme_order` - interpolation order"""

    P3MAD = "P3M-AD"
    """Particle-Particle Particle-Mesh algorithm with analytical derivative
    for long range electrostatic interactions, same to `PME`, more accurate."""

    ReactionField = "Reaction-Field"
    """Reaction field electrostatics.

    - active => `rlist >= rvdw`
    > Controlled by:
    - `rcoulomb` - Coulomb cut-off
    - `epsilon_rf` - dielectric constant, may be set 0 to infinity"""


# TODO: desc
class CoulombModifier(enum.Enum):
    """Couloumb potential modifier."""

    PotentialShiftVerlet = "Potential-shift-Verlet"
    """Shift the Coulomb potential by a constant such that it is zero at the cut-off,
    makes the potential the integral of the force."""

    PotentialShift = "Potential-shift"

    PotentialSwitch = "Potential-switch"

    ExactCutoff = "Exact-cutoff"

    ForceSwitch = "Force-switch"

    NONE = "None"
    """Use an unmodified Coulomb potential."""


class GroupElectrostatics(pydantic.BaseModel):
    """Coulomb interaction parameters."""

    coulombtype: CoulombType | None = None
    """Coulomb electrostatics type. (Cutoff)"""

    coulomb_modifier: CoulombModifier | None = None
    """Coulomb potential modifier. (PotentialShiftVerlet)"""

    rcoulomb_switch: float | None = None
    """Where to start switching the Coulomb potential, in `nm`. (0)

    - ACTIVE IF force or potential switching is on"""

    rcoulomb: float | None = None
    """Distance for the Coulomb cut-off, in `nm`. (1)"""

    epsilon_r: float | None = None
    """The relative dielectric constant. (1)

    - `= 0` => infinity"""

    epsilon_rf: float | None = None
    """The relative dielectric constant of the reaction field. (0)

    - ACTIVE IF reaction-field electrostatics.
    - `= 0` => infinity"""


class VDWType(enum.Enum):
    """Van der Waals interactions type."""

    Cutoff = "cut-off"
    """Plain cut-off with pair list and VdW cut-off.

    - active => `rlist >= rvdw`
    > Controlled by:
    - `rlist` - cut-off radius
    - `rvdw` - VdW cut-off radius"""

    PME = "PME"
    """Fast smooth Particle-mesh Ewald (SPME) for Van der Waals interactions.

    > Controlled by:
    - `fourierspacing` - grid dimensions
    - `pme_order` - interpolation order
    - `ewald_rtol_lj` - relative accuracy of direct/reciprocal space
    - `lj_pme_comb_rule` - specific combination rules"""


# TODO: desc
class VDWModifier(enum.Enum):
    """Van der Waals potential modifications."""

    PotentialShiftVerlet = "Potential-shift-Verlet"
    """Selects Potential-shift with the Verlet cutoff-scheme."""

    PotentialShift = "Potential-shift"
    """Shift the Van der Waals potential by a constant such that it is zero at the cut-off."""

    NONE = "None"
    """Use an unmodified Van der Waals potential."""

    ForceSwitch = "Force-switch"
    """Smoothly switches the forces to zero between `rvdw_switch` and `rvdw`.
    This shifts the potential shift over the whole range and switches it to zero at the cut-off.
    Slowly and conserves energy as `PotentialShift`."""

    PotentialSwitch = "Potential-switch"
    """Smoothly switches the forces to zero between `rvdw_switch` and `rvdw`.
    Very slow, need use only in forcefield requires it."""

    ExactCutoff = "Exact-cutoff"


# TODO:  'AllEnerPres' 'AllEner' is not in docs, but is in gromacs stdout
class DispCorr(enum.Enum):
    """Dispersion correction type."""

    NO = "no"
    """Don't apply any correction."""

    EnerPres = "EnerPres"
    """Apply long range dispersion corrections for Energy and Pressure."""

    Ener = "Ener"
    """Apply long range dispersion corrections for Energy only."""


class GroupVanDerWaals(pydantic.BaseModel):
    """Van der Waals interaction parameters."""

    vdw_type: VDWType | None = None
    """Van der Waals interactions type. (Cutoff)"""

    vdw_modifier: VDWModifier | None = None
    """Van der Waals potential modifications. (PotentialShiftVerlet)"""

    rvdw_switch: float | None = None
    """Where to start switching the LJ force and possibly the potential, in `nm`. (0)
    - ACTIVE IF force/potential switching is on."""

    rvdw: float | None = None
    """Distance for the LJ or Buckingham cut-off, in `nm`. (1)"""

    dispcorr: DispCorr | None = None
    """Dispersion correction type. (NO)"""


class GroupTables(pydantic.BaseModel):
    """Table parameters."""

    table_extension: float | None = None
    """Extension of the non-bonded potential lookup tables beyond the largest cut-off distance.
    For log interactions (1-4 for example), in `nm`. (1)"""


class LJPMEcombRule(enum.Enum):
    """The combination rules used to combine VdW-parameters in the reciprocal part of LJ-PME."""

    Geometric = "Geometric"
    """Apply geometric combination rules, faster."""

    LorentzBerthelot = "Lorentz-Berthelot"
    """Apply Lorentz-Berthelot combination rules."""


class EwaldGeometry(enum.Enum):
    """The Ewald sum in dimensions type."""

    D3 = "3d"
    """The Ewald sum is performed in all three dimensions."""

    D3c = "3dc"
    """The reciprocal sum is still performed in 3D, but
    a force and potential correction applied in the z dimension to produce a pseudo-2D summation,
    system has a slab geometry in the x-y plane => may be to increase the z-dimension of the box
    (a box height of 3 times the slab height is usually ok)."""


class GroupPotentialMesh(pydantic.BaseModel):
    """Potential calculation in space parameters."""

    fourierspacing: float | None = None
    """Depends on coulomb type, in `nm`. (0.12)

    - `Ewald` => ratio of the box dimensions and the spacing determines
    a lower bound for the number of wave vectors to use in each (signed) direction
    - `PME`/`P3MAD` => lower bound for the number of Fourier-space that will be used along axis
    > Controlled by:
    - `fourier-nx` - overrides fourierspacing in one directions
    - `rcoulomb` - scale fourierspacing"""

    fourier_nx: int | None = None
    """FFT grid size in x direction. Depends on coulomb type. Best - [2,3,5,7]. (0)

    - `Ewald` => highest magnitude of wave vectors in reciprocal space
    - `PME`/`P3MAD` => grid size
    - `= 0` => no overwrite `fourierspacing`"""

    fourier_ny: int | None = None
    """FFT grid size in y direction. Depends on coulomb type.Best - [2,3,5,7]. (0)

    - `Ewald` => highest magnitude of wave vectors in reciprocal space
    - `PME`/`P3MAD` => grid size
    - `= 0` => no overwrite `fourierspacing`"""

    fourier_nz: int | None = None
    """FFT grid size in z direction. Depends on coulomb type. Best - [2,3,5,7]. (0)

    - `Ewald` => highest magnitude of wave vectors in reciprocal space
    - `PME`/`P3MAD` => grid size
    - `= 0` => no overwrite `fourierspacing`"""

    pme_order: int | None = None
    """Interpolation order for PME, range=[3:12], GPU support only 4. (4)"""

    ewald_rtol: float | None = None
    """Relative strength of the Ewald-shifted
    direct potential at `rcoulomb` is given by `ewald_rtol`. (1e-5)"""

    ewald_rtol_lj: float | None = None
    """`PME` for vdw => relative strength of the dispersion potential at `rvdw`
    in the same way as `ewald_rtol`. (0.001)"""

    lj_pme_comb_rule: LJPMEcombRule | None = None
    """The combination rules used to combine VdW-parameters
    in the reciprocal part of LJ-PME. (Geometric)"""

    ewald_geometry: EwaldGeometry | None = None
    """The Ewald sum in dimensions type. (D3)"""

    epsilon_surface: float | None = None
    """The dipole correction to the Ewald summation in 3D. (0)

    - ACTIVE IF `ewald_geometry` = `D3`
    - `= 0` => turn off
    - 3 mobile charges in system => 0"""
