"""Molecular dynamic parameters for NMR refinement, Collective Variables.
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

from pydantic import BaseModel

from . import core_mdp


class Disre(enum.Enum):
    """Type of distance constraints for NMR refinement."""

    No = "no"
    """Ignore distance restraint information in topology file."""

    Simple = "simple"
    """Simple (per-molecule) distance restraints."""

    Ensemble = "ensemble"
    """Distance restraints over an ensemble of molecules in one simulation box,
    for dmx mdrun -multidir."""


class DisreWeighting(enum.Enum):
    """Atom weighting type."""

    Equal = "equal"
    """Divide the restraint force equally over all atom pairs in the restraint."""

    Conservative = "conservative"
    """The forces are the derivative of the restraint potential,
    this results in an weighting of the atom pairs to the reciprocal
    seventh power of the displacement."""


class NMRDihedralConstraintMethod(enum.Enum):
    """Method for dihedral constraint potential."""

    cosine = "cosine"
    """Cosine-based potential."""

    harmonic = "harmonic"
    """Harmonic potential."""


class GroupNMRRefinement(BaseModel):
    """NMR refinement parameters
    Parameters for NMR structure refinement using distance, dihedral,
    and other constraints derived from NMR experimental data.
    """

    # TODO: defaults (may be NO)
    disre: Disre | None = None
    """Type of distance constraints for NMR refinement."""

    disre_weighting: DisreWeighting | None = None
    """Use weighting for distance restraints. (Conservative)
    - ACTIVE IF `disre_tau` = 0."""

    disre_mixed: core_mdp.bool_yn | None = None
    """Violation =sqrt(t-mean(violation) * instantaneous violation). (False)"""

    disre_fc: float | None = None
    """Force constant for distance restraints, multiplied by a (possibly) different factor
    for each restraint given in the `fac` column of the interaction in the topology .top file.
    In `kJ/(mol*nm2)`. (1000)"""

    disre_tau: float | None = None
    """Time constant for distance restraints running average, in `ps`. (0)

    - ` =0` => time averaging is off"""

    nstdisreout: int | None = None
    """Period between steps when the running time-averaged and instantaneous distances
    of all atom pairs involved in restraints are written to the energy file, in `steps`. (100)"""

    orire: core_mdp.bool_yn | None = None
    """Use orientation restraints. (False)"""

    orire_fc: float | None = None
    """Force constant for orientation restraints, in `kJ/mol`. (0)

    - `= 0` => free simulation. """

    orire_tau: float | None = None
    """Time constant for orientation restraints running average, in `ps`. (0)

    - `= 0` => no time averaging"""

    orire_fitgrp: str | None = None
    """Fit group for orientation restraining.
    This group of atoms is used to determine the rotation R of the system with respect to
    the reference orientation. The reference orientation is the starting conformation of
    the first subsystem."""

    nstorireout: int | None = None
    """Period between steps when the running time-averaged and instantaneous orientations
    for all restraints, in `steps`. (100)"""


class GroupColVars(BaseModel):
    """Collective variables module parameters."""

    colvars_active: core_mdp.bool_tf | None = None
    """Activate Colvars computation.
    ACTIVE IF Colvars library was compiled with GROMACS. (False)"""

    colvars_configfile: str | None = None
    """Name of the Colvars configuration file."""

    colvars_seed: int | None = None
    """Seed used to initialize the random generator associated with certain
    stochastic methods implemented within Colvars. (-1)

    - `= -1` => random"""
