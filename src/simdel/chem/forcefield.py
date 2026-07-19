"""System forcefield definition and internal classes."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from simdel import _log, _utils
from simdel._parsers import top_parser


class Defaults(_utils.Table):
    """Non-bonded interactions parameters.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/topologies/parameter-files.html#nbpar.
    """

    nonbounded_func_type: pd.Series[int]
    """Non-bonded function type: Lennard-Jones (1), Buckingham (2)."""

    combination_rule: pd.Series[int]
    """Define the Lennard-Jones potential function,
    change meaning of sigma/epsilon parameters in atom types
    """

    generate_pairs: pd.Series[bool]
    """Generate pair parameters from Lennard-Jones potential."""

    fudge_LJ: pd.Series[float]
    """Scaling factor for the Lennard-Jones interaction, is used only when `generate_pairs=True`."""

    fudge_QQ: pd.Series[float]
    """Scaling factor for the Coulomb interaction, is always used."""


class Atomtypes(_utils.Table):
    """Atom types, specifying default `atoms` parameters: bonded and non-bonded.
    The Lennard-Jones potential V = (sigma/r^6)-(epsilon/r^12),
    meaning sigma and epsilon in it depends on `Defaults.combination_rule`.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/functions/nonbonded-interactions.html.
    """

    type: pd.Series[str]
    """Atom type name."""

    bonded_type: pd.Series[str]
    """Optional atom type name for bonded interactions."""

    atomic_number: pd.Series[int]
    """Optional atom number."""

    mass: pd.Series[float]
    """Optional atom mass [a.m.u.]."""

    charge: pd.Series[float]
    """Optional atom charge [electron]."""

    particle_type: pd.Series[str]
    """Particle type.

    - A - atom
    - S - shell
    - V/D - virtual site."""

    C: pd.Series[list]
    """Parameters list."""


class BondTypes(_utils.Table):
    """Bond types, specifying default `bonds` parameters in topologies.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#bond-stretching.
    """

    i: pd.Series[int]
    """Atomtype."""

    j: pd.Series[int]
    """Atomtype."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class PairTypes(_utils.Table):
    """Non-bonded atom interactions types for concrete atoms types,
    specifying default `pairs` in topologies.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/topologies/molecule-definition.html#intramolecular-pair-interactions.
    """

    i: pd.Series[int]
    """Atomtype."""

    j: pd.Series[int]
    """Atomtype."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


# TODO: restraints!=Constraint
class ConstraintTypes(_utils.Table):
    """Constraint types, specifying default `constraints` in topologies.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/functions/restraints.html.
    """

    i: pd.Series[int]
    """Atomtype."""

    j: pd.Series[int]
    """Atomtype."""

    func_type: pd.Series[int]
    """Function type."""

    b: pd.Series[float]
    """Constraint force constant."""


class AngleTypes(_utils.Table):
    """Angles types, specifying default `angles` parameters in topologies.
    Meaning theta, cth depends on function type.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#harmonic-angle-potential.
    """

    i: pd.Series[int]
    """Atomtype."""

    j: pd.Series[int]
    """Atomtype."""

    k: pd.Series[int]
    """Atomtype."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class DihedralTypes(_utils.Table):
    """Dihedral types, meaning cX depend on function type,
    specifying default `dihedrals` angle parameters in topologies.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/functions/bonded-interactions.html#bond-bond-cross-term.
    """

    i: pd.Series[int]
    """Atomtype."""

    j: pd.Series[int]
    """Atomtype."""

    k: pd.Series[int]
    """Atomtype."""

    l: pd.Series[int]
    """Atomtype."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class ImplicitGenbornParams(_utils.Table):
    """Implicit solvation parameters."""

    i: pd.Series[str]
    """Atomtype."""

    C: pd.Series[list]
    """Parameters list."""


class NonbondParams(_utils.Table):
    """Non-bonded atom interactions types for all atom types,
    specifying default `nonbond_params` parameters.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/topologies/parameter-files.html#non-bonded-parameters.
    """

    i: pd.Series[int]
    """Atomtype."""

    j: pd.Series[int]
    """Atomtype."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class Cmaptypes(_utils.Table):
    """C-map types for charm ff, specifying default `cmaptypes` parameters.

    See GROMACS documentation:
    httop_parsers://manual.gromacs.org/current/reference-manual/topologies/parameter-files.html#non-bonded-parameters.
    """

    i: pd.Series[int]
    """Atomtype."""

    j: pd.Series[int]
    """Atomtype."""

    k: pd.Series[int]
    """Atomtype."""

    l: pd.Series[int]
    """Atomtype."""

    m: pd.Series[int]
    """Atomtype."""

    f1: pd.Series[int]
    """Function type."""

    f2: pd.Series[int]
    """Function type."""

    f3: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class Forcefield(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """System forcefield - general interaction parameters for atoms, bonds..."""

    defaults: Defaults
    """Nonbounded interactions data."""

    atomtypes: Atomtypes
    """Atom types, specifying default `atom` parameters."""

    bondtypes: BondTypes
    """Bond types, specifying default `bond` parameters in topologies."""

    pairtypes: PairTypes
    """Non-bonded atom interactions types, specifying default `pairs` in topologies."""

    constrainttypes: ConstraintTypes
    """Constraint types, specifying default `constraints` in topologies."""

    angletypes: AngleTypes
    """Angles types, specifying default `angle` parameters in topologies."""

    dihedraltypes: DihedralTypes
    """Dihedral types, specifying default `dihedral` angle parameters in topologies."""

    implicit_genborn_params: ImplicitGenbornParams
    """Implicit solvation parameters"""

    nonbond_params: NonbondParams
    """Lennard-Jones homo-/hetero-atomic interactions."""

    cmaptypes: Cmaptypes

    def __add__(self, ff: Forcefield) -> Forcefield:
        """Sum forcefields together and resolve conflicts or raise error.
        Take defaults parameters from self if they not same.

        :param ff: Other forcefield
        :return: Mixed forcefield
        """
        if self.defaults != ff.defaults:
            msg = (
                "Nonbonded interaction parameters not equal:\n"
                f"Cols :{' '.join(self.defaults.keys())}\n"
                f"Self :{list(self.defaults[0])}\n"
                f"Other:{list(ff.defaults[0])}"
            )
            _log.warning(msg)

        new_data = {
            key: table.__class__(
                **pd.concat([table.to_df(), getattr(ff, key).to_df()])
                .reset_index(drop=True)
                .drop_duplicates()
            )
            for key, table in dict(self).items()
            if key != "defaults"
        } | dict(defaults=self.defaults)

        new_atomtypes: Atomtypes = new_data["atomtypes"]
        if any(new_atomtypes.type.duplicated()):
            types = set(new_atomtypes.type[new_atomtypes.type.duplicated()])
            msg = f"Atomtypes conflict: {types}"
            raise ValueError(msg)

        return Forcefield(**new_data)

    def __hash__(self) -> int:
        hash_str = " ".join(str(i) for i in dict(self).values())
        return hash(hash_str)

    def __repr__(self) -> str:
        return f"<ForceField {len(self.atomtypes)} atomtypes>"

    def clear_atomtypes(self, atom_types: set[str]) -> Forcefield:
        """Clear atomtypes from all forcefield tables.

        :param atom_types: Atomtypes to exclude
        :return: Cleared forcefield
        """
        new_atomtypes = self.atomtypes[self.atomtypes.type.map(lambda x: x not in atom_types)]

        def f(x):  # type: ignore
            x: _utils.Table
            mask = x.to_df().map(lambda x: x not in atom_types)
            return x[mask.all(axis=1)]

        return Forcefield(
            defaults=self.defaults,
            atomtypes=Atomtypes(**dict(new_atomtypes)),
            bondtypes=f(self.bondtypes),
            pairtypes=f(self.pairtypes),
            constrainttypes=f(self.constrainttypes),
            angletypes=f(self.angletypes),
            dihedraltypes=f(self.dihedraltypes),
            implicit_genborn_params=f(self.implicit_genborn_params),
            nonbond_params=f(self.nonbond_params),
            cmaptypes=f(self.cmaptypes),
        )

    @classmethod
    def load(cls, ff: top_parser.FFData) -> Forcefield:
        """Create FF from forcefield data.

        :param ff: Topology parser forcefield data
        :return: Forcefield
        """
        return Forcefield(
            defaults=Defaults(**dict(ff.defaults)),
            atomtypes=Atomtypes(**dict(ff.atomtypes)),
            bondtypes=BondTypes(**dict(ff.bondtypes)),
            pairtypes=PairTypes(**dict(ff.pairtypes)),
            constrainttypes=ConstraintTypes(**dict(ff.constrainttypes)),
            angletypes=AngleTypes(**dict(ff.angletypes)),
            dihedraltypes=DihedralTypes(**dict(ff.dihedraltypes)),
            implicit_genborn_params=ImplicitGenbornParams(**dict(ff.implicit_genborn_params)),
            nonbond_params=NonbondParams(**dict(ff.nonbond_params)),
            cmaptypes=Cmaptypes(**dict(ff.cmaptypes)),
        )

    def dump(self) -> list[str]:
        """Dump forcefield parameters to GROMACS topology .top format.

        :return: List of strings
        """
        return top_parser.FFData(
            defaults=top_parser.Defaults(**self.defaults),  # type: ignore
            atomtypes=top_parser.AtomTypes(**self.atomtypes),  # type: ignore
            bondtypes=top_parser.BondTypes(**self.bondtypes),  # type: ignore
            pairtypes=top_parser.PairTypes(**self.pairtypes),  # type: ignore
            constrainttypes=top_parser.ConstraintTypes(**self.constrainttypes),  # type: ignore
            angletypes=top_parser.AngleTypes(**self.angletypes),  # type: ignore
            dihedraltypes=top_parser.DihedralTypes(**self.dihedraltypes),  # type: ignore
            implicit_genborn_params=top_parser.Implicit_Genborn_Params(
                **self.implicit_genborn_params  # type: ignore
            ),
            nonbond_params=top_parser.Nonbond_Params(**self.nonbond_params),  # type: ignore
            cmaptypes=top_parser.Cmaptypes(**self.cmaptypes),  # type: ignore
        ).dump()
