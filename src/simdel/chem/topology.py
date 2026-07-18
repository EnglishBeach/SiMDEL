"""System topologies definition and internal classes."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from simdel import _log, _utils
from simdel._parsers import top_parser

from . import views


class Atoms(_utils.Table):
    """Atom parameters,
    if any parameter not set, value will be set from forcefield atom types.
    """

    ai: pd.Series[int]
    """Internal index."""

    sequence: pd.Series[int]
    """Sequence number."""

    icode: pd.Series[str]
    """Additional letter for sequence."""

    residue: pd.Series[str]
    """Residue name."""

    name: pd.Series[str]
    """Atom name."""

    mass: pd.Series[float]
    """Optional atom mass [a.m.u.]."""

    charge: pd.Series[float]
    """Optional atom charge [electron]."""

    charge_group: pd.Series[int]
    """Charge group number to control zero charge in group."""

    type: pd.Series[str]
    """Atom type name."""

    typeB: pd.Series[str]
    """Atom type B name for alchemy - atom B state in hybrids."""

    massB: pd.Series[float]
    """Optional atom mass [a.m.u.]."""

    chargeB: pd.Series[float]
    """Optional atom charge [electron]."""


class Bonds(_utils.Table):
    """Bond parameters,
    if any parameter not set, value will be set from forcefield bond types.
    """

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class Pairs(_utils.Table):
    """Non-bonded atom interactions for concrete atoms,
    if any parameter not set, value will be set from forcefield pairs types.
    """

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class Constraints(_utils.Table):
    """Fixed distances between atoms, meaning cX depend on function type,
    if any parameter not set, value will be set from forcefield dihedrals types.
    """

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type: 1,2, on difference between, but 2 uses for exclusions."""

    b: pd.Series[float]
    """Distance [nm]."""


class Angles(_utils.Table):
    """Angles parameters table,
    if any parameter not set, value will be set from forcefield angle types.
    """

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    ak: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class Dihedrals(_utils.Table):
    """Dihedral parameters, meaning cX depend on function type,
    if any parameter not set, value will be set from forcefield dihedrals types.
    """

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    ak: pd.Series[int]
    """Atom index."""

    al: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class Exclusions(_utils.Table):
    """Excluded atoms in non-bonded interactions parameters,
    1 or more atoms in interaction.
    """

    A: pd.Series[list]
    """Atom indices."""


class Settles(_utils.Table):
    """Atom distances in rigid water for SETTLE constraint algorithm."""

    OW: pd.Series[int]
    """Water type, always = 1."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class VirtualSites1(_utils.Table):
    """1-body virtual site parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""


class VirtualSites2(_utils.Table):
    """2-body virtual site parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    ak: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class VirtualSites3(_utils.Table):
    """3-body virtual site parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    ak: pd.Series[int]
    """Atom index."""

    al: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class VirtualSites4(_utils.Table):
    """4-body virtual site parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    ak: pd.Series[int]
    """Atom index."""

    al: pd.Series[int]
    """Atom index."""

    af: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class VirtualSitesN(_utils.Table):
    """n-body virtual site parameters."""

    ai: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class PositionRestraints(_utils.Table):
    """Position restraints parameters."""

    ai: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


class DistanceRestraints(_utils.Table):
    """Distance restraints parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    type: pd.Series[str]

    label: pd.Series[str]

    C: pd.Series[list]
    """Parameters list."""


class DihedralRestraints(_utils.Table):
    """Dihedral restraints parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    ak: pd.Series[int]
    """Atom index."""

    al: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    C: pd.Series[list]
    """Parameters list."""


# TODO: exp,label desc
class OrientationRestraints(_utils.Table):
    """Orientation restraints parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    exp: pd.Series[str]
    """"""

    label: pd.Series[str]

    C: pd.Series[list]
    """Parameters list."""


# TODO: n desc
class AngleRestraints(_utils.Table):
    """Angle restraints parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    ak: pd.Series[int]
    """Atom index."""

    al: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    tetta: pd.Series[float]
    """Angle parameter."""

    k: pd.Series[float]
    """Force constant."""

    n: pd.Series[int]


# TODO: n desc
class AngleRestraintsZ(_utils.Table):
    """Angle in z axis restraints parameters."""

    ai: pd.Series[int]
    """Atom index."""

    aj: pd.Series[int]
    """Atom index."""

    func_type: pd.Series[int]
    """Function type."""

    tetta: pd.Series[float]
    """Angle parameter."""

    k: pd.Series[float]
    """Force constant."""

    n: pd.Series[int]


class Topology(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """Molecule topology - interaction and own parameters for atoms, bonds.
    for one molecule type.
    """

    name: str
    """Topology name."""

    nrexcl: int
    """Exclude non-bonded interactions far more 3 bonds."""

    atoms: Atoms
    """Atom parameters utils."""

    bonds: Bonds
    """Bonds parameters utils."""

    pairs: Pairs
    """Pairs parameters utils."""

    # TODO: pairs_nb

    constraints: Constraints
    """Constraints parameters utils."""

    angles: Angles
    """Angles parameters utils."""

    dihedrals: Dihedrals
    """Dihedrals parameters utils."""

    settles: Settles
    """Settles parameters utils."""

    virtual_sites1: VirtualSites1
    """1-body virtual sites parameters utils."""

    virtual_sites2: VirtualSites2
    """2-body virtual sites parameters utils."""

    virtual_sites3: VirtualSites3
    """3-body virtual sites parameters utils."""

    virtual_sites4: VirtualSites4
    """4-body virtual sites parameters utils."""

    exclusions: Exclusions
    """Exclusions parameters utils."""

    position_restraints: PositionRestraints
    """Position restraints for atoms in topology utils."""

    distance_restraints: DistanceRestraints
    """Distance restraints for atoms in topology utils."""

    dihedral_restraints: DihedralRestraints
    """Dihedral restraints for atoms in topology utils."""

    orientation_restraints: OrientationRestraints
    """Orientation restraints for atoms in topology utils."""

    angle_restraints: AngleRestraints
    """Angle restraints for atoms in topology utils."""

    angle_restraints_z: AngleRestraintsZ
    """Angle restraints in z axis for atoms in topology utils."""

    def __hash__(self) -> int:
        hash_str = " ".join(str(i) for i in dict(self).values())
        return hash(hash_str)

    def __repr__(self) -> str:
        return f"<Topology: {self.name}>"

    @property
    def view(self) -> views.TopView:
        """Topology atoms view."""
        if len(self.position_restraints):
            posres_C = (
                pd.concat([self.position_restraints.func_type, self.position_restraints.C], axis=1)
                .apply(lambda x: (x["func_type"], *x["C"]), axis=1)
                .to_numpy()
            )
        else:
            posres_C = None
        posres = pd.DataFrame(dict(ai=self.position_restraints.ai, posres_C=posres_C))

        df = self.atoms.to_df().set_index("ai").join(posres.set_index("ai")).reset_index()
        df["molecule"] = self.name
        return views.TopView(**df)

    @property
    def atomtypes(self) -> set[str]:
        """Topology atomtypes set."""
        return set(self.atoms.type.unique()) | set(self.atoms.typeB.unique())

    @classmethod
    def load(cls, top: top_parser.TopologyData) -> Topology:
        """Create Topology from topology data.

        :param top: Topology parser topology data
        :return: Topology
        """
        max_resname_len = 3
        long_resname = max(top.atoms.residue, key=len)
        if len(long_resname) > max_resname_len:
            msg = f"Residue has too long name: {long_resname}"
            raise ValueError(msg)

        if len(top.atoms.ai) != len(range(1, len(top.atoms.ai) + 1)):
            msg = "Atom index is not continuous"
            _log.warning(msg)

        return Topology(
            name=top.moleculetype.name[0],
            nrexcl=top.moleculetype.nrexcl[0],
            atoms=Atoms(**dict(top.atoms)),
            pairs=Pairs(**dict(top.pairs)),
            bonds=Bonds(**dict(top.bonds)),
            angles=Angles(**dict(top.angles)),
            dihedrals=Dihedrals(**dict(top.dihedrals)),
            exclusions=Exclusions(**dict(top.exclusions)),
            constraints=Constraints(**dict(top.constraints)),
            settles=Settles(**dict(top.settles)),
            virtual_sites1=VirtualSites1(**dict(top.virtual_sites1)),
            virtual_sites2=VirtualSites2(**dict(top.virtual_sites2)),
            virtual_sites3=VirtualSites3(**dict(top.virtual_sites3)),
            virtual_sites4=VirtualSites4(**dict(top.virtual_sites4)),
            position_restraints=PositionRestraints(**dict(top.position_restraints)),
            distance_restraints=DistanceRestraints(**dict(top.distance_restraints)),
            dihedral_restraints=DihedralRestraints(**dict(top.dihedral_restraints)),
            orientation_restraints=OrientationRestraints(**dict(top.orientation_restraints)),
            angle_restraints=AngleRestraints(**dict(top.angle_restraints)),
            angle_restraints_z=AngleRestraintsZ(**dict(top.angle_restraints_z)),
        )

    def dump(self) -> list[str]:
        """Dump topology.

        :return: List of strings
        """
        return top_parser.TopologyData(
            moleculetype=top_parser.MoleculeType(name=[self.name], nrexcl=[self.nrexcl]),
            atoms=top_parser.Atoms(**self.atoms),  # type: ignore
            bonds=top_parser.Bonds(**self.bonds),  # type: ignore
            pairs=top_parser.Pairs(**self.pairs),  # type: ignore
            constraints=top_parser.Constraints(**self.constraints),  # type: ignore
            angles=top_parser.Angles(**self.angles),  # type: ignore
            dihedrals=top_parser.Dihedrals(**self.dihedrals),  # type: ignore
            settles=top_parser.Settles(**self.settles),  # type: ignore
            virtual_sites1=top_parser.Virtual_Sites1(**self.virtual_sites1),  # type: ignore
            virtual_sites2=top_parser.Virtual_Sites2(**self.virtual_sites2),  # type: ignore
            virtual_sites3=top_parser.Virtual_Sites3(**self.virtual_sites3),  # type: ignore
            virtual_sites4=top_parser.Virtual_Sites4(**self.virtual_sites4),  # type: ignore
            exclusions=top_parser.Exclusions(**self.exclusions),  # type: ignore
            position_restraints=top_parser.Position_Restraints(**self.position_restraints),  # type: ignore
            distance_restraints=top_parser.Distance_Restraints(**self.distance_restraints),  # type: ignore
            dihedral_restraints=top_parser.Dihedral_Restraints(**self.dihedral_restraints),  # type: ignore
            orientation_restraints=top_parser.Orientation_Restraints(
                **self.orientation_restraints  # type: ignore
            ),
            angle_restraints=top_parser.Angle_Restraints(**self.angle_restraints),  # type: ignore
            angle_restraints_z=top_parser.Angle_RestraintsZ(**self.angle_restraints_z),  # type: ignore
        ).dump()

    def rename(self, name: str) -> Topology:
        """Rename topology, get new Topology.

        :param name: New name
        :return: Renamed Topology
        """
        return self.model_copy(update=dict(name=name))

    def set_restraints(self, restraints_type: str, restraints_data: _utils.Table) -> Topology:
        """Set restraints, overwrite older restraints with same type.

        :param restraints_type: Restraints type (Restraints class name)
        :param restraints: Topology restraints table
        :return: Updated topology
        """
        return self.model_copy(update={restraints_type: restraints_data})
