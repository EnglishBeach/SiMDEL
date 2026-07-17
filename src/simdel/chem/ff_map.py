"""Forcefield, water type classes and maps."""

from __future__ import annotations

import enum
from pathlib import Path

from pydantic import BaseModel

from simdel import _ff_collection, run
from simdel._misc import log
from simdel._parsers import top_parser


class WaterType(enum.Enum):
    """Water type."""

    spc = "spc"
    spce = "spce"
    tip3p = "tip3p"
    tips3p = "tips3p"
    tip4p = "tip4p"
    tip4pew = "tip4pew"
    tip5p = "tip5p"
    tip5pe = "tip5pe"


_WATER_GEOMETRY_MAP = {
    WaterType.spc: "spc216.gro",
    WaterType.spce: "spc216.gro",
    WaterType.tip3p: "spc216.gro",
    WaterType.tips3p: "spc216.gro",
    WaterType.tip4p: "tip4p.gro",
    WaterType.tip4pew: "tip4p.gro",
    WaterType.tip5p: "tip5p.gro",
    WaterType.tip5pe: "tip5p.gro",
}
"""Map: water type - water geometry file name."""


class FF(BaseModel):
    """Forcefield type."""

    name: str
    """Forcefield name."""

    atomtypes: frozenset[str]
    """Forcefield atomtypes set."""

    moleculetypes: frozenset[str]
    """Forcefield molecules set."""

    def __repr__(self) -> str:
        return f"<Forcefield: {self.name}>"

    def get_water_info(self, water: WaterType | None) -> tuple[int, str]:
        """Get index and geometry from water type.

        :param water: Water type
        :return: Index and geometry file
        """
        ...


class GromacsFF(FF):
    """GROMACS Forcefield."""

    water_map: dict[WaterType, str]
    """Compatible waters with geometries."""

    paths: list[Path]
    """Forcefield path on local anr remote host."""

    @classmethod
    def load(cls, ff: Path, atomtypes: Path | None = None, ions: Path | None = None) -> GromacsFF:
        """Load GROMACS Forcefield from folder.

        :param ff: GROMACS Forcefield folder
        :return: GROMACS Forcefield
        """
        if not ff.exists():
            msg = f"Forcefield does not exist: {ff}"
            raise ValueError(msg)

        elif ff.suffix != ".ff":
            msg = "Custom forcefield dir must have .ff suffix"
            raise ValueError(msg)

        dat = next(i for i in ff.iterdir() if i.suffix == ".dat")
        waters = [WaterType[i.split()[0]] for i in dat.read_text().split("\n") if i.strip()]
        water_map = {i: _WATER_GEOMETRY_MAP[i] for i in waters}

        atomtypes_data = (atomtypes or ff / "atomtypes.atp").read_text().split("\n")
        const_atomtypes = frozenset(i.split()[0] for i in atomtypes_data if i)

        top_file = top_parser.TOPFile.parse(topology=ions or ff / "ions.itp")
        moleculetypes = {"SOL"} | {i.moleculetype.name[0] for i in top_file.top_list}

        return GromacsFF(
            name=ff.stem.replace(".ff", ""),
            paths=[ff],
            water_map=water_map,
            atomtypes=const_atomtypes,
            moleculetypes=frozenset(moleculetypes),
        )

    def remote_register(self, session: run.Session, destination: Path) -> Path:
        """Push forcefield folder on remote volume and add remote path.

        :param session: Remote session
        :param destination: Destination folder on remote
        :return: Remote path
        """
        try:
            remote_path = session.push_folder(
                session=session,
                destination=destination,
                folder=self.paths[0],
            )
            self.paths.append(remote_path)
        except RuntimeError:
            msg = (
                f"Forcefield: {self.name} is already registered on remote: "
                f"{destination}/{self.paths[0].name}"
            )
            log.warning(msg)
        return destination / self.paths[0].name

    def get_water_info(self, water: WaterType | None) -> tuple[int, str]:
        """Get index and geometry from water type.

        :param water: Water type
        :return: Index and geometry file
        """
        if not water:
            return len(self.water_map) + 1, ""
        if not self.water_map.get(water):
            msg = f"Incorrect water type: {water.name} for forcefield: {self.name}"
            raise ValueError(msg)
        return list(self.water_map).index(water) + 1, self.water_map[water]


class OpenFF(FF):
    """Openff forcefield."""

    @classmethod
    def load(cls, name: str) -> OpenFF:
        """Create Openff by name.

        :param name: Openff name
        :return: Openff Forcefield
        """
        return OpenFF(name=name, atomtypes=frozenset(), moleculetypes=frozenset())

    def get_water_info(self, water: WaterType | None) -> tuple[int, str]:
        """Get index and geometry from water type.

        :param water: Water type
        :return: Index and geometry file
        """
        if water:
            msg = "Openff forcefield is incompatible with water types"
            raise ValueError(msg)
        else:
            return 0, ""


class Ion(BaseModel):
    """Base ion."""

    name: str
    """Ion name."""

    charge: int
    """Ion charge."""


# TODO: split to gromacs openff
class DefaultFF:
    """Default forcefield types."""

    amber94: GromacsFF = GromacsFF.load(_ff_collection.AMBER94)
    amber96: GromacsFF = GromacsFF.load(_ff_collection.AMBER96)
    amber99: GromacsFF = GromacsFF.load(_ff_collection.AMBER99)
    amber99sb: GromacsFF = GromacsFF.load(_ff_collection.AMBER99sb)
    amber99sb_ildn: GromacsFF = GromacsFF.load(_ff_collection.AMBER99sb_ildn)
    amber03: GromacsFF = GromacsFF.load(_ff_collection.AMBER03)
    amberGS: GromacsFF = GromacsFF.load(_ff_collection.AMBERGS)
    amber14sb_OL24: GromacsFF = GromacsFF.load(_ff_collection.AMBER14sb_OL24)
    openff210: OpenFF = OpenFF.load("openff-2.1.0")


class DefaultIon:
    """Default ion."""

    Na = Ion(name="NA", charge=1)
    Ca = Ion(name="CA", charge=2)
    Mg = Ion(name="MG", charge=2)
    K = Ion(name="K", charge=1)
    Rb = Ion(name="NA", charge=2)
    Cs = Ion(name="NA", charge=1)
    Li = Ion(name="NA", charge=1)
    Zn = Ion(name="ZN", charge=2)
    Cl = Ion(name="CL", charge=-1)
