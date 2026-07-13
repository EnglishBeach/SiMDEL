"""GROMACS geometry .gro file parser."""

from __future__ import annotations

from pathlib import Path
import typing
from typing import Any

from pydantic import BaseModel


class Table(BaseModel):
    """Base structure for geometry data, table of contents."""

    def __repr__(self) -> str:
        return f"<GRO {self.__class__.__name__}>"

    def __getitem__(self, key: Any) -> list:
        if key not in self.__annotations__:
            msg = f"Table has no {key} column"
            raise KeyError(msg)
        return getattr(self, key)

    def parse(self, string: str):
        """Parse line and add it to table.

        :param string: Content string.
        """
        try:
            datas = self._parse(string)
            for key, value in datas.items():
                self[key].append(value)
        except Exception as e:
            msg = f"Parsing error of {self.__class__.__name__}"
            raise ValueError(msg) from e

    def dump(self) -> list[str]:
        """Save Table to list of stings."""
        return [self._dump(*i) for i in zip(*dict(self).values(), strict=True)]

    def _parse(self, string: str) -> dict: ...

    def _dump(self, *args: typing.Any) -> str: ...


class Info(Table):
    """Optional geometry information."""

    entries: list[str] = []

    def _parse(self, string: str) -> dict:
        return dict(entries=string)


class Atoms(Table):
    """Atoms information."""

    resSeq: list[int] = []
    resName: list[str] = []
    name: list[str] = []
    serial: list[int] = []
    x: list[float] = []
    y: list[float] = []
    z: list[float] = []
    vx: list[float | None] = []
    vy: list[float | None] = []
    vz: list[float | None] = []

    def _parse(self, string: str) -> dict:
        s = f"{string: <68}"

        vx = s[44:52].strip()
        vy = s[52:60].strip()
        vz = s[60:68].strip()
        return dict(
            resSeq=int(s[:5]),
            resName=s[5:10].strip(),
            name=s[10:15].strip(),
            serial=int(s[15:20]),
            x=float(s[20:28]),
            y=float(s[28:36]),
            z=float(s[36:44]),
            vx=float(vx) if vx != "" else None,
            vy=float(vy) if vy != "" else None,
            vz=float(vz) if vz != "" else None,
        )

    def _dump(self, *args) -> str:
        sq, rs, nm, s, x, y, z, vx, vy, vz = args
        vx = f"{vx: >8.4f}" if vx is not None else ""
        vy = f"{vy: >8.4f}" if vy is not None else ""
        vz = f"{vz: >8.4f}" if vz is not None else ""
        return f"{sq: >5}{rs: >5}{nm: >5}{s: >5}{x: >8.3f}{y: >8.3f}{z: >8.3f}{vx}{vy}{vz}"


class Box(Table):
    """Box information."""

    x1: list[float] = []
    y2: list[float] = []
    z3: list[float] = []
    y1: list[float | None] = []
    z1: list[float | None] = []
    x2: list[float | None] = []
    z2: list[float | None] = []
    x3: list[float | None] = []
    y3: list[float | None] = []

    def _parse(self, string: str) -> dict:
        data = [float(i) for i in string.split()]
        return dict(
            zip(
                ["x1", "y2", "z3", "y1", "z1", "x2", "z2", "x3", "y3"],
                data + [None] * (9 - len(data)),
                strict=True,
            )
        )

    def _dump(self, *args) -> str:
        return "".join([f"{i: >10.5f}" for i in args if i is not None])


class GROFile(BaseModel):
    """Data container for .gro file, parse and save it."""

    info: Info = Info()
    """Other information."""

    atoms: Atoms = Atoms()
    """Atoms coordinates."""

    box: Box | None = None
    """Box information."""

    @classmethod
    def parse(cls, gro: Path) -> GROFile:
        """Parse .gro file.

        :param gro: Path to .gro file
        :return: Data container for .gro file
        """
        if gro.suffix != ".gro":
            msg = f"Incorrect file suffix: {gro.suffix}"
            raise ValueError(msg)

        parser = GROFile()
        with gro.open() as file:
            next(file)
            number = int(next(file).strip())
            for i, line in enumerate(file):
                block = parser._get_table(i < number)
                try:
                    block.parse(line)
                except Exception as e:
                    msg = f"Parsing error of in {gro.resolve()}:\n[{i + 3: >6}] {line}"
                    raise ValueError(msg) from e
        return parser

    def dump(self) -> list[str]:
        """Dump .gro file container to list of strings.

        :return: GROMACS geometry .gro text
        """
        data = [
            "Geometry",
            f"{len(self.atoms.name): >5}",
        ]
        data.extend(self.atoms.dump())
        if self.box:
            data.extend(self.box.dump())
        data.append("")
        return data

    def _get_table(self, is_atoms: bool) -> Table:
        """Get table accordingly section in .gro file.

        :param is_atoms: Is atom string
        :return: Table
        """
        if is_atoms:
            return self.atoms
        self.box = Box()
        return self.box
