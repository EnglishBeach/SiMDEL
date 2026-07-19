"""GROMACS topology .top/.itp files parser."""

from __future__ import annotations

from pathlib import Path
import re
import typing
from typing import Any

from pydantic import BaseModel

from simdel import _log


class Table(BaseModel):
    """Base structure for geometry data, table of contents."""

    def __repr__(self) -> str:
        return f"<TOP {self.__class__.__name__}>"

    def __getitem__(self, key: Any) -> list:
        if key not in self.__annotations__:
            msg = f"Table has no {key} column"
            raise KeyError(msg)
        return getattr(self, key)  # type: ignore

    def parse(self, string: str):
        """Parse content line (without comment) and add it to table.

        :param string: Content string.
        """
        try:
            datas = self._parse(string.split())
            for key, value in datas.items():
                self[key].append(value)
        except Exception as e:
            msg = f"Parsing error of {self.__class__.__name__}"
            raise ValueError(msg) from e

    def dump(self) -> list[str]:
        """Save Table to list of stings."""
        fields = self.__class__.model_fields.copy()
        title = [
            f"[ {self.__class__.__name__.lower()} ]",
            "; " + "  ".join(fields),
        ]

        dump = [" ".join(self._dump(*i)) for i in zip(*dict(self).values(), strict=True)]
        if dump:
            dump = title + dump + [""]
        return dump

    def _parse(self, words: list[str]) -> dict: ...

    def _dump(self, *args: typing.Any) -> list[str]: ...


class Data(BaseModel):
    """Base data container for topology file data."""

    def dump(self) -> list[str]:
        """Dump topology data.

        :return list: Content list of strings.
        """
        data = []
        tables: dict[str, Table] = dict(self)
        for field in tables.values():
            data.extend(field.dump())

        if data:
            data.append("")
        return data


class Info(Table):
    """Optional geometry information."""

    entries: list[str] = []

    def _parse(self, words: list[str]) -> dict:
        return dict(entries=" ".join(words))


# FF
class Defaults(Table):
    """Section in topology .top/.itp file: [ defaults ]."""

    nonbounded_func_type: list[int] = []
    combination_rule: list[int] = []
    generate_pairs: list[bool] = []
    fudge_LJ: list[float] = []
    fudge_QQ: list[float] = []

    def _parse(self, words: list[str]) -> dict:
        nbfunc, combrule, genpairs, fudgeLJ, fudgeQQ = words
        return dict(
            nonbounded_func_type=int(nbfunc),
            combination_rule=int(combrule),
            generate_pairs=genpairs == "yes",
            fudge_LJ=float(fudgeLJ),
            fudge_QQ=float(fudgeQQ),
        )

    def _dump(self, *args) -> list[str]:
        nb, cb, gn, lj, qq = args
        gn_str = "yes" if gn else "no"
        return [
            f"{nb: <10}",
            f"{cb: <10}",
            f"{gn_str: <10}",
            f"{lj: <10}",
            f"{qq: <10}",
        ]


class AtomTypes(Table):
    """Section in topology .top/.itp file: [ atomtypes ]."""

    type: list[str] = []
    bonded_type: list[str | None] = []
    atomic_number: list[int | None] = []
    mass: list[float] = []
    charge: list[float] = []
    particle_type: list[str] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        type_ = words.pop(0)
        bonded_atomic_field_n = 7
        atomic_field_n = 5
        nobonded_atomic_field_n = 6

        if len(words) == bonded_atomic_field_n:
            bonded_type = words.pop(0)
            atomic_num = words.pop(0)

        elif len(words) == atomic_field_n:
            atomic_num = bonded_type = None

        elif len(words) == nobonded_atomic_field_n and words[0].isnumeric():
            bonded_type = None
            atomic_num = words.pop(0)

        else:
            bonded_type = words.pop(0)
            atomic_num = None

        mass, charge, ptype, sigma, epsilon = words
        return dict(
            type=type_,
            bonded_type=bonded_type,
            atomic_number=int(atomic_num) if atomic_num else None,
            mass=float(mass),
            charge=float(charge),
            particle_type=ptype,
            C=(float(sigma), float(epsilon)),
        )

    def _dump(self, *args) -> list[str]:
        name, bt, an, mass, charge, pt, C = args
        return [
            _at(name),
            f"{bt: >4}" if bt else " " * 4,
            f"{an or 0: >4}",
            _c(mass),
            _c(charge),
            f"{pt:<4}",
            *[_c(c) for c in C],
        ]


class BondTypes(Table):
    """Section in topology .top/.itp file: [ bondtypes ]."""

    i: list[str] = []
    j: list[str] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        i, j, func_type = words[:3]
        return dict(
            i=i,
            j=j,
            func_type=int(func_type),
            C=tuple(float(i) for i in words[3:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, funct, C = args
        return [
            _at(ai),
            _at(aj),
            _f(funct),
            *[_c(c) for c in C],
        ]


class PairTypes(Table):
    """Section in topology .top/.itp file: [ pairtypes ]."""

    i: list[str] = []
    j: list[str] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        i, j, func_type = words[:3]
        return dict(
            i=i,
            j=j,
            func_type=int(func_type),
            C=tuple(float(i) for i in words[3:]),
        )

    def _dump(self, *args) -> list[str]:
        i, j, func_type, C = args
        return [
            _at(i),
            _at(j),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class ConstraintTypes(Table):
    """Section in topology .top/.itp file: [ constrainttypes ]."""

    i: list[str] = []
    j: list[str] = []
    func_type: list[int] = []
    b: list[float] = []

    def _parse(self, words: list[str]) -> dict:
        i, j, func_type, b = words
        return dict(
            i=i,
            j=j,
            func_type=int(func_type),
            b=float(b),
        )

    def _dump(self, *args) -> list[str]:
        i, j, func_type, b = args
        return [
            _at(i),
            _at(j),
            _f(func_type),
            _c(b),
        ]


class AngleTypes(Table):
    """Section in topology .top/.itp file: [ angeltypes ]."""

    i: list[str] = []
    j: list[str] = []
    k: list[str] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        i, j, k, func_type = words[:4]
        return dict(
            i=i,
            j=j,
            k=k,
            func_type=int(func_type),
            C=tuple(float(i) for i in words[4:]),
        )

    def _dump(self, *args) -> list[str]:
        i, j, k, func_type, C = args
        return [
            _at(i),
            _at(j),
            _at(k),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class DihedralTypes(Table):
    """Section in topology .top/.itp file: [ dihedraltypes ]."""

    i: list[str] = []
    j: list[str] = []
    k: list[str] = []
    l: list[str] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        i, j, k, l, func_type = words[:5]
        return dict(
            i=i,
            j=j,
            k=k,
            l=l,
            func_type=int(func_type),
            C=tuple(float(i) for i in words[5:]),
        )

    def _dump(self, *args) -> list[str]:
        i, j, k, l, func_type, C = args
        return [
            _at(i),
            _at(j),
            _at(k),
            _at(l),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Implicit_Genborn_Params(Table):
    """Section in topology .top/.itp file: [ implicit_genborn_params ]."""

    i: list[str] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        return dict(
            i=words[0],
            C=tuple(float(i) for i in words[1:]),
        )

    def _dump(self, *args) -> list[str]:
        i, C = args
        return [
            _at(i),
            *[_c(i) for i in C],
        ]


class Nonbond_Params(Table):
    """Section in topology .top/.itp file: [ nonbond_params ]."""

    i: list[str] = []
    j: list[str] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        i, j, func_type = words[:3]
        return dict(
            i=i,
            j=j,
            func_type=int(func_type),
            C=tuple(float(i) for i in words[3:]),
        )

    def _dump(self, *args) -> list[str]:
        i, j, func_type, sigma, epsilon = args
        return [
            _at(i),
            _at(j),
            _f(func_type),
            _c(sigma),
            _c(epsilon),
        ]


class Cmaptypes(Table):
    """Section in topology .top/.itp file: [ cmaptypes ]."""

    i: list[str] = []
    j: list[str] = []
    k: list[str] = []
    l: list[str] = []
    m: list[str] = []

    f1: list[int] = []
    f2: list[int] = []
    f3: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        i, j, k, l, m, f1, f2, f3 = words[:8]
        return dict(
            i=i,
            j=j,
            k=k,
            l=l,
            m=m,
            f1=int(f1),
            f2=int(f2),
            f3=int(f3),
            C=tuple(float(i) for i in words[8:]),
        )

    def _dump(self, *args) -> list[str]:
        i, j, k, l, m, f1, f2, f3, C = args
        return [
            _at(i),
            _at(j),
            _at(k),
            _at(l),
            _at(m),
            _f(f1),
            _f(f2),
            _f(f3),
            *[_c(i) for i in C],
        ]


# Molecule
class MoleculeType(Table):
    """Section in topology .top/.itp file: [ moleculetype ]."""

    name: list[str] = []
    nrexcl: list[int] = []

    def _parse(self, words: list[str]) -> dict:
        name, nrexcl = words
        return dict(
            name=name,
            nrexcl=int(nrexcl),
        )

    def _dump(self, *args) -> list[str]:
        name, nrexcl = args
        return [
            f"{name: <25}",
            f"{nrexcl}",
        ]


class Atoms(Table):
    """Section in topology .top/.itp file: [ atoms ]."""

    ai: list[int] = []
    sequence: list[int] = []
    icode: list[str | None] = []
    residue: list[str] = []

    name: list[str] = []

    mass: list[float | None] = []
    charge: list[float | None] = []
    charge_group: list[int] = []

    type: list[str] = []

    typeB: list[str | None] = []
    massB: list[float | None] = []
    chargeB: list[float | None] = []

    def _parse(self, words: list[str]) -> dict:
        resnr_ = words[2]
        if resnr_.isnumeric() or (words[2].startswith("-") and words[2][1:].isnumeric()):
            resnr = resnr_
            icode = None
        else:
            resnr = words[2][:-1]
            icode = words[2][-1]
        nr, type_, _, residue, atom, cgnr = words[:6]
        data = words[6:8]
        charge, mass = data + [None] * (2 - len(data))
        data = words[8:]
        typeB, chargeB, massB = data + [None] * (3 - len(data))
        return dict(
            ai=int(nr),
            type=type_,
            sequence=int(resnr),
            icode=icode,
            residue=residue,
            name=atom,
            charge_group=int(cgnr),
            charge=float(charge) if charge else None,
            mass=float(mass) if mass else None,
            typeB=typeB,
            chargeB=float(chargeB) if chargeB else None,
            massB=float(massB) if massB else None,
        )

    def _dump(self, *args) -> list[str]:
        nr, seq, ic, res, at, ms, ch, cgnr, type_, type_B, msB, chB = args
        ic = ic or " "
        return [
            f"{nr: >5}",
            _at(type_),
            f"{seq: >6}{ic}",
            f"{res: >6}",
            _at(at),
            f"{cgnr: >5}",
            _c(ch),
            _c(ms),
            _at(type_B) if type_B else " " * 5,
            _c(chB),
            _c(msB),
        ]


class Bonds(Table):
    """Section in topology .top/.itp file: [ bonds ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type = words[:3]
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[3:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Pairs(Table):
    """Section in topology .top/.itp file: [ pairs ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type = words[:3]
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[3:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class PairsNB(Table):
    """Section in topology .top/.itp file: [ pairs_nb ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type = words[:3]
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[3:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Constraints(Table):
    """Section in topology .top/.itp file: [ constraints ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []
    b: list[float] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type, b = words
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
            b=float(b),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, func_type, b = args
        return [
            _a(ai),
            _a(aj),
            _f(func_type),
            _c(b),
        ]


class Angles(Table):
    """Section in topology .top/.itp file: [ angles ]."""

    ai: list[int] = []
    aj: list[int] = []
    ak: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, ak, func_type = words[:4]
        return dict(
            ai=int(ai),
            aj=int(aj),
            ak=int(ak),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[4:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, ak, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _a(ak),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Dihedrals(Table):
    """Section in topology .top/.itp file: [ dihedrals ]."""

    ai: list[int] = []
    aj: list[int] = []
    ak: list[int] = []
    al: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, ak, al, func_type = words[:5]
        return dict(
            ai=int(ai),
            aj=int(aj),
            ak=int(ak),
            al=int(al),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[5:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, ak, al, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _a(ak),
            _a(al),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Exclusions(Table):
    """Section in topology .top/.itp file: [ exclusions ]."""

    A: list[tuple[int, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        return dict(
            A=tuple(int(i) for i in words),
        )

    def _dump(self, *args) -> list[str]:
        return [_a(ai) if ai else " " * 5 for ai in args[0]]


class Settles(Table):
    """Section in topology .top/.itp file: [ settles ]."""

    OW: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        OW, func_type = words[:2]
        return dict(
            OW=int(OW),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[2:]),
        )

    def _dump(self, *args) -> list[str]:
        OW, func_type, C = args
        return [
            _a(OW),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Virtual_Sites1(Table):
    """Section in topology .top/.itp file: [ virtual_sites1 ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type = words[:3]
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, func_type = args
        return [
            _a(ai),
            _a(aj),
            _f(func_type),
        ]


class Virtual_Sites2(Table):
    """Section in topology .top/.itp file: [ virtual_sites2 ]."""

    ai: list[int] = []
    aj: list[int] = []
    ak: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, ak, func_type = words[:4]
        return dict(
            ai=int(ai),
            aj=int(aj),
            ak=int(ak),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[4:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, ak, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _a(ak),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Virtual_Sites3(Table):
    """Section in topology .top/.itp file: [ virtual_sites3 ]."""

    ai: list[int] = []
    aj: list[int] = []
    ak: list[int] = []
    al: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, ak, al, func_type = words[:5]
        return dict(
            ai=int(ai),
            aj=int(aj),
            ak=int(ak),
            al=int(al),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[5:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, ak, al, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _a(ak),
            _a(al),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Virtual_Sites4(Table):
    """Section in topology .top/.itp file: [ virtual_sites4 ]."""

    ai: list[int] = []
    aj: list[int] = []
    ak: list[int] = []
    al: list[int] = []
    af: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, ak, al, af, func_type = words[:6]
        return dict(
            ai=int(ai),
            aj=int(aj),
            ak=int(ak),
            al=int(al),
            af=int(af),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[6:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, ak, al, af, funct, C = args
        return [
            _a(ai),
            _a(aj),
            _a(ak),
            _a(al),
            _a(af),
            _f(funct),
            *[_c(c) for c in C],
        ]


# TODO: need examples to check
class VirtualSitesN(Table):
    """Section in topology .top/.itp file: [ virtual_sitesn ]."""

    ai: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, func_type = words[:2]

        return dict(
            ai=int(ai),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[2:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, _, func_type, C = args
        return [
            _a(ai),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Position_Restraints(Table):
    """Section in topology .top/.itp file: [ position_restraints ]."""

    ai: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, func_type = words[:2]
        return dict(
            ai=int(ai),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[2:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, func_type, C = args
        return [
            _a(ai),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Distance_Restraints(Table):
    """Section in topology .top/.itp file: [ distance_restraints ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []
    type: list[int] = []
    label: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type, type_, label = words[:5]
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
            type=int(type_),
            label=int(label),
            C=tuple(float(i) for i in words[5:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, func_type, type_, label, C = args
        return [
            _a(ai),
            _a(aj),
            _f(func_type),
            _f(type_),
            _f(label),
            *[_c(c) for c in C],
        ]


class Dihedral_Restraints(Table):
    """Section in topology .top/.itp file: [ dihedral_restraints ]."""

    ai: list[int] = []
    aj: list[int] = []
    ak: list[int] = []
    al: list[int] = []
    func_type: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, ak, al, func_type = words[:5]
        return dict(
            ai=int(ai),
            aj=int(aj),
            ak=int(ak),
            al=int(al),
            func_type=int(func_type),
            C=tuple(float(i) for i in words[5:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, ak, al, func_type, C = args
        return [
            _a(ai),
            _a(aj),
            _a(ak),
            _a(al),
            _f(func_type),
            *[_c(c) for c in C],
        ]


class Orientation_Restraints(Table):
    """Section in topology .top/.itp file: [ orientation_restraints ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []
    exp: list[int] = []
    label: list[int] = []
    C: list[tuple[float, ...]] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type, exp, label = words[:5]
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
            exp=int(exp),
            label=int(label),
            C=tuple(float(i) for i in words[5:]),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, funct, exp, label, C = args
        return [
            _a(ai),
            _a(aj),
            _f(funct),
            _f(exp),
            _f(label),
            _f(funct),
            *[_c(c) for c in C],
        ]


class Angle_Restraints(Table):
    """Section in topology .top/.itp file: [ angle_restraints ]."""

    ai: list[int] = []
    aj: list[int] = []
    ak: list[int] = []
    al: list[int] = []
    func_type: list[int] = []
    tetta: list[float] = []
    k: list[float] = []
    n: list[int] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, ak, al, func_type, tetta, k, n = words[:5]
        return dict(
            ai=int(ai),
            aj=int(aj),
            ak=int(ak),
            al=int(al),
            func_type=int(func_type),
            tetta=float(tetta),
            k=float(k),
            n=int(n),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, ak, al, func_type, tetta, k, n = args
        return [
            _a(ai),
            _a(aj),
            _a(ak),
            _a(al),
            _f(func_type),
            _c(tetta),
            _c(k),
            f"{n: >2}",
        ]


class Angle_RestraintsZ(Table):
    """Section in topology .top/.itp file: [ angle_restraints_z ]."""

    ai: list[int] = []
    aj: list[int] = []
    func_type: list[int] = []
    tetta: list[float] = []
    k: list[float] = []
    n: list[int] = []

    def _parse(self, words: list[str]) -> dict:
        ai, aj, func_type, tetta, k, n = words
        return dict(
            ai=int(ai),
            aj=int(aj),
            func_type=int(func_type),
            tetta=float(tetta),
            k=float(k),
            n=int(n),
        )

    def _dump(self, *args) -> list[str]:
        ai, aj, func_type, tetta, k, n = args
        return [
            _a(ai),
            _a(aj),
            _f(func_type),
            _c(tetta),
            _c(k),
            f"{n: >2}",
        ]


# System
class System(Table):
    """Section in topology .top/.itp file: [ system ]."""

    name: list[str] = []

    def _parse(self, words: list[str]) -> dict:
        return dict(name=" ".join(words))

    def _dump(self, *args) -> list[str]:
        name = args[0]
        return [name]


class Molecules(Table):
    """Section in topology .top/.itp file: [ molecules ]."""

    name: list[str] = []
    n_mols: list[int] = []

    def _parse(self, words: list[str]) -> dict:
        return dict(
            name=words[0],
            n_mols=int(words[1]),
        )

    def _dump(self, *args) -> list[str]:
        name, n_mols = args
        return [
            f"{name: <25}",
            f"{n_mols:>5}",
        ]


# Datas
class FFData(Data):
    """Data container for forcefield."""

    defaults: Defaults = Defaults()

    atomtypes: AtomTypes = AtomTypes()

    bondtypes: BondTypes = BondTypes()
    pairtypes: PairTypes = PairTypes()
    constrainttypes: ConstraintTypes = ConstraintTypes()

    angletypes: AngleTypes = AngleTypes()
    dihedraltypes: DihedralTypes = DihedralTypes()
    implicit_genborn_params: Implicit_Genborn_Params = Implicit_Genborn_Params()
    nonbond_params: Nonbond_Params = Nonbond_Params()

    cmaptypes: Cmaptypes = Cmaptypes()


class TopologyData(Data):
    """Data container for molecule topology."""

    moleculetype: MoleculeType = MoleculeType()

    atoms: Atoms = Atoms()

    bonds: Bonds = Bonds()
    pairs: Pairs = Pairs()
    pairs_nb: PairsNB = PairsNB()
    constraints: Constraints = Constraints()
    angles: Angles = Angles()
    dihedrals: Dihedrals = Dihedrals()

    settles: Settles = Settles()
    virtual_sites1: Virtual_Sites1 = Virtual_Sites1()
    virtual_sites2: Virtual_Sites2 = Virtual_Sites2()
    virtual_sites3: Virtual_Sites3 = Virtual_Sites3()
    virtual_sites4: Virtual_Sites4 = Virtual_Sites4()
    exclusions: Exclusions = Exclusions()

    position_restraints: Position_Restraints = Position_Restraints()
    distance_restraints: Distance_Restraints = Distance_Restraints()
    dihedral_restraints: Dihedral_Restraints = Dihedral_Restraints()
    orientation_restraints: Orientation_Restraints = Orientation_Restraints()
    angle_restraints: Angle_Restraints = Angle_Restraints()
    angle_restraints_z: Angle_RestraintsZ = Angle_RestraintsZ()


class SystemData(Data):
    """Data container for system information."""

    system: System = System()
    molecules: Molecules = Molecules()


# Top file structure
_title_pattern = re.compile(r"s*\[\s*(?P<title>\w+)\s*\].*")
"""Topology section regexp pattern."""


# TODO: intermolecular_interactions,virtual_sitesX, X_restraints
class TOPFile(BaseModel):
    """Data container for all topology and system information in .top/.itp file."""

    top_list: list[TopologyData] = []
    """Topology datas list."""

    ff: FFData = FFData()
    """Forcefield data."""

    system: SystemData = SystemData()
    """System data."""

    section: Data | None = None
    """Parser state."""

    def get_table(self, title_str: re.Match) -> Table:
        """Get table accordingly section and molecule type in .top/.itp file.

        :param title_str: String matched title pattern
        :return: Table
        """
        title = title_str.group("title")
        if title == "defaults":
            self.section = self.ff
        elif title in FFData.model_fields and not self.section:
            msg = f"Forcefield table without previous data: {title}"
            _log.warning(msg)
            self.section = self.ff

        elif title == "moleculetype":
            self.section = TopologyData()
            self.top_list.append(self.section)
        elif title in TopologyData.model_fields and not self.section:
            msg = f"Topology table without previous data: {title}"
            _log.warning(msg)
            self.section = TopologyData()
            self.top_list.append(self.section)

        elif title == "system":
            self.section = self.system
        elif title in SystemData.model_fields and not self.section:
            msg = f"System table without previous data: {title}"
            _log.warning(msg)
            self.section = self.system

        elif not self.section:
            msg = f"Incorrect section {self.section} for: {title}"
            raise ValueError(msg)
        return getattr(self.section, title)

    @classmethod
    def parse(cls, topology: Path) -> TOPFile:
        """Parse .top/.itp file.

        :param topology: Path to .top/.itp file
        :return: Data container for .top/.itp file
        """
        parser = TOPFile()
        block = Info()
        if topology.suffix not in [".top", ".itp"]:
            msg = "Only .top/.itp files are supported"
            raise ValueError(msg)

        line_text = ""
        with topology.open() as file:
            for i, line_ in enumerate(file):
                line = line_.strip()

                if line.endswith("\\"):
                    line_text = line_text + line.replace("\\", " ")
                    continue
                if not line_text:
                    line_text = line
                else:
                    line_text = line_text + line.replace("\\", " ")

                try:
                    content = line_text.split(";")[0]
                    if title_str := _title_pattern.search(content):
                        block = parser.get_table(title_str)
                    elif content.strip():
                        block.parse(content)
                    line_text = ""
                except Exception as e:
                    msg = f"Parsing error of in {topology.resolve()}:\nLine {i + 1: >6}: {line}"
                    raise ValueError(msg) from e
        return parser

    def dump(self) -> tuple[list[str], list[str], list[str]]:
        """Dump topology file container to list of strings in 3 sections:
        forcefield, list molecule topologies data, system data.

        :return: FF dump, list of topologies dump, system data dump
        """
        mol_dumps = ["\n".join(i for i in mol.dump() if i) for mol in self.top_list]
        return self.ff.dump(), mol_dumps, self.system.dump()


def _a(x: str) -> str:
    """Format atom id.

    :param x: Atom id
    :return: Formatted atomtype
    """
    return f"{x: >5}"


def _at(x: str) -> str:
    """Format atom type.

    :param x: Atom type name
    :return: Formatted name
    """
    return f"{x: >8}"


def _f(x: int) -> str:
    """Format function type.

    :param x: Function type number
    :return: Function type string
    """
    return f"{x:3}"


def _c(x: float | None) -> str:
    """Format parameters.

    :param x: Parameter float or None to empty yield
    :return: Formatted parameter
    """
    if x is None:
        return ""
    if x % 1:
        return f"{x: >12}"
    return f"{int(x): >12}"
