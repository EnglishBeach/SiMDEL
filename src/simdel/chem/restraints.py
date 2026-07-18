"""Restraints classes."""

from __future__ import annotations

import typing

import pandas as pd

from simdel import _utils

from . import topology, views

T = typing.TypeVar("T", bound="Restraints")


class Restraints(_utils.Table):
    """Base user defined restraints class."""

    _field_name: str

    def convert(self) -> dict[str, _utils.Table]:
        """Convert user defined restraints to the topology table for each molecule.

        :return: Topology restraints table map corresponding to user defined restraints,
        in format {molecule: Topology restraints table}
        """
        ...

    def clear(self: T) -> T:
        """Clear restraints table.

        :return: Cleared Restraints table
        """
        ...

    @classmethod
    def from_top_view(cls, view: views.TopologyView) -> Restraints:
        """Create configurable restraints table, read position restraints for system.

        :param view: System topology view
        :return: Restraints table
        """
        ...


class PositionRestraints(Restraints):
    """User defined position restraints. Correct parameters:
    ftype, x, y, z all is None or not None.
    """

    _field_name = "position_restraints"

    molecule: pd.Series[str]
    ai: pd.Series[int]
    ftype: pd.Series[int]
    x: pd.Series[float]
    y: pd.Series[float]
    z: pd.Series[float]

    @classmethod
    def from_top_view(cls, view: views.TopologyView) -> PositionRestraints:
        """Create configurable position restraints table, read position restraints for system.

        :param view: System topology view
        :return: Restraints table
        """
        data = pd.DataFrame(
            (i or (None, None, None, None) for i in view.posres_C.to_list()),
            columns=["ftype", "x", "y", "z"],
        )
        return PositionRestraints(molecule=view.molecule, ai=view.ai, **data)

    def convert(self) -> dict[str, _utils.Table]:
        """Convert positional restraints to the topology table for each molecule.

        :return: Topology restraints table {molecule: topology.PositionRestraints table}
        """
        # TODO: apply -> map
        df = pd.DataFrame(dict(ftype=self.ftype, x=self.x, y=self.y, z=self.z))
        if not (df.isna().sum(axis=1).apply(lambda x: x in [0, 4]).all()):
            msg = "Position restraints table has incorrect values"
            raise ValueError(msg)

        posres = {}
        for mol, mol_df_ in self.to_df().groupby("molecule"):
            mol_df = mol_df_.dropna()
            posres[str(mol)] = topology.PositionRestraints(
                ai=mol_df["ai"],
                func_type=mol_df["ftype"],
                C=list(mol_df[["x", "y", "z"]].to_records(index=False)),
            )

        return posres

    def clear(self) -> PositionRestraints:
        """Clear restraints table.

        :return: Cleared Restraints table
        """
        new = self.to_df()
        new.loc[:, ["ftype", "x", "y", "z"]] = None
        return PositionRestraints(**new)
