"""System view generators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simdel import _log, chem

# TODO: del _wrappers.gromacs


# Basic
def select_molecules(view: chem.View, *, molecule_names: list[str]) -> pd.Series[bool]:
    """Select molecules.

    :param view: Any system view
    :param molecule_names: Molecules
    :return: Selection mask
    """
    return view.molecule.map(lambda v: v in molecule_names)


def select_H(view: chem.View) -> pd.Series[bool]:
    """Select all Hs."""
    return view.atomic_number == 1


def select_C(view: chem.View) -> pd.Series[bool]:
    """Select all Cs."""
    return view.atomic_number == 6  # noqa: PLR2004


def select_heavy_atoms(view: chem.View) -> pd.Series[bool]:
    """Select all heavy (atomic_number >= 6)."""
    return view.atomic_number >= 6  # noqa: PLR2004


# Geometry
def select_sphere(
    view: chem.GeometryView,
    *,
    center: tuple[float, float, float],
    distance: float,
    within: bool = True,
) -> pd.Series[bool]:
    """Select sphere around center.

    :param view: System geometry view
    :param center: Sphere center
    :param distance: Radius in `nm`
    :param within: Within sphere or not, defaults to True
    :return: Selection mask
    """
    if isinstance(view, chem.TopologyView):
        msg = "Only Geometry view is allowed"
        raise TypeError(msg)
    val1 = (
        np.power(view.x - center[0], 2)
        + np.power(view.y - center[1], 2)
        + np.power(view.z - center[2], 2)
    )
    val2 = pow(distance, 2)
    if within:
        mask = val1 <= val2
    else:
        mask = val1 > val2
    return pd.Series(mask)


def get_geometric_center(
    view: chem.GeometryView,
) -> tuple[float, float, float]:
    """Get system box center from geometry view.

    :param view: System geometry view
    :return: Center coordinates
    """
    if isinstance(view, chem.TopologyView):
        msg = "Only Geometry view is allowed"
        raise TypeError(msg)
    return (
        float(view.x.mean()),
        float(view.y.mean()),
        float(view.z.mean()),
    )


# Complex
def select_total(view: chem.View) -> pd.Series[bool]:
    """Select all system."""
    return ~view.name.isna()


def select_solvent(
    view: chem.View,
    *,
    solvent_mols: list[str] | None = None,
    include_hydrogens: bool = True,
) -> pd.Series[bool]:
    """Select solvent molecules: SOL NA CL.

    :param view: Any system view
    :param solvent_mols: Solvent mol names, defaults to None (SOL, NA, CL)
    :param include_hydrogens: Include Hs or not, defaults to True
    :return: Selection mask
    """
    if solvent_mols is None:
        solvent_mols = ["SOL", "NA", "CL"]
    mask = select_molecules(view=view, molecule_names=solvent_mols)
    if not include_hydrogens:
        mask *= ~select_H(view=view)
    return mask


# Templates
class BaseSelectionGroups:
    """General selections, used in pipelines."""

    protein = "Protein"
    nonProtein = "non_Protein"
    system = "System"


# Restraints
def set_constant_posres(
    view: chem.TopologyView,
    selection: pd.Series[bool],
    value: float,
) -> chem.PositionRestraints:
    """Create simple constant positional restraints.

    :param view: System topology view
    :param selection: Mask to positional restraints
    :param value: Positional restraints value to fx, fy, fz
    :return: Positional restraints object
    """
    posres = chem.PositionRestraints.from_top_view(view)
    cols = ["x", "y", "z"]

    selected = posres[selection]
    df = pd.DataFrame(dict(x=selected.x, y=selected.y, z=selected.z))
    overwrite_mask = df.isna().any(axis=1)
    if overwrite_mask.any():
        indexes = list(overwrite_mask[~overwrite_mask].index)
        l = len(overwrite_mask)
        max_len = 10
        msg = f"Older position restraints will be overwritten in: {indexes if l <= max_len else l}"
        _log.warning(msg)

    posres_df = posres.to_df()
    posres_df.loc[selection, cols] = (value, value, value)  # type: ignore
    posres_df.loc[selection, "ftype"] = 1
    return chem.PositionRestraints(**posres_df)


class SelectionGroups:
    """General selections, used in pipelines."""

    protein = "Protein"

    nonProtein = "non_Protein"

    system = "System"
