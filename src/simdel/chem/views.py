"""Selection, molecule separation tools and classes."""

from __future__ import annotations

import pandas as pd

from simdel._misc import utils


class TopView(utils.Table):
    """Only topology parameters view."""

    molecule: pd.Series[str]
    """Molecule name (topology name)."""

    ai: pd.Series[int]
    """Atom index."""

    sequence: pd.Series[int]
    """Sequence number."""

    icode: pd.Series[str]
    """Insertion code."""

    residue: pd.Series[str]
    """Residue name."""

    name: pd.Series[str]
    """Atom name."""

    type: pd.Series[str]
    """Atom type name."""

    mass: pd.Series[float]
    """Atom mass."""

    charge: pd.Series[float]
    """Atom charge."""

    charge_group: pd.Series[int]
    """Charge group."""

    typeB: pd.Series[str]
    """Hybrid atom type name."""

    massB: pd.Series[float]
    """Hybrid atom mass."""

    chargeB: pd.Series[float]
    """Hybrid atom charge."""

    posres_C: pd.Series[list]
    """Position restraints parameters: ftype, x, y, z."""


class View(TopView):
    """Base view class.

    Topology + forcefield parameters for atom
    """

    atomic_number: pd.Series[int]
    """Atomic number."""

    ff_ptype: pd.Series[str]
    """Particle type:A - atom, S - shell, V/D - virtual site."""

    ff_bonded_type: pd.Series[str]
    """Optional atom type name for bonded interactions."""

    ff_C: pd.Series[list]
    """The Lennard-Jones potential parameters"""


class TopologyView(View):
    """Pivot table for atom parameters for each atom in topologies
    (if atom parameter != atomtype parameter => take atom parameter).
    """


class GeometryView(View):
    """Pivot table for atom parameters for each atom in geometry
    (if atom parameter != atomtype parameter => take atom parameter).
    """

    molecule_n: pd.Series[int]
    """Molecule id (shared for all topologies)."""

    chain: pd.Series[str]
    """Chain letter."""

    x: pd.Series[float]
    """X coordinate."""

    y: pd.Series[float]
    """Y coordinate."""

    z: pd.Series[float]
    """Z coordinate."""

    vx: pd.Series[float]
    """Velocity in x coordinate."""

    vy: pd.Series[float]
    """Velocity in y coordinate."""

    vz: pd.Series[float]
    """Velocity in z coordinate."""
