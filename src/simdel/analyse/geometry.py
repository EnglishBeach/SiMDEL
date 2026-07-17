"""Geometry analyze functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from simdel import chem


# TODO: to numpy
def find_box_distance(
    system: chem.System,
    center_point: tuple[float, float, float],
    direction_point: tuple[float, float, float],
) -> float:
    """Find distance from center to box borders in
    center_point->direction_point direction.

    :param system: System
    :param center_point: Start point coordinates, in `nm`
    :param direction_point: End point coordinates, set only direction, in `nm`
    :return: Distance in `nm`
    """
    X0 = np.array(center_point)
    X1 = np.array(direction_point)

    if system.geometry.box is None:
        msg = "System must be boxed"
        raise ValueError(msg)

    M = system.geometry.box.T
    inside, r = _get_box_distance(center=X0, direction=X1 - X0, box_matrix=M)
    if not inside:
        msg = "Center point outside system box"
        raise ValueError(msg)
    return r


def get_site_mask(
    view: chem.GeometryView,
    ligand_mask: pd.Series[bool],
    protein_mask: pd.Series[bool],
    search_radius: float = 0.3,
) -> pd.Series[bool]:
    """Get site using algorithm:
    1. Find atoms around ligand <= r
    2. Find all protein residues which have these atoms
    3. Site - mid point of residue nonH atoms.

    :param view: System geometry view
    :param ligand_mask: Ligand mask to search around
    :param protein_mask: Protein mask to exclude non-protein molecules
    :param search_radius: Search radius in `nm`, defaults to 0.3
    :return: Geometry view with site atoms
    """
    nonH_mass = 2

    ligand = view[ligand_mask & (view.mass > nonH_mass)]
    ligand_df = pd.DataFrame(dict(x=ligand.x, y=ligand.y, z=ligand.z))

    def _f(x: pd.DataFrame) -> float:
        return _find_distance_to_shield(
            point=x.to_numpy(),
            shield=ligand_df.to_numpy(),
        )

    protein = view[protein_mask]
    shield_mask = (
        pd.DataFrame(dict(x=protein.x, y=protein.y, z=protein.z)).agg(_f, axis=1) < search_radius
    )
    shield = protein[shield_mask]
    shield_df = pd.DataFrame(
        dict(
            chain=shield.chain,
            sequence=shield.sequence,
            residue=shield.residue,
        )
    )
    view_df = pd.DataFrame(
        dict(
            chain=view.chain,
            sequence=view.sequence,
            residue=view.residue,
        )
    )

    def _f2(x: pd.DataFrame) -> bool:
        site_residues = shield_df.agg(lambda s: ":".join(str(i) for i in s), axis=1)
        site_residues = set(site_residues)
        return ":".join(str(i) for i in x) in site_residues

    return view_df.agg(_f2, axis=1)


def _find_distance_to_shield(point: np.ndarray, shield: np.ndarray) -> float:
    """Get min distance from point to points set.

    :param x: Point xyz
    :param shield_xyz: Points set coordinates:
        |x1 y1 z1|
        |x2 y2 z2|
    :return: Min distance
    """
    dist = (shield - point) ** 2
    return (dist.sum(axis=1) ** (1 / 2)).min()


def _get_box_distance(
    center: np.ndarray,
    direction: np.ndarray,
    box_matrix: np.ndarray,
) -> tuple[bool, float]:
    """Get distance to box sides and inside/outside mark.

    :param center: Start point vector: |x y z|

    :param direction: Direction vector: |x y z|
    :param box_matrix: Box matrix:
        |Xx Yx Zx|
        |Xy Yy Zy|
        |Xz Yz Zz|
    :return: _description_
    """
    _round = np.vectorize(lambda x: round(x, 4))
    direction = direction / np.linalg.norm(direction)

    M_ = np.linalg.inv(box_matrix)
    center_c = _round(M_ @ center)
    direction_c = _round(M_ @ direction)

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = ((direction_c >= 0) - center_c) / direction_c

    inside = ((0 <= center_c) & (center_c <= 1)).all()
    return inside, round(min(rs), 4)
