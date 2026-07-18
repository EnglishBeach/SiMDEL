"""High level geometry manipulation functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simdel import _utils, chem, func
from simdel._wrappers import gmx


@_utils.require(gmx)
def create_box(
    system: chem.System,
    workdir: Path,
    box_type: chem.BoxType | None = None,
    box_distance: float = 1.0,
    center: bool = True,
) -> chem.System:
    """Create box.

    :param system: System to create box
    :param workdir: Workdir path
    :param box_type: Box type, defaults to automatic
    :param box_distance: Gap between protein and wall, defaults to 1.0, in `nm`
    :return: System with box
    """
    workdir.mkdir(parents=True, exist_ok=True)
    files = system.save(workdir)

    box_gro = gmx.editconf(
        workdir=workdir,
        geometry=files.gro,
        out_fname=f"{system.name}.gro",
        box_type=box_type.value if box_type else None,
        box_distance=box_distance,
        center=center,
    )
    return chem.System.load(
        top=files.top,
        gro=box_gro,
    ).set_info(**dict(system.info))


def axial_align(
    system: chem.System,
    vector: np.ndarray,
    axis: int,
    center: np.ndarray | None = None,
) -> chem.System:
    """Align the system with the same transformation as
    the alignment of a vector along a specific axis.

    :param system: System to align
    :param vector: Aligning vector to get transformation matrix
    :param axis: Axis to align: 0 - Ox, 1 - Oy, 2 - Oz
    :param center: Rotation center, defaults to None (auto):
     - (0,0,0) when system has no box
     - box center - when system has no box
    :return: Aligned system
    """
    vector = vector / np.linalg.norm(vector)

    vec = np.array((-vector[1] / vector[0], 1, 0))
    vec = vec / np.linalg.norm(vec)

    vec2 = np.cross(vector, vec)

    basis = [vec, vec2]
    basis.insert(axis, vector)

    rotation = np.linalg.inv(np.stack(basis, axis=1))
    R = np.eye(4)
    R[:3, :3] = rotation

    # Shift
    if not center:
        center = np.array((0, 0, 0))
    S = np.array(
        [
            [1, 0, 0, -center[0]],
            [0, 1, 0, -center[1]],
            [0, 0, 1, -center[2]],
            [0, 0, 0, 1],
        ]
    )
    S_ = np.linalg.inv(S)

    return transform_geometry(system=system, matrix=S_ @ R @ S)


def transform_geometry(system: chem.System, matrix: np.ndarray) -> chem.System:
    """Transform system's geometry by 4x4 general linear transformation matrix M:
    R - rotation 3x3 matrix
    T - translation 3x1 vector
        |R3x3 T3x1|
        |0    1   |
    For coordinate 4x1 column-vector v: |x y z 1|.T
    Transformed 4x1 column-vector: M @ v.

    :param system: System to transform
    :param matrix: Linear transformation matrix
    :return: System with transformed geometry
    """
    if not system.geometry.check_transform(matrix=matrix):
        msg = (
            "Some atoms go outside the box after transformation. "
            "Or transformation is no translation/rotation. "
            "Reset or rebuilt box before transformation or change matrix."
        )
        raise ValueError(msg)
    return chem.System(
        name=system.name,
        forcefield=system.forcefield,
        topology_map=system.topology_map,
        molecules=system.molecules,
        geometry=system.geometry.transform(matrix),
        index=system.index,
        info=system.info,
    )


def reset_box(system: chem.System) -> chem.System:
    """Remove box.

    :param system: System to reset box
    :return: Unboxed system
    """
    info_dict = dict(system.info)
    info_dict |= dict(box_type=None)
    return chem.System(
        name=system.name,
        forcefield=system.forcefield,
        topology_map=system.topology_map,
        molecules=system.molecules,
        geometry=system.geometry.unbox(),
        index=system.index,
        info=chem.SystemInfo(**info_dict),
    )


def rescale_box(
    system: chem.System,
    a_scale: float = 1,
    b_scale: float = 1,
    c_scale: float = 1,
) -> chem.System:
    """Rescale box size. Rescale only X_x, Y_y, Z_z (projections on axis) - only cubic boxes.

    :param system: System to rescale box
    :param a_scale: Scale factor of X_x, defaults to 1
    :param b_scale: Scale factor of Y_y, defaults to 1
    :param c_scale: Scale factor of Z_z, defaults to 1
    :return: System with rescaled box
    """
    new_geometry = system.geometry.rescale_box(
        a_scale=a_scale,
        b_scale=b_scale,
        c_scale=c_scale,
    )
    return chem.System(
        name=system.name,
        forcefield=system.forcefield,
        topology_map=system.topology_map,
        molecules=system.molecules,
        geometry=new_geometry,
        index=system.index,
        info=system.info,
    )


@_utils.require(gmx)
def create_gromacs_indexes(system: chem.System, workdir: Path) -> dict[str, pd.Series[bool]]:
    """Create all selections by GROMACS.

    :param system: System to create indexes
    :param view: System view
    :param workdir: Workdir path
    :return: All gromacs indexes dict
    """
    workdir.mkdir(parents=True, exist_ok=True)
    system_files = system.save(workdir)

    index_file = gmx.make_ndx(
        geometry=system_files.gro,
        out_name="index",
        workdir=workdir,
        groups=[],
    )
    view = system.geometry_view
    indexes = func.load_index(index_file=index_file, n_atoms=len(view))
    return {k.replace("-", "_"): v for k, v in indexes.items()}
