"""Trajectory analysis and visualization functions."""

from __future__ import annotations

import enum

import numpy as np
from pydantic import BaseModel

from simdel import chem, traj

ATOL = 1e-7
"""Relative tolerance"""


class PiType(enum.Enum):
    """Pi-stacking interaction type."""

    NO = "no"
    """No pi-stacking."""

    T = "t-shaped"
    """T-shaped pi-stacking."""

    P = "parallel_shifted"
    """Parallel shifted pi-stacking."""

    R = "parallel_non-shifted"
    """Parallel non-shifted pi stacking (may be repulsive)."""


class PiStakingData(BaseModel, arbitrary_types_allowed=True):
    """Pi-stacking information."""

    pi_type: PiType
    """Pi-stacking interaction type."""

    ring_distance: float
    """Distance between ring centroids."""

    cos: float
    """Angle cos between ring 1 normal and ring 2 normal."""


# TODO: docstring
def get_pi_type(
    ring_dist: float,
    ring_shift: np.ndarray,
    ring_1_n: np.ndarray,
    ring_2_n: np.ndarray,
) -> PiType:
    """Determine stacking type from ring configuration.

    :param ring_dist: _description_
    :param ring_shift: _description_
    :param ring_1_n: _description_
    :param ring_2_n: _description_
    :return: _description_
    """
    ring_cos = np.dot(ring_1_n, ring_2_n)
    shift_cos = np.dot(ring_1_n, ring_shift)

    # Ring configuration constants
    # Distances
    ShiftD = (0.320, 0.450)  # nm
    ParallelD = (0.300, 0.400)
    TshapeD = (0.400, 0.600)
    # Angles
    ParallelA = (0.865, 1.000)  # cos
    TshapeA = (-0.500, 0.500)
    # Shifts
    ParallelSA = (0.995, 1.000)  # cos
    ShiftA = (0.900, 0.995)

    if _angle_in(ring_cos, ParallelA):
        if _angle_in(shift_cos, ParallelSA):
            if ParallelD[0] <= ring_dist <= ParallelD[1]:
                return PiType.R
        elif _angle_in(shift_cos, ShiftA) and (ShiftD[0] <= ring_dist <= ShiftD[1]):
            return PiType.P

    elif _angle_in(ring_cos, TshapeA) and (TshapeD[0] <= ring_dist <= TshapeD[1]):
        return PiType.T

    return PiType.NO


def analyze_pi_stacking(
    system: chem.System,
    trajectory: chem.Trajectory,
    ring_idx_1: list[int],
    ring_idx_2: list[int],
) -> list[PiStakingData]:
    """Analyze trajectory to determine if provided ring are involved in pi stacking.

    :param trajectory: Trajectory to analyze
    :param ring_idx_n: list if three core plane atom indexes in trajectory topology

    :return: an array[N_frames, 3] with pi-type (PiType value), ring distances
    and ring cos angles
    """
    ring_len = 3
    if len(ring_idx_1) != ring_len:
        msg = f"Ring 1 indexes must contain exactly 3 atom indexes, provided {ring_idx_1}"
        raise ValueError(msg)
    if len(ring_idx_2) != ring_len:
        msg = f"Ring 2 indexes must contain exactly 3 atom indexes, provided {ring_idx_2}"
        raise ValueError(msg)

    ring_dists = _get_ring_distances(
        system=system,
        trajectory=trajectory,
        ring_idx_1=ring_idx_1,
        ring_idx_2=ring_idx_2,
    )
    ring_shift = (
        _get_ring_centroids(
            system=system,
            trajectory=trajectory,
            atom_index_1=ring_idx_1[0],
            atom_index_2=ring_idx_1[1],
            atom_index_3=ring_idx_1[2],
        )
        - _get_ring_centroids(
            system=system,
            trajectory=trajectory,
            atom_index_1=ring_idx_2[0],
            atom_index_2=ring_idx_2[1],
            atom_index_3=ring_idx_2[2],
        )
    ) / ring_dists.reshape(-1, 1)

    ring_1_normals = _get_ring_normals(
        system=system,
        trajectory=trajectory,
        atom_index_1=ring_idx_1[0],
        atom_index_2=ring_idx_1[1],
        atom_index_3=ring_idx_1[2],
    )
    ring_2_normals = _get_ring_normals(
        system=system,
        trajectory=trajectory,
        atom_index_1=ring_idx_2[0],
        atom_index_2=ring_idx_2[1],
        atom_index_3=ring_idx_2[2],
    )

    staking_data = []
    for rdist, rshift, r1n, r2n in zip(
        ring_dists, ring_shift, ring_1_normals, ring_2_normals, strict=True
    ):
        staking_data.append(
            PiStakingData(
                pi_type=get_pi_type(rdist, rshift, r1n, r2n),
                ring_distance=rdist,
                cos=np.dot(r1n, r2n),
            )
        )
    return staking_data


def _angle_in(value: float, interval: tuple) -> bool:
    return (interval[0] <= value <= interval[1]) or (-interval[0] <= value <= -interval[1])


def _plane_normal(vecs: np.ndarray) -> np.ndarray | None:
    """Calculate normal vector from 3x3 array of core plane points.

    :param vecs: Cone plane vectors
    :return: Normal vector
    """
    vec_a = vecs[1] - vecs[0]
    vec_b = vecs[2] - vecs[0]
    vec_p = np.cross(vec_a, vec_b)

    norm_p = np.linalg.norm(vec_p)
    if norm_p > (np.linalg.norm(vec_a) * np.linalg.norm(vec_a) * ATOL):
        return vec_p / norm_p


def _get_ring_normals(
    system: chem.System,
    trajectory: chem.Trajectory,
    atom_index_1: int,
    atom_index_2: int,
    atom_index_3: int,
) -> np.ndarray:
    """Calculate normal vectors of provided triangles.

    :param trajectory: Trajectory to analyze Trajectory
    :param atom_index_1: Index of 1 atom in trajectory topology
    :param atom_index_2: Index of 2 atom in trajectory topology
    :param atom_index_3: Index of 3 atom in trajectory topology
    :return: Array[N_frames, 3] with normal vector per frame
    """
    trj_coords = traj.extract_cords(
        system=system,
        trajectory=trajectory,
        atom_idxs=[atom_index_1, atom_index_2, atom_index_3],
    )
    norm_vecs = []
    for cords in trj_coords:
        norm = _plane_normal(cords)
        if norm is not None:
            norm_vecs.append(norm)
        else:
            msg = (
                f"Atoms [{atom_index_1}, {atom_index_2}, {atom_index_3}] are lying on one line "
                "and do not form a plane."
            )
            raise ValueError(msg)
    return np.stack(norm_vecs)


def _get_ring_centroids(
    system: chem.System,
    trajectory: chem.Trajectory,
    atom_index_1: int,
    atom_index_2: int,
    atom_index_3: int,
) -> np.ndarray:
    """Calculate centroids of provided triangles.

    :param trajectory: Trajectory to analyze Trajectory
    :param atom_index_1: Index of 1 atom in trajectory topology
    :param atom_index_2: Index of 2 atom in trajectory topology
    :param atom_index_3: Index of 3 atom in trajectory topology
    :return: Array[N_frames, 3] with centroids
    """
    trj_coords = traj.extract_cords(
        system=system,
        trajectory=trajectory,
        atom_idxs=[atom_index_1, atom_index_2, atom_index_3],
    )
    return np.mean(trj_coords, axis=1)


def _get_ring_distances(
    system: chem.System,
    trajectory: chem.Trajectory,
    ring_idx_1: list[int],
    ring_idx_2: list[int],
) -> np.ndarray:
    """Calculate distances between centroids of provided triangles.

    :param trajectory: Trajectory to analyze Trajectory
    :param ring_idx_1: List of 3 first core plane atom indexes in trajectory topology
    :param ring_idx_2: List of 3 second core plane atom indexes in trajectory topology
    :return: Array[N_frames] with ring-ring distances
    """
    ring_len = 3
    if len(ring_idx_1) != ring_len:
        msg = f"Ring 1 indexes must contain exactly 3 atom indexes, provided {ring_idx_1}"
        raise ValueError(msg)
    if len(ring_idx_2) != ring_len:
        msg = f"Ring 2 indexes must contain exactly 3 atom indexes, provided {ring_idx_2}"
        raise ValueError(msg)

    return np.linalg.norm(
        _get_ring_centroids(
            system=system,
            trajectory=trajectory,
            atom_index_1=ring_idx_1[0],
            atom_index_2=ring_idx_1[1],
            atom_index_3=ring_idx_1[2],
        )
        - _get_ring_centroids(
            system=system,
            trajectory=trajectory,
            atom_index_1=ring_idx_2[0],
            atom_index_2=ring_idx_2[1],
            atom_index_3=ring_idx_2[2],
        ),
        axis=1,
    )


def _get_angle_cosines(
    system: chem.System,
    trajectory: chem.Trajectory,
    ring_idx_1: list[int],
    ring_idx_2: list[int],
) -> np.ndarray:
    """Calculate cosines of angles between provided plane triangles.

    :param trajectory: Trajectory to analyze Trajectory
    :param ring_idx_1: List of 3 first core plane atom indexes in trajectory topology
    :param ring_idx_2: List of 3 second core plane atom indexes in trajectory topology
    :return: an array[N_frames] with ring-ring cosines
    """
    ring_len = 3
    if len(ring_idx_1) != ring_len:
        msg = f"Ring 1 indexes must contain exactly 3 atom indexes, provided {ring_idx_1}"
        raise ValueError(msg)
    if len(ring_idx_2) != ring_len:
        msg = f"Ring 2 indexes must contain exactly 3 atom indexes, provided {ring_idx_2}"
        raise ValueError(msg)

    normals_1 = _get_ring_normals(
        system=system,
        trajectory=trajectory,
        atom_index_1=ring_idx_1[0],
        atom_index_2=ring_idx_1[1],
        atom_index_3=ring_idx_1[2],
    )
    normals_2 = _get_ring_normals(
        system=system,
        trajectory=trajectory,
        atom_index_1=ring_idx_2[0],
        atom_index_2=ring_idx_2[1],
        atom_index_3=ring_idx_2[2],
    )
    cosines = []
    for norm1, norm2 in zip(normals_1, normals_2, strict=True):
        cosines.append(np.dot(norm1, norm2))
    return np.asarray(cosines)
