"""Metadynamics classes and functions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel


class Funnel(BaseModel, arbitrary_types_allowed=True):
    """Funnel for funnel metadynamics."""

    center: np.ndarray
    """Funnel center xyz vector, in `nm`."""

    vector: np.ndarray
    """Funnel direction xyz vector, in `nm`."""

    anchor: int
    """Anchor atom id."""

    h: float
    """Funnel length, in `nm`."""

    cone_h: float
    """Cone length, in `nm`."""

    cone_angle: float
    """Cone angle tangens."""

    tube_r: float
    """Tube radius, in `nm`."""

    site_h: float
    """Site position on funnel axis from start, in `nm`."""

    @property
    def tube_h(self) -> float:
        """Tube length, in `nm`."""
        return self.h - self.cone_h

    def dump(self) -> str:
        """Dump funnel.

        :return str: Text
        """
        return json.dumps(dict(self), indent=4, default=_serialize)

    @classmethod
    def load(cls, file: Path) -> Funnel:
        """Load funnel from .json file.

        :param file: Funnel .json file path
        :return: Funnel
        """
        data = json.loads(file.read_text())
        # TODO: not save
        center = np.array(eval(data["center"]))
        vector = np.array(eval(data["vector"]))
        data["center"] = center
        data["vector"] = vector
        return Funnel(**data)

    # TODO: refactor
    @classmethod
    def create(  # noqa: PLR0913
        cls,
        anchor_id: int,
        center: np.ndarray,
        b: np.ndarray,
        total_h: float,
        tube_r: float,
        cone_angle: float,
        cone_h: float,
        site_h: float,
    ) -> Funnel:
        """Create funnel from points.

        :param anchor_id: Anchor atom id
        :param center: Cone basement xyz point, in `nm`
        :param b: Any direction xyz point, in `nm`
        :param total_h: Cone + tube funnel length, in `nm`
        :param tube_r: Tube radius, in `nm`
        :param cone_angle: Cone angle tangens
        :param cone_h: Cone length, in `nm`
        :param site_h: Site position on funnel axis from a point, in `nm`
        :return: Funnel
        """
        vector = np.array(b) - np.array(center)
        vector = vector / np.linalg.norm(vector)
        return Funnel(
            center=np.array(center),
            vector=vector,
            h=total_h,
            cone_angle=cone_angle,
            cone_h=cone_h,
            tube_r=tube_r,
            anchor=anchor_id,
            site_h=site_h,
        )

    def transform(self, matrix: np.ndarray) -> Funnel:
        """Transform funnel center and vector by general transform matrix M 4x4:
            |rot shift|
            |0   1    |
        For this matrix center vector 4x1: |x y z 1|
        Direction vector 4x1 (not react on translation): |x y z 0|
        Transformed coordinates M @ v = t 4x1.

        :param matrix: Rotation Matrix
        :return: Transformed funnel same size and form
        """
        # TODO: check only shift and rotation, det(M)==1
        center = np.array([*self.center, 1])
        vector = np.array([*self.center, 0])
        return Funnel(
            center=(matrix @ center)[:3],
            vector=(matrix @ vector)[:3],
            anchor=self.anchor,
            h=self.h,
            tube_r=self.tube_r,
            cone_h=self.cone_h,
            cone_angle=self.cone_angle,
            site_h=self.site_h,
        )


def get_boxes(
    funnel: Funnel,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Get boxes for funnel's cone and tube for draw_box script.

    :param funnel: Funnel
    :return: Box parameters
    """
    vec = np.array((-funnel.vector[1] / funnel.vector[0], 1, 0))
    vec = vec / np.linalg.norm(vec)
    vec2 = np.cross(funnel.vector, vec)
    vec2 = vec2 / np.linalg.norm(vec2)
    R = np.stack([vec, vec2, funnel.vector], axis=1)

    s = np.tan(funnel.cone_angle) * funnel.cone_h
    cone = np.array(
        [
            [s * 2, 0, 0],
            [0, s * 2, 0],
            [0, 0, funnel.cone_h],
        ]
    )
    cone_shift = np.array((-s, -s, 0))

    tube = np.array(
        [
            [funnel.tube_r * 2, 0, 0],
            [0, funnel.tube_r * 2, 0],
            [0, 0, funnel.h],
        ]
    )
    tube_shift = np.array((-funnel.tube_r, -funnel.tube_r, 0))

    return {
        "cone": (R @ cone, R @ cone_shift + funnel.center),
        "tube": (R @ tube, R @ tube_shift + funnel.center),
    }


def _serialize(x):
    """Serialize function to dump pipeline configs.

    :param x: Serializable object
    :return: Dump
    """
    try:
        return json.dumps(x)
    except Exception:
        return json.dumps(list(x))
