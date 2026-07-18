"""System geometry definition and internal classes."""

from __future__ import annotations

import enum
import json

import numpy as np
import pandas as pd
import pydantic
from pydantic import BaseModel

from simdel import _utils
from simdel._parsers import gro_parser

_bool = np.vectorize(bool)
"""Custom bool for vectors"""


class Geometry(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """System geometry."""

    box: np.ndarray | None
    """Box matrix:
        |X_x X_y X_z|
        |Y_x Y_y Y_z|
        |Z_x Z_y Z_z|

    Sides: X, Y, Z - system cell vectors
    Axis: Ox Oy Oz - absolute coordinate grid
    Projection X on Oy = X_y
    In rectangle: Ox||X, Oy||Y, Oz||Z
    """

    box_type: BoxType | None
    """Box type."""

    coordinates: Coordinates
    """Atom coordinates."""

    @pydantic.model_validator(mode="after")
    def _validate_box(self):
        """Validate box parameters.

        GROMACS supports:
        X parallel to Ox (X_y=X_z=0) and
        Y perpendicular to Oz (Y_z=0):
            |X_x   0   0|
            |Y_x Y_y   0|
            |Z_x Z_y Z_z|

        :param values: Parameters
        :return: Parameters
        """
        if self.box is None:
            return self

        M = self.box
        # TODO: fix types
        non_diagonal = M[~np.eye(3, dtype=bool)]  # type: ignore

        if (not _bool(M.diagonal()).all()) and _bool(non_diagonal).any():
            msg = "Zero diagonal with non-zero non-diagonal box is not allowed. Rebuild box"
            raise ValueError(msg)

        align_mark = np.array(
            [
                M[0, 1],
                M[0, 2],
                M[1, 2],
            ]
        )
        if _bool(align_mark).any():
            msg = (
                f"Gromacs supports only X||Ox and Y_|_Oz aligned boxes "
                f"X_y=X_z=Y_z=None(0): {align_mark}"
            )
            raise ValueError(msg)

        if self.box_type == BoxType.Cubic and _bool(non_diagonal).any():
            msg = "Cubic box must have only non-zero diagonal elements"
            raise ValueError(msg)
        # TODO: add another box type validation?
        return self

    @property
    def molecule_ids(self) -> list[int]:
        """Molecule ids list."""
        return list(self.coordinates.molecule_id.unique())

    def __add__(self, geometry: Geometry) -> Geometry:
        """Sum geometries, reset boxes.

        :param geometry: Other geometry
        :return: New geometry
        """
        df = geometry.coordinates.to_df()
        df.loc[:, "molecule_id"] = df["molecule_id"] + max(self.coordinates.molecule_id) + 1
        # TODO: to numpy
        coordinates = Coordinates(
            **pd.concat([self.coordinates.to_df(), df]).reset_index(drop=True)
        )
        return Geometry(box=None, box_type=None, coordinates=coordinates)

    def __repr__(self) -> str:
        return f"<Geometry box={self.box is None} {len(self.coordinates)} atoms>"

    def __hash__(self) -> int:
        return hash(
            (
                json.dumps(self.box),
                self.box_type,
                hash(self.coordinates),
            )
        )

    @classmethod
    def load(
        cls,
        atom_counts: list[int],
        gro_data: gro_parser.GROFile,
        box_type: BoxType | None = None,
    ) -> Geometry:
        """Create geometry from gro data and atom counts.
        Ignore box type if box is not exists.

        :param atom_counts: List of atom counts in topologies in system molecules order
        :param gro_data: GRO parser data
        :param box_type: Box type, ignore if no box
        :return: Geometry
        """
        molecule_id = []
        for i, n_atoms in enumerate(atom_counts):
            molecule_id.extend([i] * n_atoms)

        if len(molecule_id) != len(gro_data.atoms.x):
            msg = "Num atoms does not match with num molecules"
            raise ValueError(msg)

        coordinates = Coordinates.load(
            atoms=gro_data.atoms,
            chain=[None] * len(gro_data.atoms.x),
            molecule_id=molecule_id,
        )
        box = cls._load_box(box=gro_data.box) if gro_data.box else None
        return Geometry(box=box, box_type=box_type, coordinates=coordinates)

    def dump(
        self,
        atom_names: list[str],
        residues: list[str],
        sequences: list[int],
    ) -> list[str]:
        """Dump geometry to .gro file format.

        :param atom_names: Atom name list for each atom in geometry
        :param residues: Residue list for each atom in geometry
        :param sequences: Sequence list for each atom in geometry
        :return: List of strings
        """
        atoms = self.coordinates.dump(
            atom_names=atom_names,
            residues=residues,
            sequences=sequences,
        )
        box_data = self._dump_box()
        return gro_parser.GROFile(atoms=atoms, box=box_data).dump()

    def sort(self, molecules_map: dict[int, int]) -> Geometry:
        """Sort coordinates by molecule id map, save atoms order.

        :param molecules_map: Molecule index map {old_id:new_id}
        :return: Sorted geometry
        """
        df = self.coordinates.to_df()
        df.loc[:, "molecule_id"] = self.coordinates.molecule_id.map(molecules_map)
        coordinates = Coordinates(**df.sort_values("molecule_id", kind="stable"))
        return Geometry(
            box=self.box,
            box_type=self.box_type,
            coordinates=coordinates,
        )

    def unbox(self) -> Geometry:
        """Remove box.

        :return: Unboxed geometry
        """
        return Geometry(
            box=None,
            box_type=self.box_type,
            coordinates=self.coordinates,
        )

    def extract(self, index: pd.Series[bool]) -> Geometry:
        """Extract geometry, save box.

        :param index: Extraction masks
        :return: Extracted geometry
        """
        molecules_ids = self.coordinates.molecule_id[index].unique()
        molecules_map = dict(zip(molecules_ids, range(len(molecules_ids)), strict=True))

        coords = self.coordinates[index]
        df = coords.to_df()
        df.loc[:, "molecule_id"] = coords.molecule_id.map(molecules_map)
        return Geometry(
            box=self.box,
            box_type=self.box_type,
            coordinates=Coordinates(**df),
        )

    def transform(self, matrix: np.ndarray) -> Geometry:
        """Transform coordinates by general transform matrix M 4x4:
            |rot shift|
            |0   1    |
        For this matrix coordinates vector v 4x1: |x1 y1 z1 1|.T
        Transformed coordinates M @ v = t 4x1
        Reset box.

        (Really pandas.DataFrame table + ones is matrix rV nx4:
            |x1 y1 z1 1|
            |x2 y2 z2 1|
        So real transformed coordinates pandas.DataFrame is (M @ rV.T).T = rV @ M.T)

        :param rotation: Rotation Matrix
        :return: Transformed geometry
        """
        X = (
            pd.DataFrame(dict(x=self.coordinates.x, y=self.coordinates.y, z=self.coordinates.z))
            .to_numpy()
            .astype(float)
        )
        V = (
            pd.DataFrame(
                dict(vx=self.coordinates.vx, vy=self.coordinates.vy, vz=self.coordinates.vz)
            )
            .to_numpy()
            .astype(float)
        )
        coords = np.column_stack([X, np.ones(len(self.coordinates))])
        velocities = np.column_stack([V, np.zeros(len(self.coordinates))])

        new_coords = coords @ matrix.T
        new_vels = velocities @ matrix.T
        coordinates = Coordinates(
            molecule_id=self.coordinates.molecule_id,
            chain=self.coordinates.chain,
            x=new_coords[:, 0],
            y=new_coords[:, 1],
            z=new_coords[:, 2],
            vx=new_vels[:, 0],
            vy=new_vels[:, 1],
            vz=new_vels[:, 2],
        )
        return Geometry(
            box=None,
            box_type=None,
            coordinates=coordinates,
        )

    def check_transform(self, matrix: np.ndarray) -> bool:
        """Check outside atoms after geometry transformation.

        :param matrix: General transform matrix M 4x4, see .transform method to details
        :return: Correct transform or not
        """
        if self.box is None:
            return True

        new_geo = self.transform(matrix=matrix)
        box_M = np.array(self.box)

        coords = np.array([new_geo.coordinates.x, new_geo.coordinates.y, new_geo.coordinates.z])
        coeffs = np.linalg.inv(box_M) @ coords

        return all((0 <= coeffs) & (coeffs <= 1))

    def rescale_box(
        self,
        a_scale: float = 1,
        b_scale: float = 1,
        c_scale: float = 1,
    ) -> Geometry:
        """Rescale box.

        :param a_scale: Scale factor for a side, defaults to 1
        :param b_scale: Scale factor for b side, defaults to 1
        :param c_scale: Scale factor for c side, defaults to 1
        :return: Geometry with rescaled box
        """
        if self.box is None:
            msg = "Geometry without box"
            raise ValueError(msg)

        s = np.array((a_scale, b_scale, c_scale))[:, None]
        return Geometry(
            box=self.box * s,
            box_type=self.box_type,
            coordinates=self.coordinates,
        )

    @classmethod
    def _load_box(
        cls,
        box: gro_parser.Box,
    ) -> np.ndarray:
        """Load box from gro parser data.

        :param box: Gro parser box
        :param box_type: Box type or None
        :return: Box
        """
        (X_x, Y_y, Z_z, X_y, X_z, Y_x, Y_z, Z_x, Z_y) = [
            i[0] if i[0] is not None else 0 for i in dict(box).values()
        ]
        return np.array(
            (
                (X_x, X_y, X_z),
                (Y_x, Y_y, Y_z),
                (Z_x, Z_y, Z_z),
            )
        )

    def _dump_box(self) -> gro_parser.Box:
        """Dump box to gro parser data.

        :return: Gro parser box
        """
        if self.box is None:
            return gro_parser.Box()
        (
            (X_x, X_y, X_z),
            (Y_x, Y_y, Y_z),
            (Z_x, Z_y, Z_z),
        ) = self.box

        return gro_parser.Box(
            x1=[X_x],
            y2=[Y_y],
            z3=[Z_z],
            z2=[Y_z],
            x3=[Z_x],
            y3=[Z_y],
            y1=[X_y],
            z1=[X_z],
            x2=[Y_x],
        )


class BoxType(enum.Enum):
    """Box types."""

    Dodecahedron = "dodecahedron"
    """Dodecahedron box."""

    Triclinic = "triclinic"
    """Triclinic box."""

    Cubic = "cubic"
    """Cubic box."""

    Octahedron = "octahedron"
    """Octahedron box."""


class Coordinates(_utils.Table):
    """Atom coordinates data table variant for geometry."""

    molecule_id: pd.Series[int]
    """Molecule index in System.molecules."""

    chain: pd.Series[str]
    """Chain number."""

    x: pd.Series[float]
    """X coordinate in `nm`."""

    y: pd.Series[float]
    """Y coordinate in `nm`."""

    z: pd.Series[float]
    """Z coordinate in `nm`."""

    vx: pd.Series[float]
    """Velocity in x coordinate in `nm/ps`."""

    vy: pd.Series[float]
    """Velocity in y coordinate in `nm/ps`."""

    vz: pd.Series[float]
    """Velocity in z coordinate in `nm/ps`."""

    # TODO: to geometry methods
    @classmethod
    def load(
        cls,
        molecule_id: list[int],
        chain: list[str | None],
        atoms: gro_parser.Atoms,
    ) -> Coordinates:
        """Load coordinates from gro parser data.

        :param molecule_id: Molecule id list
        :param chain: Chain name
        :param atoms: Gro parser atoms
        :return: Coordinates
        """
        return Coordinates(
            molecule_id=molecule_id,
            chain=chain,
            x=atoms.x,
            y=atoms.y,
            z=atoms.z,
            vx=atoms.vx,
            vy=atoms.vy,
            vz=atoms.vz,
        )

    def dump(
        self,
        atom_names: list[str],
        residues: list[str],
        sequences: list[int],
    ) -> gro_parser.Atoms:
        """Dump coordinates to gro parser data.

        :param atom_names: Atom name list
        :param residues: Residue list
        :param sequences: Sequence list
        :return: Gro parser Atoms
        """
        if not (len(atom_names) == len(residues) == len(sequences)):
            msg = "All lists mus have same length: atom_names, residues, sequences"
            raise ValueError(msg)
        # TODO: apply->map
        return gro_parser.Atoms(
            serial=pd.Series(self.index + 1).apply(lambda x: x % 100_000).to_list(),
            name=atom_names,
            resName=residues,
            resSeq=sequences,
            x=self.x.to_list(),
            y=self.y.to_list(),
            z=self.z.to_list(),
            vx=self.vx.to_list(),  # type: ignore
            vy=self.vy.to_list(),  # type: ignore
            vz=self.vz.to_list(),  # type: ignore
        )
