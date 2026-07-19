"""Trajectory class."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import mdtraj
from pydantic import BaseModel

from simdel import _utils

from . import system as system_


class EnergyDump(_utils.PathContainer):
    """Energy files path container."""

    edr: Path | None = None
    """Energy data .edr file path."""

    xvg: Path | None = None
    """Optional analyze data .xvg file path."""


# TODO: write vel, f, write freq...
class Trajectory(BaseModel):
    """Trajectory linked to trajectory file."""

    file: Path
    """Trajectory .xtc/.trr file path."""

    dt: float
    """Time step, in `ps`."""

    frames: int
    """Number of frames."""

    @property
    def name(self):
        """Trajectory name."""
        return self.file.stem

    def get_mdtraj(self, system: system_.System) -> mdtraj.Trajectory:
        """Get mdtraj.Trajectory object.

        :param system: Trajectory system
        :return: mdtraj.Trajectory object
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            system_dump = system.save(Path(temp_dir))
            return mdtraj.load(self.file, top=system_dump.gro)

    def replace(self, destination_dir: Path):
        """Replace trajectory file, update object.

        :param destination_dir: Destination dir path
        """
        traj = destination_dir / self.file.name
        if traj != self.file:
            _utils.backup(traj)
            self.file = self.file.replace(traj)

    def copy(self, destination_dir: Path) -> Trajectory:  # type: ignore
        """Copy trajectory and trajectory files to another directory.

        :param destination_dir: Destination dir path
        :return: New trajectory
        """
        destination_dir.mkdir(parents=True, exist_ok=True)
        traj = destination_dir / self.file.name
        if traj != self.file:
            _utils.backup(traj)
            shutil.copy(src=self.file, dst=traj)
        return Trajectory(
            file=traj,
            dt=self.dt,
            frames=self.frames,
        )

    def remove_file(self):
        """Remove trajectory file."""
        self.file.unlink()
