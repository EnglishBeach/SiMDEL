"""A collection of Gromacs manipulation utilities and MD-based pipelines."""

from . import analyse, chem, func, pipelines, run, sim, traj
from ._misc.context import GMX, PLUMED, PMX, STRICT
from .version import VERSION as __version__
