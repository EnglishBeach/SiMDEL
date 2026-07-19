"""A collection of Gromacs manipulation utilities and MD-based pipelines."""

from importlib.metadata import version

import lazyimports

with lazyimports.lazy_imports("simdel._wrappers", "plumed", "lomap"):
    from . import analyse, chem, func, run, sim, traj
    from ._utils import STRICT

__version__ = version("simdel")
