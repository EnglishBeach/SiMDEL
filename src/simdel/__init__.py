"""A collection of Gromacs manipulation utilities and MD-based pipelines."""

import lazyimports

with lazyimports.lazy_imports("simdel._wrappers", "plumed", "lomap"):
    from . import analyse, chem, func, run, sim, traj
    from ._utils import STRICT
    from .version import VERSION as __version__
