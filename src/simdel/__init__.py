"""A collection of Gromacs manipulation utilities and MD-based pipelines."""

import importlib.metadata as _meta

import lazyimports as _lazyimports

with _lazyimports.lazy_imports("simdel._wrappers", "plumed", "lomap"):
    from . import analyse, chem, func, run, sim, traj
    from ._utils import STRICT

__version__ = _meta.version("simdel")
