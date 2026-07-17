"""Library initialization functions."""

from __future__ import annotations

from collections.abc import Callable
import functools
import importlib.util
from pathlib import Path
import subprocess
from typing import Any

from . import log

_GMX_WARN = (
    "GROMACS is not installed.\n"
    "   Install gmx separately or using conda/mamba:\n"
    "   mamba install gromacs"
)
_PMX_WARN = (
    "PMX library is not installed: FEP functions is off.\n"
    "   Install pmx using pip or from source:\n"
    "   pip install pmx / cd pmx && pip install ."
)
_PLUMED_WARN = (
    "PLUMED plugin is not installed: metadynamics functions is off.\n"
    "   Install PLUMED plugin separately."
)
_MAMBA_WARN = (
    "Mamba dependencies are not installed: openff/lomap functions is off.\n"
    "   Install mamba dependencies:\n"
    "   mamba activate env && mamba install -f dep.yml"
)


def initialize_gmx() -> Path:
    """Check GMX path and GROMACS folders.

    :return: GMX dir
    """
    sp = subprocess.run("which gmx", capture_output=True, check=False, shell=True)
    if sp.returncode:
        raise ImportError(_GMX_WARN)
    return Path(sp.stdout.decode().strip()).parent.parent.resolve()


def initialize_pmx() -> Path | None:
    """Check pmx executable file.

    :return: PMX dir or None
    """
    sp = subprocess.run(
        "which pmx",
        capture_output=True,
        check=False,
        shell=True,
    )
    if sp.returncode:
        log.warning(_PMX_WARN)
        return None
    return Path(sp.stdout.decode()).resolve()


def initialize_plumed() -> Path | None:
    """Check plumed executable file.

    :return: PLUMED binary dir or None
    """
    sp = subprocess.run(
        "which plumed",
        capture_output=True,
        check=False,
        shell=True,
    )
    if sp.returncode:
        log.warning(_PLUMED_WARN)
        return None
    return Path(sp.stdout.decode().strip()).parent.parent.resolve()


def initialize_mamba() -> bool:
    """Check all mamba dependencies are installed.

    :return: Dependencies are installed or not
    """
    packages = ["openff", "lomap"]
    result = all(bool(importlib.util.find_spec(p)) for p in packages)
    if not result:
        log.warning(_MAMBA_WARN)
    return result


GMX = initialize_gmx()
"""GROMACS bin path."""

PLUMED = initialize_plumed()
"""PLUMED plugin bin path."""

PMX = initialize_pmx()
"""PMX lib path."""

_MAMBA_FLAG = initialize_mamba()
"""Install dependencies from mamba or not."""

# TODO: rework
STRICT: bool = False
"""Allow 5 warnings in gmx functions."""


def require_mamba(func):
    """Require for mamba dependencies installation."""
    return _require_generator(func, _MAMBA_WARN, _MAMBA_FLAG)


def require_pmx(func):
    """Require for install pmx python lib."""
    return _require_generator(func, _MAMBA_WARN, PMX)


def require_plumed(func):
    """Require for install PLUMED plugin for GROMACS and python plumed lib."""
    return _require_generator(func, _PLUMED_WARN, PLUMED)


@functools.singledispatch
def _require_generator(target: Callable | type, warn: str, flag: Any | None) -> Callable | type:
    """Require dependencies wrapper for functions and classes generator.

    :param target: Function/class
    :param warn: Warn text in dependency is not installed
    :param flag: Dependency install flag
    :return: Wrapped function/class
    """
    ...


@_require_generator.register
def _(target: type, warn: str, flag: Any | None) -> type:
    original_init = target.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        raise ImportError(warn)

    new_init.__doc__ = f"{'*' * 60}\nFUNCTIONALITY IS OFF!\n{warn}\n{'*' * 60}{new_init.__doc__}"
    if flag:
        target.__doc__ = target.__doc__ = (
            f"{'*' * 60}\nFUNCTIONALITY IS OFF!\n{warn}\n{'*' * 60}{target.__doc__}"
        )

    target.__init__ = target.__init__ if flag else new_init
    return target


@_require_generator.register
def _(target: Callable, warn: str, flag: Any | None):
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        raise ImportError(warn)

    wrapper.__doc__ = f"{'*' * 60}\nFUNCTIONALITY IS OFF!\n{warn}\n{'*' * 60}{wrapper.__doc__}"
    return target if flag else wrapper
