"""Dependency check module."""

from __future__ import annotations

from collections.abc import Callable
import functools
from typing import Any

from simdel._wrappers import gmx, openff, plumed, pmx

_REQUIRE_STATUS: dict[Any, None | bool] = {i.__name__: None for i in [gmx, openff, plumed, pmx]}


def require(*modules):
    """Mark function/class to depends on wrapper modules."""

    def require_middle(target):
        module_map = {module.__name__.split(".")[-1]: module for module in modules}

        if isinstance(target, Callable):
            func = target

        elif isinstance(target, type):
            func = target.__init__
        else:
            msg = "Require decorator only for functions and classes"
            raise TypeError(msg)

        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            checks = {name: _check_module(module) for name, module in module_map.items()}
            errors = {name: ok for name, ok in checks.items() if not ok}

            if errors:
                groups_text = " ".join(f"simdel[{name}]" for name in errors)
                msg = f"Install {groups_text}"
                raise ImportError(msg)

            return target(*args, **kwargs)

        require_text = " ".join(f"simdel[{m_name}]" for m_name in module_map)
        _wrapper.__doc__ = f"Require groups: {require_text}\n{func.__doc__}"

        if isinstance(target, Callable):
            return _wrapper

        if isinstance(target, type):
            target.__init__ = _wrapper
            return target

    if any(i.__name__ not in _REQUIRE_STATUS for i in modules):
        msg = "Only simdel._wrapper modules would be required."
        raise ModuleNotFoundError(msg)
    return require_middle


def _check_module(module) -> bool:  # noqa: ANN001
    mname = module.__name__
    if _REQUIRE_STATUS[mname] is None:
        try:
            module.__doc__  # noqa: B018
            _REQUIRE_STATUS[mname] = True
        except Exception:
            _REQUIRE_STATUS[mname] = False
    return _REQUIRE_STATUS[mname]  # type: ignore
