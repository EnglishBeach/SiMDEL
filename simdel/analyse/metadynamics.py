"""Calculate trajectory energy functions and classes."""

from __future__ import annotations

import collections
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import plumed
import pydantic

from simdel._misc import context
from simdel._wrappers import plumed as md_plumed

R: float = 8.314
"""Ideal Gas Constant, in `J/(mol K)`."""


class CVData(pydantic.BaseModel):
    """CV data."""

    min: float
    """Minimum cv value."""

    max: float
    """Maximum value."""

    nbins: int
    """CV frequency"""

    periodic: bool
    """Periodic or not"""

    @property
    def d(self) -> float:
        """Min - max diapason."""
        return (self.max - self.min) / self.nbins


class FES(pydantic.BaseModel, arbitrary_types_allowed=True):
    """Free energy space in time."""

    df: pd.DataFrame
    """Free, cv1, cv2... table"""

    time: float
    """Time, in `ps`"""

    cvs: dict[str, CVData]
    """Cv dict"""

    @classmethod
    def load(cls, fes: Path, time: float) -> FES:
        """Load free energy surface.

        :param fes: Free energy surface file
        :param time: Time. In `ns`
        :return FES: Free energy surface object
        """
        df = plumed.read_as_pandas(fes.as_posix()).rename(columns={"file.free": "free"})
        cv_data_ = collections.defaultdict(dict)

        for var, value_, _ in df.plumed_constants:
            property_, cv = var.split("_")
            value = value_.replace("pi", "3.141592654") if "pi" in str(value_) else value_
            cv_data_[cv][property_] = value

        return FES(
            time=time,
            df=df,
            cvs={k: CVData(**v) for k, v in cv_data_.items()},
        )


@context.require_plumed
def integrate_hills(
    hills: Path,
    T: float,
    dt: float,
    stride: int,
    workdir: Path,
) -> list[FES]:
    """REQUIRE PLUMED!
    Integrate hills.

    :param hills: Hills .dat file
    :param T: Temperature, in `K`
    :param dt: Time step in hills.dat file, in `ps`
    :param stride: Integration stride
    :param workdir: Workdir path
    :return: Hill integrals DataFrame without plumed metadata
    """
    workdir.mkdir(parents=True, exist_ok=True)
    prefix = "fes"

    hills_file = Path(shutil.copy(src=hills, dst=workdir / hills.name))
    integrals = md_plumed.plumed_sum_hills(
        workdir=workdir,
        hills=hills_file,
        kt=T * R / 1000,
        stride=stride,
        outfile=prefix,
    )
    return [
        FES.load(file, time=int(file.stem.replace(prefix, "")) * stride * dt) for file in integrals
    ]


def calculate_funnel_dG(
    fes: FES,
    T: float,
    tube_r: float,
    bound: dict[str, tuple[float, float]],
    unbound: dict[str, tuple[float, float]],
) -> float:
    """Calculate dG in view of funnel bias.

    :param fes: Last (more realistic) dG(cv) surface
    :param T: Temperature, in `K`
    :param tube_r: Funnel tube radius, in `nm`
    :param bound: Bound region range {cv: cv_start, cv_end}
    :param unbound: Unbound region range {cv: cv_start, cv_end}
    :param C0: Complex standard concentration, in `M`, defaults to 0.6020
    :return: dG
    """
    kT = T * 8.314 / 1000
    # 1 mol on 1660 A^3 -> 1/1660 A^3 * (10 A/nm)^3
    C0 = 1 / 1660 * 1000

    if not (len(bound) == len(unbound) == len(fes.cvs)):
        msg = "CV number must be same in fes, bind, unbind"
        raise ValueError(msg)

    bind_mask = np.ones(len(fes.df)).astype(bool)
    unbind_mask = bind_mask.copy()

    for cv, bound_data in bound.items():
        d = fes.cvs[cv].d
        x = fes.df[cv]

        b0, b1 = bound_data
        u0, u1 = unbound[cv]

        bind_mask = bind_mask * ((b0 - d) <= x) & (x <= (b1 + d))
        unbind_mask = unbind_mask * ((u0 - d) <= x) & (x <= (u1 + d))

    w_unbind = fes.df["free"][unbind_mask].mean()
    w = fes.df["free"][bind_mask]
    K = np.exp(-(w - w_unbind) / kT).sum()
    for cv in fes.cvs.values():
        K *= cv.d

    return -kT * np.log(K * np.pi * (tube_r**2) * C0)


def calculate_hysteresis_transitions(x: np.ndarray, eps: float = 0.1) -> int:
    """Calculate transitions x through 0 by hysteresis method.
    Works well to fractal function with fast fluctuations.

    :param x: X series
    :param eps: Relative error, defaults to 0.1
    :return: Transitions number
    """
    min_points = 2
    if len(x) < min_points:
        return 0

    state = 0
    crossings = []

    for i in range(len(x)):
        xi = x[i]
        if (state == 0) & (xi > eps):
            state = 1
        elif (state == 0) & (xi < -eps):
            state = -1
        elif state == 0:
            pass
        elif (state == 1) & (xi < -eps):
            crossings.append(i)
            state = -1
        elif (state == -1) & (xi > eps):
            crossings.append(i)
            state = 1

    crossing_indices = np.array(crossings, dtype=int)
    directions = np.sign(x[crossing_indices])
    return int(np.sum(directions > 0))


# TODO: get plateau more correctly
def analyse_time_series(
    t: np.ndarray,
    y: np.ndarray,
    tail_range: float,
    abs_error: float,
) -> tuple[float, float, float, float]:
    """Analyse time series with big fluctuations at the start and correlated data.
    Find plateau and do BSEM.

    :param t: Time, in `ps`
    :param y: Y(t)
    :param tail_range: Tail range from time end, in `ps`
    :param abs_error: Absolute error to find plateau, in Y(t) units
    :return: Plateau start time, plateau start y, block sem, block sem relative standard error
    """
    min_points = 2
    t_plateau, y_plateau = _get_plateau(
        x=t,
        y=y,
        tile=tail_range,
        abs_error=abs_error,
    )

    t_mask = t >= t_plateau
    if sum(t_mask) < min_points:
        return t[-1], y[-1], np.nan, np.nan

    tb, yb = _calculate_bsem(t[t_mask], y[t_mask])
    if len(tb) < min_points:
        return t[-1], y[-1], np.nan, np.nan
    tb_plateau, _ = _get_plateau(tb, yb, abs_error=yb.mean() * 0.5, tile=0)

    tb_mask = tb > tb_plateau

    if sum(tb_mask) < min_points:
        return t[-1], y[-1], np.nan, np.nan

    yb = yb[tb_mask]
    tb = tb[tb_mask]

    error = yb.mean()
    cv = yb.std() / error

    return t_plateau, y_plateau, error, cv


def _get_plateau(
    x: np.ndarray,
    y: np.ndarray,
    tile: float,
    abs_error: float,
) -> tuple[float, float]:
    """Get plateau for y(x) series.

    :param x: X values
    :param y: Y(x) values
    :param tile: Tile range from end, in x units
    :param abs_error: Absolute error
    :return: Plateau x start value, y mean on plateau
    """
    dx = x[1] - x[0]
    y_ref = (y[-int(tile / dx) :]).mean()
    mask = abs(y - y_ref) < abs_error
    plateau_i = np.concatenate([mask[::-1], [False]]).argmin()
    if plateau_i == len(mask):
        x_p = float(x[0] - dx)
    else:
        x_p = float(x[::-1][plateau_i])
    return x_p, y[x >= x_p].mean()


def _calculate_bsem(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate BSEM.

    :param x: X values
    :param y: Y(x) values
    :return: Block series, SEM series
    """
    min_blocks = 2
    n = len(y)
    dx = x[1] - x[0]

    block_sizes = np.arange(1, n // 2 + 1)
    sems = []
    for bs in block_sizes:
        n_blocks = n // bs
        if n_blocks < min_blocks:
            sems.append(np.nan)
            continue
        blocks = y[: n_blocks * bs].reshape(n_blocks, bs)
        block_means: np.ndarray = blocks.mean(axis=1)
        sem = block_means.std(ddof=1) / np.sqrt(n_blocks)
        sems.append(sem)
    return block_sizes * dx, np.array(sems)
