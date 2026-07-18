"""PLUMED function wrappers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plumed as plumed_adapter

from simdel import _utils

_utils.run("plumed check", ["plumed"])


def parse_plumed(
    fes: Path,
) -> tuple[
    pd.DataFrame,
    list[tuple[str, str | float | int]],
]:
    """Read FES .dat file.

    :param fes: FES .dat file
    :return: pandas.DataFrame with additional plumed fields
    """
    data = plumed_adapter.read_as_pandas(fes.as_posix())
    constants = [(key, value) for key, value, _ in data.plumed_constants]
    return (pd.DataFrame(data), constants)


def plumed_driver(
    plumed: Path,
    mf_xtc: Path,
    workdir: Path,
):
    """REQUIRE PLUMED!
    Process trajectory using plumed driver.

    :param plumed: PLUMED .dat config file path
    :param mf_xtc: Trajectory .xtc file path
    :param workdir: Workdir path
    """
    command = [
        "plumed driver",
        f"--plumed {plumed.resolve()}",
        f"--mf_xtc {mf_xtc.resolve()}",
    ]
    _utils.run(
        command=command,
        title=f"plumed driver {mf_xtc.name}",
        workdir=workdir,
    )


# TODO: sigma?
# TODO: refactor
def plumed_sum_hills(  # noqa: PLR0913
    workdir: Path,
    hills: Path,
    stride: int,
    outfile: str,
    kt: float = 2.494339,
    histro: Path | None = None,
    outhisto: str = "",
    negbias: bool = False,
    mintozero: bool = False,
    min: float | None = None,
    max: float | None = None,
    bin: int | None = None,
    spacing: float | None = None,
    sigma: float | None = None,
) -> list[Path]:
    """REQUIRE PLUMED!
    Integrate hills after metadynamics using plumed sum_hills.

    :param workdir: Workdir path
    :param hills: Hills .dat file path
    :param stride: Stride to integrate hills
    :param outfile: Out file .dat name
    :param kt: Value kB * T, defaults to 2.494339 (298 `K`)
    :param histro: Histogram .dat file path, defaults to None
    :param outhisto: Output histogram .dat file path, defaults to ""
    :param negbias: Use negative bias, defaults to False
    :param mintozero: Shift global minimum of all integrals to zero, defaults to False
    :param min: Minimum value, defaults to None
    :param max: Maximum value, defaults to None
    :param bin: Histogram bins, defaults to None
    :param spacing: Space between bins in histogram, defaults to None
    :param sigma:
    """
    command = [
        "plumed sum_hills",
        f"--hills {hills.resolve()}",
        f"--outfile {outfile}",
        f"--stride {stride}",
        f"--kt {kt}",
    ]

    if histro and not outhisto:
        msg = "Set outhisto option"
        raise ValueError(msg)
    elif outhisto and not histro:
        msg = "Set input histro file"
        raise ValueError(msg)

    if histro:
        command.append(f"--histro {histro}")

    if outhisto:
        command.append(f"--outhisto {outhisto}")

    if negbias:
        command.append("--negbias")

    if mintozero:
        command.append("--mintozero")

    if min:
        command.append(f"--min {min}")

    if max:
        command.append(f"--max {max}")

    if bin and spacing:
        msg = "Bin and spacing options can not set together"
        raise ValueError(msg)

    if bin:
        command.append(f"--bin {bin}")

    if spacing:
        command.append(f"--spacing {spacing}")

    if sigma:
        command.append(f"--sigma {sigma}")

    _utils.run(
        command=command,
        title=f"plumed sum_hills {hills.name}",
        workdir=workdir,
    )
    files = (i for i in workdir.iterdir() if (i.suffix == ".dat") and (outfile in i.stem))
    # TODO: another files
    return sorted(files, key=lambda x: int(x.stem.replace(outfile, "")))
