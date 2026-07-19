"""Calculate trajectory energy functions and classes."""

from pathlib import Path

from pydantic import BaseModel

from simdel import _deps
from simdel._wrappers import pmx


class AnalyzeResult(BaseModel):
    """Analyze trajectory output data."""

    dG: float
    """Mean transition dG, in `kJ/mol`."""

    analytical_error: float
    """Calculate dG analytical error, in `kJ/mol`."""

    bootstrap_error: float
    """Calculate dG bootstrap error, in `kJ/mol`."""


@_deps.require(pmx)
def analyze_dG(
    workdir: Path,
    xvgs_a: list[Path],
    xvgs_b: list[Path],
    temperature: int = 298,
    samples: int = 100,
) -> AnalyzeResult:
    """Parse result calculate_BAR data.

    :param workdir: Workdir path
    :param xvgs_a: List of .xvgs file paths of state A
    :param xvgs_b: List of .xvgs file paths of state B
    :param temperature: Temperature of trajectory analysis in `K`, defaults to 298
    :param samples: Calculate samples count from trajectory, defaults to 100
    :return: Data container
    """
    workdir.mkdir(parents=True, exist_ok=True)
    files = pmx.calculate_BAR(
        xvgs_a=xvgs_a,
        xvgs_b=xvgs_b,
        temperature=temperature,
        samples=samples,
        workdir=workdir,
    )
    with files.BAR.open("r") as file:
        lines = file.readlines()

    dG = analytical_error = bootstrap_error = 0
    for line_ in lines:
        line = line_.rstrip()
        parts = line.split()
        if "BAR: dG" in line:
            dG = float(parts[-2])
        elif "BAR: Std Err (analytical)" in line:
            analytical_error = float(parts[-2])
        elif "BAR: Std Err (bootstrap)" in line:
            bootstrap_error = float(parts[-2])

    if any(not i for i in [dG, analytical_error, bootstrap_error]):
        msg = "Error in dG calculation"
        raise ValueError(msg)
    dG: float
    analytical_error: float
    bootstrap_error: float

    return AnalyzeResult(
        dG=dG,
        analytical_error=analytical_error,
        bootstrap_error=bootstrap_error,
    )
