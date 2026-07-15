"""MD pipeline functions."""

from pathlib import Path

from simdel import chem, func

from . import mdps


# TODO: refactor
def create_box(  # noqa: PLR0913
    workdir: Path,
    system: chem.System,
    ff_type: chem.GromacsFF,
    water_type: chem.WaterType,
    positive_ion: chem.Ion,
    negative_ion: chem.Ion,
    flexible_water: bool = False,
) -> chem.System:
    """Create standard box.

    :param workdir: Workdir path
    :param system: Raw system
    :param ff_type: Forcefiled path
    :param water_type: Water type, must be compatible with forcefield
    :param flexible_water: Use flexible water, defaults to False
    :return: Ionic box
    """
    box = func.create_box(
        system=system,
        box_distance=1.5,
        workdir=workdir / "1_box",
    )
    solvated_box = func.solvate(
        system=box,
        ff=ff_type,
        water_type=water_type,
        flexible_water=flexible_water,
        workdir=workdir / "2_solvated",
    )
    ions_config = mdps.add_ions_mdp()
    return func.add_ions(
        system=solvated_box,
        parameters=ions_config,
        concentration=0.15,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        workdir=workdir / "3_ionic",
    )
