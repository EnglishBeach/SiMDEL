"""Parametrization functions."""

from pathlib import Path
import random
import shutil

from simdel import _log, _utils, chem
from simdel._wrappers import gmx, openff

from . import geometry_transform, topology_transform


# TODO: refactor
@_utils.require(gmx)
def parametrize_protein(  # noqa: PLR0913
    geometry: Path,
    ff: chem.GromacsFF,
    workdir: Path,
    name: str = "",
    water_type: chem.WaterType | None = None,
    fix_missing: bool = False,
) -> chem.System:
    """Parametrize protein by GROMACS, clear positional restraints.

    :param geometry: Geometry .pdb/.gro file path
    :param ff_type: Forcefield type
    :param workdir: Workdir path
    :param name: Protein name, defaults to geometry file name
    :param water_type: Water type for parametrize protein,
    must be compatible with forcefield, defaults to None
    :param fix_missing: Fix missing Hs by GROMACS, can be incorrect, defaults to False
    :return: Parametrized protein system
    """
    name = name or geometry.stem
    workdir.mkdir(parents=True, exist_ok=True)
    geo = workdir / geometry.name
    shutil.copy(geometry, geo)

    box_geo = gmx.editconf(
        workdir=workdir,
        geometry=geo,
        out_fname=f"{name}_fixed{geometry.suffix}",
        box_distance=1,
        center=False,
    )
    water_index, _ = ff.get_water_info(water_type)
    parametrized = gmx.pdb2gmx(
        workdir=workdir,
        ff_paths=ff.paths,
        out_name=name,
        geometry=box_geo,
        ff_name=ff.name,
        water_index=water_index,
        reset_H=True,
        fix_missing=fix_missing,
    )

    mdp_proxy = workdir / "mdp.mdp"
    _utils.backup(mdp_proxy)
    mdp_proxy.write_text("define =")
    preprocessed = gmx.grompp(
        workdir=workdir,
        geometry=parametrized.gro,
        top=parametrized.top,
        posres_geometry=parametrized.gro,
        mdp=mdp_proxy,
        out_name=name,
        merge=True,
        maxwarn=5 if not _utils.STRICT else 0,
    )

    system = (
        chem.System.load(
            top=preprocessed.top,
            gro=parametrized.gro,
        )
        .set_info(
            ff_type=ff,
            water_type=water_type,
            water_flexibility=False if water_type else None,
        )
        .rename(name)
    )
    cleared = topology_transform.clear_topologies(system)
    return geometry_transform.reset_box(cleared)


@_utils.require(openff)
def parametrize_small(
    sdf: Path,
    ff: chem.OpenFF,
    workdir: Path,
    name: str = "UNL",
    fast: bool = False,
) -> chem.System:
    """REQUIRE MAMBA DEPENDENCIES!
    Parametrize small molecule by openff.

    :param sdf: Small molecule .sdf file path
    :param ff_type: Forcefield type
    :param workdir: Workdir path
    :param name: Small molecule name, if len>4 - generate new name Lxxx, defaults to "UNL"
    :param fast: Use fast algorithm calculating partial charges `gasteiger`,
    instead of `am1bcc` (slow variant), defaults to True
    :return: Parametrized small molecule system
    """
    max_name_l = 4
    workdir = workdir or sdf.parent
    workdir.mkdir(parents=True, exist_ok=True)

    if len(name) > max_name_l:
        new_name = f"L{int(random.random() * 100)}"
        msg = f"Name too long, will be renamed to {new_name}"
        _log.warning(msg)
        name = new_name

    files = openff.parametrize(
        sdf=sdf,
        ff=ff.name,
        workdir=workdir,
        out_name=name,
        fast=fast,
    )
    s = (
        chem.System.load(top=files.top, gro=files.gro)
        .set_info(
            ff_type=ff,
            water_type=None,
            water_flexibility=None,
        )
        .rename(name)
    )
    return geometry_transform.reset_box(s)
