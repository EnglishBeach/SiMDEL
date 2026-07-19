"""High level assembly system and simulation functions."""

from pathlib import Path

from simdel import _deps, _utils, chem, sim
from simdel._wrappers import gmx

from . import converters, geometry_transform, topology_transform

_PROXY_NAME = "_PROXY"
_PROXY_GEOM = """_PROXY
 14
   32ASN      N    7   7.028   7.097   4.312
   32ASN     CA    9   7.043   6.961   4.364
   32ASN     CB   11   6.916   6.924   4.446
   32ASN     CG   14   6.928   6.793   4.525
   32ASN    OD1   15   7.037   6.753   4.565
   32ASN    ND2   16   6.815   6.727   4.550
   32ASN      C   19   7.069   6.863   4.249
   32ASN      O   20   6.975   6.821   4.182
   33SER      N   21   7.198   6.828   4.231
   33SER     CA   23   7.251   6.732   4.133
   33SER     CB   25   7.402   6.719   4.159
   33SER     OG   28   7.472   6.832   4.112
   33SER      C   30   7.183   6.593   4.146
   33SER      O   31   7.196   6.534   4.254
  11.48460  10.08590   9.47140
"""


@_deps.require(gmx)
def solvate(
    system: chem.System,
    ff: chem.GromacsFF,
    water_type: chem.WaterType,
    flexible_water: bool,
    workdir: Path,
) -> chem.System:
    """Add water molecules to prepared system (with box).

    :param system: System with box to solvate
    :param ff_type: Forcefield type
    :param water_type: Water type
    :param flexible_water: Use flexible water
    :param workdir: Workdir path
    :return: Solvated system
    """
    workdir.mkdir(parents=True, exist_ok=True)

    water_index, water_geometry = ff.get_water_info(water_type)

    proxy_system = _gen_water_system(
        workdir=workdir,
        ff_type=ff,
        water_type=water_type,
        flexible_water=flexible_water,
        water_id=water_index,
    )

    mixed = topology_transform.add_topologies(
        target=system,
        source=proxy_system,
        topologies=list(set(proxy_system.topology_map) - {_PROXY_NAME}),
    )
    system_dump = mixed.save(workdir)

    solvate_files = gmx.solvate(
        workdir=workdir,
        geometry=system_dump.gro,
        top=system_dump.top,
        out_name=system.name,
        solvent_geometry=water_geometry,
    )
    return (
        chem.System.load(
            top=solvate_files.top,
            gro=solvate_files.gro,
        )
        .set_info(
            ff_type=ff,
            water_type=water_type,
            water_flexibility=flexible_water,
        )
        .rename(system.name)
    )


# TODO: refactor
@_deps.require(gmx)
def resolvate(
    system: chem.System,
    water_type: chem.WaterType,
    flexible_water: bool,
    workdir: Path,
) -> chem.System:
    """Change water if new water type is convenient with
    forcefield type and old water geometry.

    :param system: Solvated system
    :param water_type: New water type
    :param flexible_water: Use flexible water
    :param workdir: Workdir path
    :return: Solvated system with new solvation environment
    """
    workdir.mkdir(parents=True, exist_ok=True)

    ff_type = system.info.ff_type
    if not isinstance(ff_type, chem.GromacsFF):
        msg = f"System must have GROMACS forcefield, not: {ff_type}"
        raise TypeError(msg)

    if not system.info.water_type:
        msg = "System must be solvated"
        raise ValueError(msg)

    _, old_water_geometry = ff_type.get_water_info(system.info.water_type)
    water_id, water_geometry = ff_type.get_water_info(water_type)

    if old_water_geometry != water_geometry:
        msg = f"Old and new water type must have same geometry: {old_water_geometry}"
        raise ValueError(msg)

    proxy_system = _gen_water_system(
        workdir=workdir,
        water_type=water_type,
        flexible_water=flexible_water,
        ff_type=ff_type,
        water_id=water_id,
    )
    mixed = topology_transform.replace_topologies(
        target=system, source=proxy_system, topologies=["SOL"]
    )
    return mixed.set_info(
        ff_type=ff_type,
        water_type=water_type,
        water_flexibility=flexible_water,
    )


@_deps.require(gmx)
def add_ions(  # noqa: PLR0913
    system: chem.System,
    concentration: float,
    parameters: sim.GromacsSimulator,
    positive_ion: chem.Ion,
    negative_ion: chem.Ion,
    workdir: Path,
    neutralize: bool = True,
) -> chem.System:
    """Add ions into geometry, ignore existing ions.

    :param system: Solvated system
    :param concentration: Ions concentration, in `mol/l` (only for added ions)
    :param parameters: Ionization parameters
    :param positive_ion: Positive ion type
    :param negative_ion: Negative ion type
    :param workdir: Workdir path
    :param neutralize: Neutralize system, defaults to True
    :return: System with ions
    """
    workdir.mkdir(parents=True, exist_ok=True)

    if "SOL" not in system.topology_map:
        msg = "System contain no 'SOL' topology"
        raise ValueError(msg)

    if positive_ion.name not in system.topology_map:
        msg = f"System contain no '{positive_ion.name}' topology"
        raise ValueError(msg)
    if positive_ion.charge <= 0:
        msg = "Positive ion charge must be > 0"
        raise ValueError(msg)

    if negative_ion.name not in system.topology_map:
        msg = f"System contain no '{negative_ion.name}' topology"
        raise ValueError(msg)
    if negative_ion.charge >= 0:
        msg = "Negative ion charge must be < 0"
        raise ValueError(msg)

    system_dump = system.save(workdir)
    mdp_file = parameters.save(workdir)

    preprocessed = gmx.grompp(
        workdir=workdir,
        geometry=system_dump.gro,
        top=system_dump.top,
        mdp=mdp_file,
        posres_geometry=system_dump.gro,
        out_name="ionic",
        merge=True,
        maxwarn=5 if not _utils.STRICT else 0,
    )

    sol_index = workdir / "sol.ndx"
    _utils.backup(sol_index)
    converters.dump_index(
        index_file=sol_index,
        indexes=dict(SOL=system.geometry_view.molecule.map(lambda v: v in ["NA", "CL", "SOL"])),
    )
    ionic_data = gmx.genion(
        workdir=workdir,
        tpr=preprocessed.tpr,
        top=preprocessed.top,
        sol_index=sol_index,
        positive_ion=positive_ion.name,
        positive_charge=positive_ion.charge,
        negative_ion=negative_ion.name,
        negative_charge=negative_ion.charge,
        neutral=neutralize,
        concentration=concentration,
        out_name=system.name,
    )
    preprocessed.tpr.unlink()

    return chem.System.load(
        top=ionic_data.top,
        gro=ionic_data.gro,
    ).set_info(**dict(system.info))


def _gen_water_system(
    workdir: Path,
    water_type: chem.WaterType,
    flexible_water: bool,
    ff_type: chem.GromacsFF,
    water_id: int,
) -> chem.System:
    """Generate water and ions topologies.

    :param workdir: Workdir path
    :param water_type: Water type
    :param flexible_water: Flexible water flag
    :param ff_type: FF type
    :param water_id: Water index
    :return: System with water and ion topologies
    """
    proxy_gro = workdir / f"{_PROXY_NAME}.gro"
    _utils.backup(proxy_gro)
    proxy_gro.write_text(_PROXY_GEOM)
    parametrized_proxy = gmx.pdb2gmx(
        workdir=workdir,
        ff_paths=ff_type.paths,
        geometry=proxy_gro,
        ff_name=ff_type.name,
        water_index=water_id,
        reset_H=True,
        out_name=_PROXY_NAME,
        fix_missing=True,
    )
    mdp_proxy = workdir / "mdp.mdp"
    posres_water = "-DFLEXIBLE" if flexible_water else ""
    _utils.backup(mdp_proxy)
    mdp_proxy.write_text(f"define = {posres_water}")
    preprocessed_proxy = gmx.grompp(
        workdir=workdir,
        geometry=parametrized_proxy.gro,
        top=parametrized_proxy.top,
        posres_geometry=parametrized_proxy.gro,
        mdp=mdp_proxy,
        out_name=_PROXY_NAME,
        merge=True,
        maxwarn=5 if not _utils.STRICT else 0,
    )
    preprocessed_proxy.tpr.unlink()
    s = (
        chem.System.load(
            top=preprocessed_proxy.top,
            gro=parametrized_proxy.gro,
        )
        .set_info(
            ff_type=ff_type,
            water_type=water_type,
            water_flexibility=flexible_water,
        )
        .rename_topologies(name_map={"Protein": _PROXY_NAME})
    )
    return geometry_transform.reset_box(s)
