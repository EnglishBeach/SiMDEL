"""System topology manipulation functions."""

from __future__ import annotations

from frozendict import frozendict
import pandas as pd

from simdel import chem


def clear_topologies(
    system: chem.System,
    topologies: list[str] | None = None,
) -> chem.System:
    """Delete topologies which does not present in geometry.

    :param system: System to remove topologies in it
    :param topologies: Topologies list to delete them, defaults to all
    :return: System without topologies
    """
    topologies = topologies or []
    in_molecules = set(system.molecules)
    if set(topologies) & in_molecules:
        msg = f"Geometry contains some topologies: {topologies}"
        raise ValueError(msg)

    ex_molecules = set(topologies) if topologies else (set(system.topology_map) - in_molecules)
    topology_map = {
        top_name: top
        for top_name, top in system.topology_map.items()
        if top_name not in ex_molecules
    }

    ex_atomtypes = set()
    for top_name in ex_molecules:
        ex_atomtypes.update(system.topology_map[top_name].atomtypes)
    in_atomtypes = set()
    for top in topology_map.values():
        in_atomtypes.update(top.atomtypes)

    ff_atomtypes = system.info.ff_type.atomtypes if system.info.ff_type else set()
    forcefield = system.forcefield.clear_atomtypes(ex_atomtypes - in_atomtypes - ff_atomtypes)
    return chem.System(
        name=system.name,
        forcefield=forcefield,
        topology_map=frozendict(topology_map),
        molecules=system.molecules,
        geometry=system.geometry,
        index=system.index,
        info=system.info,
    )


def add_topologies(
    target: chem.System,
    source: chem.System,
    topologies: list[str],
) -> chem.System:
    """Add topologies from source to target system.

    :param target: Target system to add topologies
    :param source: Source system with topologies
    :param topologies: Topologies to add
    :return: System with additional topologies
    """
    add_atomtypes = set()
    for i in source.topology_map.values():
        if i.name in topologies:
            add_atomtypes.update(i.atomtypes)
    ff_atomtypes = source.info.ff_type.atomtypes if source.info.ff_type else set()
    add_forcefield = source.forcefield.clear_atomtypes(
        set(source.forcefield.atomtypes.type) - add_atomtypes - ff_atomtypes
    )
    new_forcefield = target.forcefield + add_forcefield

    # TODO: refactor
    topology_map = target._mix_topologies(  # noqa: SLF001
        frozendict({i: source.topology_map[i] for i in topologies})
    )

    return chem.System(
        name=target.name,
        forcefield=new_forcefield,
        topology_map=topology_map,
        molecules=target.molecules,
        geometry=target.geometry,
        index=target.index,
        info=target.info,
    )


def replace_topologies(
    target: chem.System,
    source: chem.System,
    topologies: list[str],
) -> chem.System:
    """Replace topologies with same name from source to target system.
    Add atomtypes if need.

    :param target: Target system with topologies which will be replaced
    :param source: Source system with source topologies
    :param topologies: Topology names to replace
    :return: System with replaced topologies
    """
    add_atomtypes = set()
    for i in source.topology_map.values():
        if i.name in topologies:
            add_atomtypes.update(i.atomtypes)

    ff_atomtypes = source.info.ff_type.atomtypes if source.info.ff_type else set()
    add_forcefield = source.forcefield.clear_atomtypes(
        set(source.forcefield.atomtypes.type) - add_atomtypes - ff_atomtypes
    )
    new_forcefield = target.forcefield + add_forcefield

    topology_map = dict(target.topology_map)
    for top_name in topologies:
        self_top = topology_map[top_name]
        system_top = source.topology_map[top_name]

        if len(self_top.atoms) != len(system_top.atoms):
            msg = "Old and new topologies have different number of atoms"
            raise ValueError(msg)
        topology_map.update({top_name: system_top})

    return chem.System(
        name=target.name,
        forcefield=new_forcefield,
        topology_map=frozendict(topology_map),
        molecules=target.molecules,
        geometry=target.geometry,
        index=target.index,
        info=target.info,
    )


def extract_subsystem(
    system: chem.System,
    index: pd.Series[bool],
) -> chem.System:
    """Extract subsystem by mask. Splitting molecules is not allowed.

    Independently extracted molecules inherits
    - molecule topologies, which permanent for forcefield
    - forcefield, water types
    - atomtypes from forcefield

    :param system: Source system
    :param index: Extract mask
    :return: Extracted subsystem
    """
    in_mol_ids = system.geometry.coordinates.molecule_id[index].unique()
    ex_mol_ids = system.geometry.coordinates.molecule_id[~index].unique()
    if set(in_mol_ids) & set(ex_mol_ids):
        msg = "Splitting molecule is not allowed"
        raise ValueError(msg)

    ff_molecules = system.info.ff_type.moleculetypes if system.info.ff_type else set()
    in_molecules = {system.molecules[i] for i in in_mol_ids}
    ex_molecules = {system.molecules[i] for i in ex_mol_ids} - ff_molecules - in_molecules
    s = chem.System(
        name=system.name,
        topology_map=system.topology_map,
        forcefield=system.forcefield,
        geometry=system.geometry.extract(index),
        molecules=tuple(system.molecules[i] for i in in_mol_ids),
        index=frozendict(),
        info=system.info,
    )
    return clear_topologies(system=s, topologies=list(ex_molecules))


def set_restraints(
    system: chem.System,
    restraints: chem.Restraints,
) -> chem.System:
    """Set restraints to topologies in system. Clear all older restraints.

    Setting all None restraints reset restraints in system.

    :param system: System
    :param restraints: Restraints object
    :return: System with restraints
    """
    restraints_map = restraints.convert()
    top_map = dict(system.topology_map)
    top_map.update(
        {
            top_name: top_map[top_name].set_restraints(
                restraints_type=restraints._field_name,  # noqa: SLF001
                restraints_data=res,
            )
            for top_name, res in restraints_map.items()
        }
    )
    return chem.System(
        name=system.name,
        forcefield=system.forcefield,
        topology_map=frozendict(top_map),
        molecules=system.molecules,
        geometry=system.geometry,
        index=system.index,
        info=system.info,
    )
