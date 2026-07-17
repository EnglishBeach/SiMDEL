"""System class."""

from __future__ import annotations

from pathlib import Path

from frozendict import frozendict
import numpy as np
import pandas as pd
from pydantic import BaseModel

from simdel._misc import log, utils
from simdel._parsers import gro_parser, index_parser, top_parser

from . import (
    ff_map,
    forcefield,
    geometry,
    info,
    topology,
    views,
)


class SystemDump(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """File paths container after saving system."""

    top: Path
    """Topology .top file path."""

    gro: Path
    """Geometry .gro file path."""

    index: Path
    """Index .ndx file path."""


class System(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """Molecular system with forcefield, molecule topologies and geometry,
    ready for manipulations.
    """

    name: str
    """System name."""

    forcefield: forcefield.Forcefield
    """System forcefield."""

    topology_map: frozendict[str, topology.Topology]
    """System topologies (keys synced with topology names)."""

    molecules: tuple[str, ...]
    """Molecules (synced with topology names) in geometry order."""

    geometry: geometry.Geometry
    """System geometry."""

    index: frozendict[str, pd.Series] = frozendict()
    """Atom selections (indexes)."""

    info: info.SystemInfo
    """System forcefield, water types info container."""

    @property
    def topology_view(self) -> views.TopologyView:
        """Get atom properties table for each topology in system.

        (if defined different parameters in atomtype, atoms -
        will be used parameter from topology atoms information).
        """
        view_list = [self._get_topology_view(i) for i in self.topology_map.values()]

        views_head = view_list[0]

        columns = views_head.keys()
        types = views_head.to_df().dtypes.to_dict()
        # Optimized concatenation
        data = np.concatenate([i.to_df() for i in view_list])
        df = pd.DataFrame(data, columns=columns).astype(types)
        return views.TopologyView(**df)

    @property
    def geometry_view(self) -> views.GeometryView:
        """Pivot table for atom parameters for each atom in geometry.

        MAY BE SLOW FOR BIG SYSTEMS!
        (if defined different parameters in atomtype, atoms -
        will be used parameter from topology atoms information).
        """
        view_map = {
            top_name: self._get_topology_view(top) for top_name, top in self.topology_map.items()
        }

        # Optimized concatenation
        views_head = next(iter(view_map.values()))
        columns = views_head.keys() + ["molecule_n"]
        types = views_head.to_df().dtypes.to_dict()

        atoms_data = []
        for i, molecule in enumerate(self.molecules):
            data = view_map[molecule].to_df()
            data["molecule_n"] = i
            atoms_data.append(np.array(data))
        atoms_df = pd.DataFrame(np.concatenate(atoms_data), columns=columns).astype(types)

        c = self.geometry.coordinates
        coords_df = pd.DataFrame(
            dict(
                x=c.x,
                y=c.y,
                z=c.z,
                vx=c.vx,
                vy=c.vy,
                vz=c.vz,
                chain=c.chain,
            )
        )
        return views.GeometryView(**pd.concat([coords_df, atoms_df], axis=1))

    def __repr__(self) -> str:
        data = dict.fromkeys(self.molecules)
        mol_counts = " ".join([f"{i}={self.molecules.count(i)}" for i in data])
        ghost_tops = " ".join([i for i in self.topology_map if i not in data])
        return f"<System: {self.name} [{mol_counts}] ({ghost_tops}); {self.info}>"

    def __add__(self, system: System) -> System:
        """Mix systems together: geometries, topologies, forcefields.

        Do not resolve topology/forcefield conflicts, mix compatible systems:
        1. different topologies have different names
        2. different atomtypes have different names
        3. same name atomstypes have same atomtypes parameters, bondstypes...
        4. same topologies can have different names
        5. same atomtypes can have different names

        :param system: Other system
        :return: Mixed system
        """
        forcefield = self.forcefield + system.forcefield
        info = self.info + system.info
        topology_map = self._mix_topologies(system.topology_map)
        geometry, molecules = self._mix_geometry(system)

        return System(
            name=self.name,
            forcefield=forcefield,
            topology_map=topology_map,
            geometry=geometry,
            molecules=molecules,
            index=frozendict(),
            info=info,
        )

    @classmethod
    def load(
        cls,
        top: Path,
        gro: Path,
        index: Path | None = None,
        *,
        box_type: geometry.BoxType | None = None,
    ) -> System:
        """Load system from GROMACS topology .top file and geometry .gro file.

        :param top: Topology .top file path
        :param top: Forcefield label
        :param pdb: Geometry .pdb file path, defaults to None
        :param gro: Geometry .gro file path, defaults to None
        :param box_type: Geometry box type, defaults to None (unknown)

        :return: System
        """
        system_name, ff, topology_map, molecules = cls._load_top(top)
        geo = cls._load_geometry(
            gro=gro, topology_map=topology_map, molecules=molecules, box_type=box_type
        )
        geo = cls._load_geometry(
            gro=gro, topology_map=topology_map, molecules=molecules, box_type=box_type
        )
        index_ = cls._load_index(index=index, n_atoms=len(geo.coordinates.x))
        return System(
            name=system_name,
            forcefield=ff,
            topology_map=topology_map,
            molecules=molecules,
            geometry=geo,
            index=index_,
            info=info.SystemInfo(
                ff_type=None,
                water_type=None,
                water_flexibility=None,
            ),
        )

    def save(self, save_dir: Path) -> SystemDump:
        """Save system to GROMACS topology .top and geometry .gro files.

        :param save_dir: Save dir path
        :return: Saved files container
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        return SystemDump(
            top=self._save_top(save_dir=save_dir),
            gro=self._save_geometry(save_dir=save_dir),
            index=self._save_index(save_dir=save_dir),
        )

    def rename(self, name: str) -> System:
        """Rename system, must be without spaces inside.

        :param name: New system name
        :return: Renamed system
        """
        name = name.strip()
        if " " in name:
            msg = "System name must be without ' ' inside"
            raise ValueError(msg)

        return System(
            name=name,
            forcefield=self.forcefield,
            topology_map=self.topology_map,
            molecules=self.molecules,
            geometry=self.geometry,
            index=self.index,
            info=self.info,
        )

    def set_info(
        self,
        ff_type: ff_map.FF | None,
        water_type: ff_map.WaterType | None,
        water_flexibility: bool | None,
    ) -> System:
        """Set info to the system.

        :param ff_type: Forcefield type
        :param water_type: Water type or None (unknown or non-solvated)
        :param water_flexibility: Flexible water or None (unknown or non-solvated)
        :return System: System with info
        """
        new_info = info.SystemInfo(
            ff_type=ff_type,
            water_type=water_type,
            water_flexibility=water_flexibility,
        )
        return System(
            name=self.name,
            forcefield=self.forcefield,
            topology_map=self.topology_map,
            molecules=self.molecules,
            geometry=self.geometry,
            index=self.index,
            info=new_info,
        )

    def set_indexes(self, **indexes: pd.Series[bool]) -> System:
        """Set indexes by geometry view selections to system
        Arg names - selection names, values - bitwise masks.

        :return: System with indexes
        """
        self_index = self.geometry_view.index
        try:
            for mask in indexes.values():
                if not all(mask.index == self_index):
                    msg = "Mask and system.view indexes must be same"
                    raise ValueError(msg)
        except Exception as e:
            msg = "Mask taken not from this system geometry view or from system topology view"
            raise ValueError(msg) from e

        return System(
            name=self.name,
            forcefield=self.forcefield,
            topology_map=self.topology_map,
            molecules=self.molecules,
            geometry=self.geometry,
            index=frozendict(indexes),
            info=self.info,
        )

    def rename_topologies(self, name_map: dict[str, str]) -> System:
        """Rename topologies by their name.
        Sync new topology names with molecules list and topology_map.

        :param name: Topology name
        :param new_name: New topology name
        """
        for i in name_map.values():
            if i in self.topology_map:
                msg = f"Topology: {i} already exists"
                raise ValueError(msg)

        topology_map = {}
        for top_name, top in self.topology_map.items():
            new_name = name_map.get(top_name, top_name)
            topology_map[new_name] = top.rename(new_name)

        return System(
            name=self.name,
            forcefield=self.forcefield,
            topology_map=frozendict(topology_map),
            molecules=tuple(name_map.get(top_name, top_name) for top_name in self.molecules),
            geometry=self.geometry,
            index=self.index,
            info=self.info,
        )

    def _mix_topologies(
        self, topology_map: frozendict[str, topology.Topology]
    ) -> frozendict[str, topology.Topology]:
        """Extend topology_map with new topologies and forcefield from other system.

        It must be compatible:
        1. different topologies have different names
        2. different atomtypes have different names
        3. same name atomstypes have same atomtypes parameters, bondstypes...

        All other variants are allowed:
        1. duplicated topologies with different names
        2. duplicated atoms/atomtypes with different names

        :param topology_map: Other topology_map dict {topology name:topology}
        """
        top_map = dict(self.topology_map)
        for top_name, top in topology_map.items():
            if (top_name not in top_map) and (top in top_map.values()):
                tops = list(top_map.values())
                old_name = tops[tops.index(top)].name
                msg = f"Same topologies have different names: {old_name} - {top_name}"
                log.warning(msg)
            elif (top_name in top_map) and (top != top_map[top_name]):
                msg = f"Different topologies have same name: {top_name}"
                raise ValueError(msg)

            top_map.update({top_name: top})
        return frozendict(top_map)

    def _mix_geometry(self, system: System) -> tuple[geometry.Geometry, tuple[str, ...]]:
        """Extend system geometry from other system, concatenate and sort geometry.

        For mixed molecules order:
        AAB + ABB = AAABBB

        :param system: Other system
        :return: New geometry and molecules
        """
        mixed_molecules = self.molecules + system.molecules
        mixed_geometry = self.geometry + system.geometry

        sorted_indexes = []
        molecules: list[str] = []
        mol_indexes = {mol: mixed_molecules.index(mol) for mol in set(mixed_molecules)}
        for newi, name in sorted(
            zip(mixed_geometry.molecule_ids, mixed_molecules, strict=True),
            key=lambda x: mol_indexes[x[1]],
        ):
            sorted_indexes.append(newi)
            molecules.append(name)
        mol_map = dict(zip(mixed_geometry.molecule_ids, sorted_indexes, strict=True))
        return mixed_geometry.sort(mol_map), tuple(molecules)

    # TODO: add docstring
    def _get_topology_view(self, topology: topology.Topology) -> views.TopologyView:
        """Get topology view from topology. Join system forcefiled with topology atoms.

        :param topology: Topology
        :return: Topology view for 1 topology
        """
        ff_df = pd.DataFrame(
            dict(
                type=self.forcefield.atomtypes.type,
                atomic_number=self.forcefield.atomtypes.atomic_number,
                ff_bonded_type=self.forcefield.atomtypes.bonded_type,
                ff_ptype=self.forcefield.atomtypes.particle_type,
                ff_charge=self.forcefield.atomtypes.charge,
                ff_mass=self.forcefield.atomtypes.mass,
                ff_C=self.forcefield.atomtypes.C,
            )
        )
        v = topology.view
        info = v.to_df().set_index("type").join(ff_df.set_index("type")).reset_index()

        charge_mask = info["charge"].isna()
        info.loc[charge_mask, "charge"] = info[charge_mask]["ff_charge"]
        mass_mask = info["mass"].isna()
        info.loc[mass_mask, "mass"] = info[mass_mask]["ff_mass"]

        return views.TopologyView(
            molecule=info["molecule"],
            ai=info["ai"],
            sequence=info["sequence"],
            icode=info["icode"],
            residue=info["residue"],
            name=info["name"],
            atomic_number=info["atomic_number"],
            type=info["type"],
            mass=info["mass"],
            charge=info["charge"],
            charge_group=info["charge_group"],
            typeB=info["typeB"],
            massB=info["massB"],
            chargeB=info["chargeB"],
            ff_ptype=info["ff_ptype"],
            ff_bonded_type=info["ff_bonded_type"],
            ff_C=info["ff_C"],
            posres_C=info["posres_C"],
        )

    @classmethod
    def _load_geometry(
        cls,
        gro: Path,
        topology_map: frozendict[str, topology.Topology],
        molecules: tuple[str, ...],
        box_type: geometry.BoxType | None,
    ) -> geometry.Geometry:
        """Load geometry from .gro file.

        :param gro: Geometry .gro file path
        :param topology_map: Topology map
        :param molecules: Molecules list in geometry
        :param box_type: Geometry box type
        :return: Geometry
        """
        atom_counts = [len(topology_map[i].atoms) for i in molecules]
        gro_data = gro_parser.GROFile.parse(gro)
        return geometry.Geometry.load(atom_counts=atom_counts, gro_data=gro_data, box_type=box_type)

    @classmethod
    def _load_top(
        cls, top: Path
    ) -> tuple[
        str,
        forcefield.Forcefield,
        frozendict[str, topology.Topology],
        tuple[str, ...],
    ]:
        """Load topology from .top file.

        :param top: Topology .top file
        :return: System name, forcefield, topology map, molecules list
        """
        gromacs_top = top_parser.TOPFile.parse(top)
        ff = forcefield.Forcefield.load(gromacs_top.ff)
        tops = [topology.Topology.load(top=i) for i in gromacs_top.top_list]
        topology_map = frozendict({i.name: i for i in tops})

        molecules_data = gromacs_top.system.molecules
        molecules = []
        for name, n_mols in zip(molecules_data.name, molecules_data.n_mols, strict=True):
            molecules.extend([name] * n_mols)

        system_name = gromacs_top.system.system.name[0].strip().replace(" ", "_")
        return system_name, ff, topology_map, tuple(molecules)

    @classmethod
    def _load_index(cls, index: Path | None, n_atoms: int) -> frozendict[str, pd.Series[bool]]:
        """Load index from .ndx file. If no file - return None.

        :param index: Index .ndx file
        :param n_atoms: Number of atoms
        :return: Index dict {str, mask}
        """
        if not index:
            return frozendict()

        return frozendict(index_parser.parse_index(index=index, n_atoms=n_atoms))

    def _save_geometry(self, save_dir: Path) -> Path:
        """Dump geometry to .gro.

        :param save_dir: Save dir path
        :return: Geometry .gro file path
        """
        atom_names: list[str] = []
        residues: list[str] = []
        sequences: list[int] = []
        icodes: list[str | None] = []
        for top in [self.topology_map[i] for i in self.molecules]:
            d = top.atoms
            atom_names.extend(list(d.name))
            residues.extend(list(d.residue))
            icodes.extend(list(d.icode))
            sequences.extend(list(d.sequence))

        gro_file = save_dir / f"{self.name}.gro"
        gro_data = self.geometry.dump(
            atom_names=atom_names,
            residues=residues,
            sequences=sequences,
        )
        utils.backup(gro_file)
        gro_file.write_text("\n".join(gro_data))
        return gro_file

    def _save_top(self, save_dir: Path) -> Path:
        """Dump forcefiled and topologies to .top file.

        :param save_dir: Save dir path
        :return: Topology .top file path
        """
        ff_dump = self.forcefield.dump()

        mol_dumps = []
        for top in self.topology_map.values():
            mol_dumps.extend(top.dump())

        collapsed_molecules = []
        mol_name = self.molecules[0]
        n_mols = 0
        for name in self.molecules:
            if name == mol_name:
                n_mols += 1
            else:
                collapsed_molecules.append((mol_name, n_mols))
                mol_name = name
                n_mols = 1
        collapsed_molecules.append((mol_name, n_mols))

        molecules = top_parser.Molecules(
            name=[i[0] for i in collapsed_molecules],
            n_mols=[i[1] for i in collapsed_molecules],
        )
        system_dump = top_parser.SystemData(
            system=top_parser.System(name=[self.name]), molecules=molecules
        ).dump()

        top_file = save_dir / f"{self.name}.top"
        utils.backup(top_file)
        top_file.write_text("\n".join([*ff_dump, *mol_dumps, *system_dump]))
        return top_file

    def _save_index(self, save_dir: Path) -> Path:
        """Dump indexes to .ndx file.

        :param save_dir: Save dir path
        :return: Index .ndx file path
        """
        index_file = save_dir / f"{self.name}.ndx"

        lines = []
        for name, selection in self.index.items():
            lines.extend(index_parser.dump_index(name=name, mask=selection))
        utils.backup(index_file)
        index_file.write_text("\n".join(lines))
        return index_file
