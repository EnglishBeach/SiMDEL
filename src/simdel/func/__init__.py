"""System assembly and manipulation functions."""

import lazyimports

with lazyimports.lazy_imports("lomap"):
    from .alchemy import create_hybrids, gen_alchemy_graph

from .converters import dump_index, gro2pdb, load_index, pdb2gro, split_sdf
from .geometry_transform import (
    axial_align,
    create_box,
    create_gromacs_indexes,
    rescale_box,
    reset_box,
    transform_geometry,
)
from .parametrization import parametrize_protein, parametrize_small
from .simulation import simulate
from .solvation import add_ions, resolvate, solvate
from .topology_transform import (
    add_topologies,
    clear_topologies,
    extract_subsystem,
    replace_topologies,
    set_restraints,
)
