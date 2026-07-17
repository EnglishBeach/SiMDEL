"""Molecular dynamic parameters for QM/MM, QM/MM + CP2K.
See documentation: https://manual.gromacs.org/current/user-guide/mdp-options.html.
"""

import enum

import pydantic

from . import core_mdp


class GroupQMMM(pydantic.BaseModel):
    """Quantum mechanics molecular dynamics parameters groups."""

    QMMM_groups: list[str] = []
    """Groups to be described at the QM level for MiMiC QM/MM"""


class QMMMCp2kQmmethod(enum.Enum):
    """Method used to describe the QM part of the system."""

    PBE = "PBE"
    """DFT using PBE functional and DZVP-MOLOPT basis set."""

    BLYP = "BLYP"
    """DFT using BLYP functional and DZVP-MOLOPT basis set."""

    INPUT = "INPUT"
    """Provide an external input file for CP2K.

    ACTIVE IF  gmx grompp -qmi"""


class GroupQMMMCp2K(pydantic.BaseModel):
    """Hybrid Quantum-Classical simulations (QM/MM) with CP2K interface parameters."""

    qmmm_cp2k_active: core_mdp.bool_tf | None = None
    """Activate QM/MM simulations.
    ACTIVE IF CP2K is linked with GROMACS. (False)"""

    qmmm_cp2k_qmgroup: str | None = None
    """Index group with atoms that are treated with QM. (System)"""

    # TODO: desc
    qmmm_cp2k_qmmethod: QMMMCp2kQmmethod | None = None

    qmmm_cp2k_qmcharge: int | None = None
    """Total charge of the QM part. (0)"""

    qmmm_cp2k_qmmultiplicity: int | None = None
    """Multiplicity or spin_state of QM part.

    1 => singlet (1)"""

    qmmm_cp2k_qmfilenames: list[str] = []
    """Name of the CP2K files that will be generated during the simulation. (_cp2k suffix)"""
