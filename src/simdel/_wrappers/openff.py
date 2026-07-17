"""Low-level openff functions. Mamba need."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from simdel._misc import context, log, utils


class ParametrizeOut(utils.PathContainer):
    """Openff parametrize output file paths container."""

    gro: Path
    """Geometry .gro path"""

    top: Path
    """Topology .top path"""


@context.require_mamba
def parametrize(
    workdir: Path,
    sdf: Path,
    out_name: str,
    ff: str,
    fast: bool = False,
) -> ParametrizeOut:
    """REQUIRE MAMBA DEPENDENCIES!
    Parametrize molecule by SMIRNOFF force field and save .top .pdb files.

    :param workdir: Workdir path
    :param sdf: Molecule .sdf file path
    :param out_name: Out name
    :param ff: Forcefield type
    :param fast: Use fast algorithm calculating partial charges `gasteiger`,
    instead of `am1bcc` (slow variant), defaults to False
    :return: Path container
    """
    from openff import interchange, units  # type: ignore # noqa: PLC0415
    from openff.toolkit import topology  # type: ignore # noqa: PLC0415
    from openff.toolkit.typing.engines import smirnoff  # type: ignore # noqa: PLC0415

    top = workdir / f"{out_name}.top"
    gro = workdir / f"{out_name}.gro"

    with log.context(msg=f"openff parametrize {sdf.name}", level=log.Level.DEBUG):
        molecule = topology.Molecule.from_file(sdf.as_posix())
        if isinstance(molecule, list):
            msg = f"Several molecules in file: {sdf}"
            raise TypeError(msg)

        molecule.name = out_name
        pc_method = "gasteiger" if fast else "am1bcc"

        try:
            molecule.assign_partial_charges(partial_charge_method=pc_method)
        except ValueError:
            # Ambertools on Mac fails to read sdf, switch to fallback pdb-based antechamber run
            _fallback_am1bcc(molecule)
        forcefield = smirnoff.ForceField(f"{ff}.offxml")
        interchange_ = interchange.Interchange.from_smirnoff(
            force_field=forcefield,
            topology=[molecule],
            charge_from_molecules=[molecule],
            box=units.Quantity([10, 10, 10], units.unit.nanometer),
        )
        interchange_.to_top(top.as_posix())
        interchange_.to_gro(gro.as_posix())
    return ParametrizeOut(gro=gro, top=top)


def _fallback_am1bcc(molecule) -> None:  # noqa: ANN001
    """Assign AM1BCC partial charges to the `molecule` using Amber tools
    `antechamber` run.

    This function is simplified version of `molecule.assign_partial_charges`
    with using PDB as `antechamber` input instead of SDF.

    :param molecule: Molecule to charge
    """
    from openff import units  # type: ignore # noqa: PLC0415
    from openff.toolkit import topology  # type: ignore   # noqa: F401, PLC0415
    from openff.toolkit.utils import exceptions  # type: ignore # noqa: PLC0415

    with TemporaryDirectory() as tmpdir:
        net_charge = molecule.total_charge.m_as(units.unit.elementary_charge)

        pdb_fname = "molecule.pdb"
        mol_fname = "charged.mol2"
        charges_fname = "charges.txt"
        molecule.to_file(f"{tmpdir}/{pdb_fname}", file_format="pdb")

        command = [
            "antechamber",
            f"-i {pdb_fname}",
            "-fi pdb",
            f"-o {mol_fname}",
            "-fo mol2",
            "-pf yes",
            "-dr n",
            "-c bcc",
            f"-nc {net_charge}",
        ]
        utils.run(
            command=command,
            title=f"antechamber convert {molecule.pdb}",
            workdir=Path(tmpdir),
        )

        # Write out charges
        command = [
            "antechamber",
            "-dr n",
            f"-i {mol_fname}",
            "-fi mol2",
            "-o charges2.mol2",
            "-fo mol2",
            "-c wc",
            f"-cf {charges_fname}",
            "-pf yes",
        ]
        utils.run(
            command=command,
            title=f"antechamber set charges {molecule.pdb}",
            workdir=Path(tmpdir),
        )

        # Check to ensure charges were actually produced
        charges_txt = Path(tmpdir) / charges_fname
        if not charges_txt.exists():
            msg = (
                f"Antechamber/sqm partial charge calculation failed on molecule {molecule.name} "
                f"(SMILES {molecule.to_smiles()})"
            )
            raise exceptions.ChargeCalculationError(msg)
        # Read the charges
        charges = np.asarray(list(map(float, charges_txt.read_text().split())), dtype=np.float64)
    charges = units.Quantity(charges, units.unit.elementary_charge)  # type: ignore
    molecule.partial_charges = charges
    molecule._normalize_partial_charges()  # noqa: SLF001
