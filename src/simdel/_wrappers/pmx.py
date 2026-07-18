"""PMX function wrappers."""

from pathlib import Path

import lomap

from simdel import _utils
from simdel._parsers import top_parser

_utils.run("pmx check", ["which pmx"])


def generate_graph(workdir: Path, sdf_map: dict[str, Path]) -> list[tuple[Path, Path]]:
    """Generate transitions graph by lomap algorithm.

    :param workdir: Workdir path
    :param sdf_map: Ligand sdf fname:path map
    :return: List of pairs
    """
    mol_db = lomap.DBMolecules(directory=workdir.as_posix(), radial=True)
    mol_db.build_matrices()
    nx_graph = mol_db.build_graph()
    pairs = []
    for s, e in nx_graph.edges:  # type: ignore
        sdf_a = Path(mol_db.dic_mapping[s]).stem
        sdf_b = Path(mol_db.dic_mapping[e]).stem

        pairs.append((sdf_map[sdf_a], sdf_map[sdf_b]))
    return pairs


class AlignedFiles(_utils.PathContainer):
    """Align ligands output file path container."""

    pair_ab: Path
    """Align A in compare to B data .dat file path"""

    pair_ba: Path
    """Align B in compare to A data .dat file path"""

    score: Path
    """Alignment data .dat file path"""

    core_pdb_a: Path
    """Core of A .pdb file path"""

    core_pdb_b: Path
    """Core of B .pdb file path"""

    aligned_pdb_a: Path
    """Aligned ligand A .pdb file path"""

    aligned_pdb_b: Path
    """Aligned ligand B .pdb file path"""

    log: Path
    """Aligning log .log file path"""


def align_ligands(
    workdir: Path,
    geometry_a: Path,
    geometry_b: Path,
) -> AlignedFiles:
    """REQUIRE PMX!
    Align ligands.

    :param workdir: Workdir path
    :param geometry_a: Ligand A geometry .pdb/.gro file path
    :param geometry_b: Ligand B geometry .pdb/.gro file path
    :return: Aligned systems path container
    """
    pair_AB = workdir / "pair_AB.dat"
    pair_BA = workdir / "pair_BA.dat"
    log_file = workdir / "match.log"
    score = workdir / "score.dat"

    core_A = workdir / "core_A.pdb"
    core_B = workdir / "core_B.pdb"

    aligned_A = workdir / "aligned_A.pdb"
    aligned_B = workdir / "aligned_B.pdb"

    command = [
        "pmx atomMapping",
        f"-i1 '{geometry_a.resolve()}'",
        f"-i2 '{geometry_b.resolve()}'",
        f"-o1 '{pair_AB.resolve()}'",
        f"-o2 '{pair_BA.resolve()}'",
        f"-opdb1 '{aligned_A.resolve()}'",
        f"-opdb2 '{aligned_B.resolve()}'",
        f"-opdbm1 '{core_A.resolve()}'",
        f"-opdbm2 '{core_B.resolve()}'",
        f"-score '{score.resolve()}'",
        f"-log '{log_file.resolve()}'",
    ]

    _utils.run(
        command=command,
        title=f"atom mapping {geometry_a.name}, {geometry_a.name}",
        workdir=workdir,
    )
    return AlignedFiles(
        pair_ab=pair_AB,
        pair_ba=pair_BA,
        score=score,
        core_pdb_a=core_A,
        core_pdb_b=core_B,
        aligned_pdb_a=aligned_A,
        aligned_pdb_b=aligned_B,
        log=log_file,
    )


class HybridizeLigandsOut(_utils.PathContainer):
    """Hybridize ligands output file path container."""

    geometry_a: Path
    """Hybrid geometry A .pdb file path."""

    geometry_b: Path
    """Hybrid geometry B .pdb file path."""

    topology: Path
    """Hybrid topology A/B .top file path."""

    log: Path
    """Hybridizing log .log file path."""


# TODO: refactor
def hybridize_ligands(  # noqa: PLR0913
    workdir: Path,
    topology_name: str,
    geometry_a: Path,
    itp_a: Path,
    ff_a: Path,
    geometry_b: Path,
    itp_b: Path,
    ff_b: Path,
    pair_ab: Path,
) -> HybridizeLigandsOut:
    """REQUIRE PMX!
    Create hybrid topology and hybrid geometry A, B from ligand A, ligand B.

    :param workdir: Workdir path
    :param topology_name: Output molecule name
    :param geometry_a: Ligand A geometry .pdb/.gro file path
    :param itp_a: Ligand A topology .itp file path
    :param ff_a: Ligand A topology .ff.itp file path
    :param geometry_b: Ligand B geometry .pdb/.gro file path
    :param itp_b: Ligand B topology .itp file path
    :param ff_b: Ligand B topology .ff.itp file path
    :param pair_ab: Align A in compare to B data .dat file path
    :return: HybridizeLigandsOut path container
    """
    pdb_a = workdir / "hybrid_A.pdb"
    pdb_b = workdir / "hybrid_B.pdb"
    hybrid_itp = workdir / "hybrid.itp"
    hybrid_ff = workdir / "hybrid_ff.itp"
    log_file = workdir / "create_hybrid_structure.log"

    command = [
        "pmx ligandHybrid",
        f"-i1 '{geometry_a.resolve()}'",
        f"-i2 '{geometry_b.resolve()}'",
        f"-itp1 '{itp_a.resolve()}'",
        f"-itp2 '{itp_b.resolve()}'",
        f"-pairs '{pair_ab.resolve()}'",
        f"-oA '{pdb_a.resolve()}'",
        f"-oB '{pdb_b.resolve()}'",
        f"-oitp '{hybrid_itp.resolve()}'",
        f"-offitp '{hybrid_ff.resolve()}'",
        f"-log '{log_file.resolve()}'",
    ]

    _utils.run(
        command=command,
        title=f"ligand hybrid {geometry_a.name} {geometry_b.name}",
        workdir=workdir,
    )
    data_a = top_parser.TOPFile.parse(ff_a)
    data_b = top_parser.TOPFile.parse(ff_b)
    data_merged = [
        *data_a.ff.defaults.dump(),
        *data_a.ff.atomtypes.dump(),
        *data_b.ff.atomtypes.dump(),
        *hybrid_ff.read_text().split("\n"),
    ]
    topology = workdir / "lig_a.top"
    _utils.backup(topology)
    topology.write_text(
        "\n".join(data_merged) + hybrid_itp.read_text() + _system_head(topology_name)
    )
    return HybridizeLigandsOut(
        geometry_a=pdb_a,
        geometry_b=pdb_b,
        topology=topology,
        log=log_file,
    )


class BAROut(_utils.PathContainer):
    """Calculate BAR output file path container."""

    BAR: Path
    """Energy difference by BAR method .txt file path."""

    integral_ab: Path
    """Energy integral from A to B .dat file path."""

    integral_ba: Path
    """Energy integral from B to A .dat file path."""

    plot: Path
    """Energy plots .png file path."""


# TODO: extract plots...
def calculate_BAR(
    workdir: Path,
    xvgs_a: list[Path],
    xvgs_b: list[Path],
    temperature: float,
    samples: int,
) -> BAROut:
    """REQUIRE PMX!
    Calculate energy difference between A and B stages.

    :param workdir: Workdir path
    :param xvgs_a: List of .xvgs file paths of stage A
    :param xvgs_b: List of .xvgs file paths of stage B
    :param temperature: Temperature of trajectory analysis in `K`
    :param samples: Calculate samples count from trajectory
    :return: Path container
    """
    result = workdir / "BAR.txt"
    integral_AB = workdir / "integral_AB.dat"
    integral_BA = workdir / "integral_BA.dat"
    plot_file = workdir / "plot.png"

    fA = " ".join([f"'{i.resolve().as_posix()}'" for i in xvgs_a])
    fB = " ".join([f"'{i.resolve().as_posix()}'" for i in xvgs_b])
    command = [
        "pmx analyse",
        f"-fA {fA}",
        f"-fB {fB}",
        f"-o '{result.resolve()}'",
        f"-oA '{integral_AB.resolve()}'",
        f"-oB '{integral_BA.resolve()}'",
        f"-w '{plot_file.resolve()}'",
        f"-t {temperature}",
        f"-b {samples}",
    ]

    _utils.run(
        command=command,
        title="analyze BAR",
        workdir=workdir,
    )
    return BAROut(
        BAR=result,
        integral_ab=integral_AB,
        integral_ba=integral_BA,
        plot=plot_file,
    )


def _system_head(name):
    """Generate system topology head.

    :param name: System name
    :return: System topology head
    """
    return f"""
[ system ]
; Name
{name}

[ molecules ]
; Compound                     #mols
{name}	 1

"""
