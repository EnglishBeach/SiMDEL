"""Base FEP pipeline."""

# ruff: disable[D101, D102]
from __future__ import annotations

from pathlib import Path
import typing

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from simdel import chem, run

from . import core

Edge: typing.TypeAlias = list[chem.System]
TrajEdge: typing.TypeAlias = list[tuple[chem.System, chem.Trajectory]]
ResultEdge: typing.TypeAlias = list[list[core.NeqResult | None]]


class SimulationParameters(BaseModel):
    ff_type: chem.GromacsFF = chem.DefaultFF.amber99sb_ildn
    """Forcefield type"""

    water_type: chem.WaterType = chem.WaterType.tip3p
    """Water to forcefield type, must be compatible with forcefield"""

    concentration: float = 0.15
    """Ions concentration in systems"""

    T: int = 300
    """Temperature to analyze in `K`"""

    emtol: float = 100
    """Min value for maximum force to stop simulation, in `kJ/(mol*nm)`"""

    eq_time: float = 5.0
    """Time to run EQ simulation in `ns`"""

    internal_steps: int = 4
    """Number os steps between frames to save to check energy"""

    discard_time: float = 2.5
    """Start time for frames splitting to neq stage in `ns`"""

    n_frames: int = 80
    """Number of transition frames to calculate"""

    neq_time: float = 0.05
    """Time to run NEQ in `ns`"""

    samples: int = 10
    """Samples to analyze"""


class ReplicaInput(BaseModel):
    receptor: chem.System
    hybrid_a: chem.System
    hybrid_b: chem.System


class ReplicaResult(BaseModel):
    """Replica result.

    - p - protein with extra molecules
    - La, Lb - ligands a/b states
    - pLa, pLb - complexes
    """

    ddG: float
    """Free energy site binding difference:
    ddG = dG(p + Lb -> pLb) - dG(c + La -> pLa)
    = dG_pL - dG_L."""

    dG_L: float
    """dG(La) - dG(Lb) in `kJ/mol`"""

    dG_pL: float
    """dG(pLa) - dG(pLb) in `kJ/mol`"""

    dG_L_error: float
    """dG(La) - dG(Lb) analytical error, s**2 = s1**2 + s2**2 in `kJ^2/mol^2`"""

    dG_pL_error: float
    """dG(pLa) - dG(pLb) analytical error, s**2 = s1**2 + s2**2 in `kJ^2/mol^2`"""

    dG_L_berror: float
    """dG(La) - dG(Lb) bootstrap error in `kJ^2/mol^2`"""

    dG_pL_berror: float
    """dG(pLa) - dG(pLb) bootstrap error in `kJ^2/mol^2`"""


class FEP(run.Pipeline, SimulationParameters):
    """FEP local pipeline."""

    raw_prot_path: Path
    """Protein .pdb file path"""

    raw_ligs_sdf: Path
    """Ligands .sdf file"""

    extra_sdf_paths: list[Path]
    """List of extra .sdf file paths"""

    pairs: list[tuple[str, str]] = []
    """Custom generate ligand pairs using their names in sdf"""

    @model_validator(mode="after")
    def _setup(self):
        self.ff_type.get_water_info(self.water_type)
        return self

    def pipeline_run(self) -> core.PipelineResult:
        protein = self.parametrize_protein(raw_prot_path=self.raw_prot_path)
        extras = self.parametrize_extras(sdf_list=self.extra_sdf_paths)
        receptor = sum(extras, protein)
        sdf_map, name_map, pair_keys = self.gen_graph(
            raw_ligs_sdf=self.raw_ligs_sdf, pairs=self.pairs
        )
        ligand_pairs = self.parametrize_ligands(sdf_map=sdf_map, pairs=pair_keys)
        hybrids_pairs = self.create_hybrids(pairs=ligand_pairs)
        replicas, inputs = self.create_replicas(hybrids_pairs=hybrids_pairs, receptor=receptor)
        results = self.run_replicas(replicas, inputs)
        return self.analyze(replicas=replicas, results=results, name_map=name_map)

    def parametrize_protein(self, raw_prot_path: Path) -> chem.System:
        workdir = self.workdir / "1_protein"

        return core.parametrize_protein(
            workdir=workdir,
            raw_prot_path=raw_prot_path,
            ff_type=self.ff_type,
            compress=self.compress,
        )

    def parametrize_extras(self, sdf_list: list[Path]) -> list[chem.System]:
        workdir = self.workdir / "2_extras"

        sdf_map = core.create_extras_map(
            workdir=workdir,
            extra_sdf=sdf_list,
        )
        return [
            core.parametrize_extra(
                sdf=sdf,
                name=name,
                workdir=workdir / sdf.stem,
                fast=True,
                compress=self.compress,
            )
            for name, sdf in sdf_map.items()
        ]

    def gen_graph(
        self,
        raw_ligs_sdf: Path,
        pairs: list[tuple[str, str]],
    ) -> tuple[
        dict[str, Path],
        dict[str, str],
        list[tuple[str, str]],
    ]:
        workdir = self.workdir / "3_ligands"

        return core.split_ligand_sdf(
            workdir=workdir,
            raw_ligs_sdf=raw_ligs_sdf,
            pairs=pairs,
        )

    def parametrize_ligands(
        self,
        sdf_map: dict[str, Path],
        pairs: list[tuple[str, str]],
    ) -> list[tuple[chem.System, chem.System]]:
        workdir = self.workdir / "3_ligands"

        lig_map = {
            i: core.parametrize_ligand(
                sdf=sdf,
                name=i,
                workdir=workdir / i,
                fast=True,
                compress=self.compress,
            )
            for i, sdf in sdf_map.items()
        }
        return [(lig_map[a], lig_map[b]) for a, b in pairs]

    def create_hybrids(
        self, pairs: list[tuple[chem.System, chem.System]]
    ) -> list[tuple[chem.System, chem.System]]:
        workdir = self.workdir / "4_hybrids"

        return [
            core.create_hybrids(
                workdir=workdir / f"{systemA.name}_{systemB.name}",
                system_a=systemA,
                system_b=systemB,
                compress=self.compress,
            )
            for systemA, systemB in pairs
        ]

    def create_replicas(
        self,
        hybrids_pairs: list[tuple[chem.System, chem.System]],
        receptor: chem.System,
    ) -> tuple[list[Replica], list[ReplicaInput]]:

        workdir = self.workdir / "5_replicas"

        sim_params = dict(SimulationParameters(**dict(self)))

        pair_list: list[tuple[chem.System, chem.System]] = []
        for i in hybrids_pairs:
            pair_list.extend([i] * self.n_replicas)

        replicas: list[Replica] = []
        inputs: list[ReplicaInput] = []

        for i, (hybrid_a, hybrid_b) in enumerate(pair_list):
            replicas.append(
                Replica(
                    workdir=workdir / f"{hybrid_a.name}_{hybrid_b.name}_{i}",
                    id=i,
                    label=f"{hybrid_a.name}_{hybrid_b.name}",
                    compress=self.compress,
                    # session=self.session,
                    **sim_params,
                )
            )
            inputs.append(
                ReplicaInput(
                    receptor=receptor,
                    hybrid_a=hybrid_a,
                    hybrid_b=hybrid_b,
                )
            )
        return replicas, inputs

    def run_replicas(
        self,
        replicas: list[Replica],
        datas: list[ReplicaInput],
    ) -> list[ReplicaResult | None]:
        n_replicas = len(replicas)
        with run.LocalPool(self.pipeline_replica_run, max_workers=self.workers) as replica_run:
            futures = [
                replica_run(
                    replica=replica,
                    replica_input=data,
                    n_replicas=n_replicas,
                )
                for replica, data in zip(replicas, datas, strict=True)
            ]
        results = [f.result() for f in futures]  # type: ignore

        if not any(i for i in results if i):
            msg = "All experiments failed"
            raise RuntimeError(msg)

        return results  # type: ignore

    def analyze(
        self,
        replicas: list[Replica],
        results: list[ReplicaResult | None],
        name_map: dict[str, str],
    ) -> core.PipelineResult:
        workdir = self.workdir / "5_result"

        return _analyze(
            workdir=workdir,
            replicas=replicas,
            results=results,
            name_map=name_map,
        )


class Replica(run.Replica, SimulationParameters):
    def replica_run(self, replica_input: ReplicaInput) -> ReplicaResult:
        combine_edges = self.combine(
            receptor=replica_input.receptor,
            hybrid_a=replica_input.hybrid_a,
            hybrid_b=replica_input.hybrid_b,
        )
        box_edges = self.create_boxes(combine_edges)
        em_edges = self.em(box_edges)
        em_edges2 = self.set_rigid_water(em_edges)
        eq_edges = self.eq(em_edges2)
        neq_edges = self.neq(eq_edges)
        return self.analyze(neq_edges)

    def combine(
        self, receptor: chem.System, hybrid_a: chem.System, hybrid_b: chem.System
    ) -> list[Edge]:
        return [
            [
                hybrid_a.rename(core.StateName.la),
                (receptor + hybrid_a).rename(core.StateName.ca),
            ],
            [
                hybrid_b.rename(core.StateName.lb),
                (receptor + hybrid_b).rename(core.StateName.cb),
            ],
        ]

    def create_boxes(self, edges: list[Edge]) -> list[Edge]:
        workdir = self.workdir / "1_box"

        with run.LocalPool(Replica._create_boxes, max_workers=4) as f:
            futures = [f(self, workdir, edge) for edge in edges]
        return [i.result() for i in futures]

    def _create_boxes(self, workdir: Path, edge: Edge) -> Edge:
        return [
            core.create_box(
                workdir=workdir / system.name,
                system=system,
                water_type=self.water_type,
                concentration=self.concentration,
                emtol=self.emtol,
                compress=self.compress,
            )
            for system in edge
        ]

    def em(self, edges: list[Edge]) -> list[Edge]:
        workdir = self.workdir / "2_em"

        with run.LocalPool(Replica._em) as f:
            futures = [f(self, workdir, edge) for edge in edges]
        return [i.result() for i in futures]

    def _em(self, workdir: Path, edge: Edge) -> Edge:
        return [
            core.em(
                workdir=workdir / system.name,
                box=system,
                emtol=self.emtol,
                n_omp=None,
                n_mpi=None,
                compress=self.compress,
            )
            for system in edge
        ]

    def set_rigid_water(self, edges: list[Edge]) -> list[Edge]:
        workdir = self.workdir / "3_resolvate"

        return [
            [
                core.set_rigid_water(
                    system=system,
                    workdir=workdir / system.name,
                    compress=self.compress,
                )
                for system in edge
            ]
            for edge in edges
        ]

    def eq(self, edges: list[Edge]) -> list[TrajEdge]:
        workdir = self.workdir / "4_eq"

        with run.LocalPool(Replica._eq) as f:
            futures = [f(self, workdir, edge) for edge in edges]
        return [i.result() for i in futures]

    def _eq(self, workdir: Path, edge: Edge) -> TrajEdge:
        return [
            core.eq(
                workdir=workdir / system.name,
                system=system,
                time=self.eq_time,
                T=self.T,
                n_frames=self.n_frames,
                internal_steps=self.internal_steps,
                n_omp=None,
                n_mpi=None,
                compress=self.compress,
            )
            for system in edge
        ]

    def neq(self, traj_edges: list[TrajEdge]) -> list[ResultEdge]:
        workdir = self.workdir / "5_neq"

        with run.LocalPool(Replica._neq) as f:
            futures = [f(self, workdir, edge) for edge in traj_edges]
        return [i.result() for i in futures]

    def _neq(self, workdir: Path, traj_edges: TrajEdge) -> ResultEdge:
        result_edges = []
        for system, trajectory in traj_edges:
            frame_systems = core.split(
                workdir=workdir / system.name,
                system=system,
                trajectory=trajectory,
                discard_time=self.discard_time,
                end_time=self.eq_time,
                n_frames=self.n_frames,
                compress=self.compress,
            )

            neq_result = []
            for frame_system in frame_systems:
                try:
                    result = core.neq(
                        workdir=workdir / frame_system.name,
                        frame_system=frame_system,
                        time=self.neq_time,
                        T=self.T,
                        n_mpi=None,
                        n_omp=None,
                        compress=self.compress,
                    )
                except Exception:
                    result = None
                neq_result.append(result)

            result_edges.append(neq_result)
        return result_edges

    def analyze(self, neq_edges: list[ResultEdge]) -> ReplicaResult:
        workdir = self.workdir / "6_analyse"

        dG_data: dict[str, list[Path]] = {}
        states = [
            [core.StateName.la, core.StateName.ca],
            [core.StateName.lb, core.StateName.cb],
        ]
        for state_name, state in zip(
            states[0] + states[1],
            neq_edges[0] + neq_edges[1],
            strict=True,
        ):
            dG_data[state_name] = [i.xvg for i in state if i]

        result = core.analyze_dG(
            workdir=workdir,
            T=self.T,
            samples=self.samples,
            table_la=dG_data[core.StateName.la],
            table_lb=dG_data[core.StateName.lb],
            table_ca=dG_data[core.StateName.ca],
            table_cb=dG_data[core.StateName.cb],
        )
        return ReplicaResult(**dict(result))


def _analyze(
    workdir: Path,
    replicas: list[Replica],
    results: list[ReplicaResult | None],
    name_map: dict[str, str],
) -> core.PipelineResult:
    workdir.mkdir(parents=True, exist_ok=True)

    df = []
    for replica, result in zip(replicas, results, strict=True):
        if result:
            df.append(
                dict(
                    ID=replica.id,
                    LABEL=replica.label,
                    ddG=result.ddG,
                    dG_L=result.dG_L,
                    dG_pL=result.dG_pL,
                    dG_L_error=result.dG_L_error,
                    dG_pL_error=result.dG_pL_error,
                    dG_L_berror=result.dG_L_berror,
                    dG_pL_berror=result.dG_pL_berror,
                )
            )
    df = pd.DataFrame(df)

    pipeline_result = []
    for pair, g_df in df.groupby("LABEL"):
        key_a, key_b = pair.split("_")  # type: ignore
        key_a: str
        key_b: str

        lig_a, lig_b = name_map[key_a], name_map[key_b]

        dG_L = g_df["dG_L"].mean()
        dG_pL = g_df["dG_pL"].mean()

        # sampling_power = int(g_df["sampling_power"].sum() / 3)
        sampling_power = 180
        e_L = g_df["dG_L_error"]
        e_pL = g_df["dG_pL_error"]
        ddG_error = np.sqrt(e_L**2 + e_pL**2)

        l_dist = []
        c_dist = []
        # TODO: refactor np.random
        for i in g_df.iloc:
            l_dist.append(np.random.normal(i["dG_L"], i["dG_L_berror"], sampling_power * 3))  # noqa: NPY002
            c_dist.append(np.random.normal(i["dG_pL"], i["dG_pL_berror"], sampling_power * 3))  # noqa: NPY002
        l_dist = np.concatenate(l_dist)
        c_dist = np.concatenate(c_dist)

        ddG_berror = np.sqrt(np.var(l_dist) ** 2 + np.var(c_dist) ** 2)

        pipeline_result.append(
            dict(
                ligand_a=lig_a,
                ligand_b=lig_b,
                ddG=dG_pL - dG_L,
                ddG_error=ddG_error,
                ddG_berror=ddG_berror,
            )
        )

    pipeline_result = pd.DataFrame(pipeline_result)
    pipeline_result.to_csv(workdir / "result.csv")
    return core.PipelineResult(**pipeline_result)
