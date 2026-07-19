"""Local metadynamics pipeline: 1 frame, 1 funnel."""

# ruff: disable[D101, D102]
from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydantic import BaseModel, model_validator

from simdel import chem, run

from . import core


class ReplicaInput(BaseModel):
    ligand: chem.System
    """Input ligand."""

    receptor: chem.System
    """Protein with extras."""

    site_residues: list[core.SiteResidue]
    """Site address."""


class ReplicaResult(BaseModel):
    dG: float
    """Mean dG, in `kJ/mol`"""

    dG_std: float | None
    """Standard error dG, in `kJ/mol`"""

    error_stable: bool
    """Error stability"""


class SimulationParameters(BaseModel, arbitrary_types_allowed=True):
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

    min_time: float = 0.1
    """Time to run NVT, NPT minimize simulation in `ns`"""

    npt_time: float = 0.3
    """Time to run NPT simulation in `ns`"""

    # TODO: to 100
    meta_time: float = 100.0
    """Time to run NPT simulation in `ns`"""


@core.mark_pipeline
class FunnelSimple(run.Pipeline, SimulationParameters):
    """Metadynamics local pipeline."""

    raw_prot_path: Path
    """Protein .pdb file path."""

    extra_sdf_paths: list[Path]
    """List of extra .sdf file paths."""

    raw_ligs_sdf: Path
    """Ligands .sdf path."""

    site_residues: list[core.SiteResidue] = []
    """Site residues to custom site marking."""

    @model_validator(mode="after")
    def setup(self):
        self.ff_type.get_water_info(self.water_type)
        return self

    def pipeline_run(self) -> core.PipelineResult:
        protein = self.parametrize_protein(self.raw_prot_path)
        extras = self.parametrize_extras(self.extra_sdf_paths)
        receptor = sum(extras, protein)
        ligands, _ = self.parametrize_ligands(self.raw_ligs_sdf)
        replicas, inputs = self.create_replicas(receptor, ligands)
        results = self.run_replicas(replicas, inputs)
        return self.analyze(replicas, results)

    def parametrize_protein(self, raw_prot_path: Path) -> chem.System:
        workdir = self.workdir / "1_protein"

        return core.parametrize_protein(
            workdir=workdir,
            raw_prot_path=raw_prot_path,
            ff_type=self.ff_type,
            water_type=self.water_type,
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

    def parametrize_ligands(
        self, raw_ligs_sdf: Path
    ) -> tuple[
        list[chem.System],
        dict[str, str],
    ]:
        workdir = self.workdir / "3_ligands"

        sdf_map, name_map = core.split_ligand_sdf(
            workdir=workdir,
            raw_ligs_sdf=raw_ligs_sdf,
        )
        ligs = [
            core.parametrize_ligand(
                sdf=sdf,
                name=i,
                workdir=workdir / i,
                fast=True,
                compress=self.compress,
            )
            for i, sdf in sdf_map.items()
        ]
        return ligs, name_map

    def create_replicas(
        self,
        receptor: chem.System,
        ligands: list[chem.System],
    ) -> tuple[list[Replica], list[ReplicaInput]]:
        workdir = self.workdir / "4_replicas"

        sim_params = dict(SimulationParameters(**dict(self)))

        ligand_list: list[chem.System] = []
        for i in ligands:
            ligand_list.extend([i] * self.n_replicas)

        replicas: list[Replica] = []
        inputs: list[ReplicaInput] = []

        for i, ligand in enumerate(ligand_list):
            replicas.append(
                Replica(
                    workdir=workdir / f"{ligand.name}_{i}",
                    id=i,
                    label=ligand.name,
                    compress=self.compress,
                    # session=self.session,
                    **sim_params,
                )
            )
            inputs.append(
                ReplicaInput(
                    receptor=receptor,
                    ligand=ligand,
                    site_residues=self.site_residues,
                )
            )
        return replicas, inputs

    def run_replicas(
        self,
        replicas: list[Replica],
        inputs: list[ReplicaInput],
    ) -> list[ReplicaResult | None]:
        n_replicas = len(replicas)
        with run.LocalPool(self.pipeline_replica_run, max_workers=self.workers) as replica_run:
            futures = [
                replica_run(
                    replica=replica,
                    replica_input=data,
                    n_replicas=n_replicas,
                )
                for replica, data in zip(replicas, inputs, strict=True)
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
    ) -> core.PipelineResult:
        workdir = self.workdir / "5_result"

        return _analyse(replicas, results, workdir)


class Replica(run.Replica, SimulationParameters):
    def replica_run(self, replica_input: ReplicaInput) -> ReplicaResult:
        complex_ = replica_input.receptor + replica_input.ligand
        box = self.create_box(complex_)
        em = self.minimize(box)
        resolvated = self.resolvate(em)
        npt = self.npt(resolvated)
        funnel, masks = self.create_funnel(system=npt, site_residues=replica_input.site_residues)
        data = self.meta(system=npt, funnel=funnel, masks=masks)
        return self.analyze(metad_out=data, funnel=funnel)

    def create_box(self, complex: chem.System) -> chem.System:
        workdir = self.workdir / "1_box"
        return core.create_box(
            workdir=workdir,
            system=complex,
            compress=self.compress,
            emtol=self.emtol,
            concentration=self.concentration,
            water_type=self.water_type,
        )

    def minimize(self, box: chem.System) -> chem.System:
        workdir = self.workdir / "2_minimize"
        return core.minimize(
            box=box,
            workdir=workdir,
            emtol=self.emtol,
            min_time=self.min_time,
            T=self.T,
            n_mpi=None,
            n_omp=None,
            compress=self.compress,
        )

    def resolvate(self, system: chem.System) -> chem.System:
        workdir = self.workdir / "3_resolvate"
        return core.set_rigid_water(
            workdir=workdir,
            system=system,
            compress=self.compress,
        )

    def npt(self, system: chem.System) -> chem.System:
        workdir = self.workdir / "4_npt"
        return core.npt(
            system=system,
            workdir=workdir,
            T=self.T,
            time=self.npt_time,
            frames=1,
            n_mpi=None,
            n_omp=None,
            compress=self.compress,
        )[0]

    def create_funnel(
        self,
        system: chem.System,
        site_residues: list[core.SiteResidue],
    ) -> tuple[
        chem.Funnel,
        tuple[pd.Series[bool], pd.Series[bool], pd.Series[bool]],
    ]:
        workdir = self.workdir / "5_funnel"

        site_mask, ligand_mask, protein_mask = core.site_search(
            system=system,
            workdir=workdir,
            ligand_name=self.label,
            site_residues=site_residues,
        )

        funnels = core.create_funnels(
            workdir=workdir,
            system=system,
            site_mask=site_mask,
            ligand_mask=ligand_mask,
            protein_mask=protein_mask,
        )
        funnel = funnels[0]

        return (
            funnel,
            (site_mask, ligand_mask, protein_mask),
        )

    def meta(
        self,
        system: chem.System,
        funnel: chem.Funnel,
        masks: tuple[pd.Series[bool], pd.Series[bool], pd.Series[bool]],
    ) -> core.MetadynamicsOut | None:
        workdir = self.workdir / "6_meta"

        site_mask, ligand_mask, protein_mask = masks
        return core.metadynamics(
            system=system,
            funnel=funnel,
            workdir=workdir,
            ligand_mask=ligand_mask,
            site_mask=site_mask,
            protein_mask=protein_mask,
            T=self.T,
            time=self.meta_time,
            n_mpi=None,
            n_omp=None,
            compress=self.compress,
            sigma=0.2,
            height=2,
            pace=500,
        )

    def analyze(
        self,
        metad_out: core.MetadynamicsOut | None,
        funnel: chem.Funnel,
    ) -> ReplicaResult:
        workdir = self.workdir / "7_analyze"
        if not metad_out:
            return ReplicaResult(dG=0, dG_std=None, error_stable=False)

        result = core.analyze_meta(
            workdir=workdir,
            hills_file=metad_out.hills,
            cv_file=metad_out.cv,
            funnel_file=metad_out.funnel,
            T=self.T,
            funnel=funnel,
            fast=True,
        )
        return ReplicaResult(dG=result.dG, dG_std=result.dG_error, error_stable=True)


def _analyse(
    replicas: list[Replica],
    results: list[ReplicaResult | None],
    workdir: Path,
) -> core.PipelineResult:
    workdir.mkdir(parents=True, exist_ok=True)

    df = []
    for replica, result in zip(replicas, results, strict=True):
        df.append(
            dict(
                ID=replica.id,
                LABEL=replica.label,
                dG=result.dG if result else None,
            )
        )
    df = pd.DataFrame(df)

    pipeline_result = []
    for label, g_df in df.groupby(by="LABEL"):
        dG = g_df["dG"].mean()
        pipeline_result.append(dict(ligand=label, dG=dG))
    pipeline_result = pd.DataFrame(pipeline_result)
    pipeline_result.to_csv(workdir / "result.csv")
    return core.PipelineResult(**pipeline_result)
