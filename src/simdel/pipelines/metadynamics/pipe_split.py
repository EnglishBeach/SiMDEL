"""Local metadynamics pipeline: n frames, 1 funnel."""

# ruff: disable[D101, D103, D102]
from __future__ import annotations

from pathlib import Path
import random
import shutil

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from simdel import chem, run

from . import core


def emulate_remote_run(
    workdir: Path,
    raw_prot_path: Path,
    raw_ligs_sdf: Path,
    extra_sdf_paths: list[Path],
):
    # Can't copy ff
    shutil.copy(raw_prot_path, workdir / raw_prot_path.name)
    shutil.copy(raw_ligs_sdf, workdir / raw_ligs_sdf.name)
    if extra_sdf_paths:
        for i in extra_sdf_paths:
            shutil.copy(i, workdir / i.name)


class SimulationParameters(BaseModel, arbitrary_types_allowed=True):
    ff_type: chem.GromacsFF = chem.DefaultFF.amber99sb_ildn
    """Forcefield type"""

    water_type: chem.WaterType = chem.WaterType.tip3p
    """Water to forrcefiled type, must be compatible with forcefield"""

    concentration: float = 0.15
    """Ions concentration in systems"""

    T: int = 300
    """Temperature to analyze in `K`"""

    emtol: float = 100
    """Min value for maximum force to stop simulation, in `kJ/(mol*nm)`"""

    min_time: float = 0.1
    """Time to run NVT simulation in `ns`"""

    npt_time: float = 2
    """Time to run NPT simulation in `ns`"""

    discard_time: float = 0.5
    """Start time for frames splitting to neq stage in `ns`"""

    n_frames: int = 5
    """Number of transition frames to calculate"""

    # TODO: to 100
    meta_time: float = 100.0
    """Time to run NPT simulation in `ns`"""

    # metadynamics parameters
    sigma: float = 0.2
    height: float = 2
    pace: int = 500
    box_distance: float = 2


class ReplicaInput(BaseModel):
    ligand: chem.System
    receptor: chem.System
    site_residues: list[core.SiteResidue]


class ReplicaResult(BaseModel):
    dG: float
    """Mean dG, in `kJ/mol`"""

    dG_std: float
    """Standard error dG, in `kJ/mol`"""

    error_stable: bool
    """Error stability"""


class FunnelSplit(run.Pipeline, SimulationParameters):
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
        protein = self.parametrize_protein(raw_prot_path=self.raw_prot_path)
        extras = self.parametrize_extras(sdf_list=self.extra_sdf_paths)
        receptor = sum(extras, protein)
        ligands, _ = self.parametrize_ligands(raw_ligs_sdf=self.raw_ligs_sdf)
        replicas, replica_datas = self.create_replicas(receptor=receptor, ligands=ligands)
        results = self.run_replicas(replicas=replicas, datas=replica_datas)
        return self.analyze(replicas=replicas, results=results)

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

        with run.LocalPool(core.parametrize_ligand) as parametrize_ligand:
            futures = [
                parametrize_ligand(
                    sdf=sdf,
                    name=name,
                    workdir=workdir / sdf.stem,
                    fast=True,
                    compress=self.compress,
                )
                for name, sdf in sdf_map.items()
            ]
            return [i.result() for i in futures]

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

        replicas: list[Replica] = []
        replica_datas: list[ReplicaInput] = []

        ligand_list = []
        for i in ligands:
            ligand_list.extend([i] * self.n_replicas)
        for i, ligand in enumerate(ligand_list):
            replicas.append(
                Replica(
                    compress=self.compress,
                    # session=self.session,
                    workdir=workdir / f"{ligand.name}_{i}",
                    id=i,
                    label=ligand.name,
                    **sim_params,
                )
            )
            replica_datas.append(
                ReplicaInput(
                    receptor=receptor,
                    ligand=ligand,
                    site_residues=self.site_residues,
                )
            )
        return replicas, replica_datas

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
        results = [f.result() for f in futures]

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
        workdir.mkdir(parents=True, exist_ok=True)

        df = []
        for replica, result in zip(replicas, results, strict=True):
            df.append(
                dict(
                    ID=replica.id,
                    LABEL=replica.label,
                    dG=result.dG if result else None,
                    dG_error=result.dG_std if result else None,
                    error_stable=result.error_stable if result else None,
                )
            )
        df = pd.DataFrame(df)
        df.to_csv(workdir / "result.csv")

        result = []
        for _, g_df in df.groupby(by="LABEL"):
            dG = g_df["dG"].mean()
            # TODO: bootstrap
            dG_error = g_df["dG_error"].mean()

            result.append(dict(dG=dG, dG_error=dG_error))
        result = pd.DataFrame(result)
        return core.PipelineResult(**result)


class Replica(run.Replica, SimulationParameters):
    def replica_run(self, replica_input: ReplicaInput) -> ReplicaResult:
        complex_ = (replica_input.receptor + replica_input.ligand).rename("complex")
        box = self.create_box(complex_)
        em = self.minimize(box)
        resolvated = self.resolvate(em)
        npt, npt_traj = self.npt(resolvated)
        frames = self.split(system=npt, trajectory=npt_traj)
        funnels, mask_list = self.create_funnels(
            frames=frames,
            site_residues=replica_input.site_residues,
        )
        meta_results = self.meta(
            frames=frames,
            funnels=funnels,
            mask_list=mask_list,
        )
        return self.analyze(
            frames=frames,
            funnels=funnels,
            metad_outs=meta_results,
        )

    def create_box(self, complex: chem.System) -> chem.System:
        workdir = self.workdir / "1_box"
        return core.create_box(
            workdir=workdir,
            system=complex,
            compress=self.compress,
            emtol=self.emtol,
            concentration=self.concentration,
            water_type=self.water_type,
            box_distance=self.box_distance,
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

    def npt(self, system: chem.System) -> tuple[chem.System, chem.Trajectory]:
        workdir = self.workdir / "4_npt"
        workdir.mkdir(parents=True, exist_ok=True)
        return core.npt(
            system=system,
            workdir=workdir,
            T=self.T,
            time=self.npt_time,
            frames=self.n_frames,
            n_mpi=None,
            n_omp=None,
            compress=self.compress,
        )

    def split(
        self,
        system: chem.System,
        trajectory: chem.Trajectory,
    ) -> list[chem.System]:
        workdir = self.workdir / "5_split"
        return core.split_trajectory(
            system=system,
            workdir=workdir,
            discard_time=self.discard_time,
            end_time=self.npt_time,
            n_frames=self.n_frames,
            trajectory=trajectory,
            compress=self.compress,
        )

    def create_funnels(
        self,
        frames: list[chem.System],
        site_residues: list[core.SiteResidue] | None = None,
    ) -> tuple[
        list[chem.Funnel],
        list[tuple[pd.Series[bool], pd.Series[bool], pd.Series[bool]]],
    ]:
        workdir = self.workdir / "6_funnel"

        frame_funnels = []
        frame_masks = []

        for frame in frames:
            w = workdir / frame.name
            site_mask, ligand_mask, protein_mask = core.site_search(
                workdir=w,
                system=frame,
                ligand_name=self.label,
                site_residues=site_residues,
            )

            funnels = core.create_funnels(
                workdir=w,
                system=frame,
                site_mask=site_mask,
                ligand_mask=ligand_mask,
                protein_mask=protein_mask,
            )
            best_i = random.randint(0, len(funnels) - 1)
            funnel = funnels[best_i]

            (w / f"best_funnel_{best_i}.json").write_text(funnel.dump())

            frame_funnels.append(funnel)
            frame_masks.append((site_mask, ligand_mask, protein_mask))
        return frame_funnels, frame_masks

    def meta(
        self,
        frames: list[chem.System],
        funnels: list[chem.Funnel],
        mask_list: list[tuple[pd.Series[bool], pd.Series[bool], pd.Series[bool]]],
    ) -> list[core.MetadynamicsOut | None]:
        workdir = self.workdir / "7_metad"

        futures: list[core.MetadynamicsOut | None] = []
        for frame, funnel, masks in zip(frames, funnels, mask_list, strict=True):
            site_mask, ligand_mask, protein_mask = masks
            futures.append(
                core.metadynamics(
                    workdir=workdir / frame.name,
                    system=frame,
                    funnel=funnel,
                    ligand_mask=ligand_mask,
                    site_mask=site_mask,
                    protein_mask=protein_mask,
                    T=self.T,
                    time=self.meta_time,
                    n_mpi=None,
                    n_omp=None,
                    compress=self.compress,
                    sigma=self.sigma,
                    height=self.height,
                    pace=self.pace,
                )
            )
        return futures

    def analyze(
        self,
        frames: list[chem.System],
        funnels: list[chem.Funnel],
        metad_outs: list[core.MetadynamicsOut | None],
    ) -> ReplicaResult:
        size = 100
        workdir = self.workdir / "8_analyze"
        data = [
            core.analyze_meta(
                workdir=workdir / frame.name,
                hills_file=metad_out.hills,
                cv_file=metad_out.cv,
                funnel_file=metad_out.funnel,
                T=self.T,
                funnel=funnel,
                fast=True,
            )
            for frame, funnel, metad_out in zip(frames, funnels, metad_outs, strict=True)
            if metad_out
        ]

        if any(i.error_stable for i in data):
            # TODO: refactor np.random
            dG_distr = [
                np.random.normal(loc=i.dG, scale=i.dG_error, size=size)  # noqa: NPY002
                for i in data
                if i.error_stable
            ]
            stable = True
        else:
            dG_distr = [np.random.normal(loc=i.dG, scale=i.dG_error, size=size) for i in data]  # noqa: NPY002
            stable = False

        distr = np.concatenate(dG_distr)
        return ReplicaResult(dG=distr.mean(), dG_std=distr.std(), error_stable=stable)
