"""Base pipeline classes module.

Base pipeline - simulation stages launcher, all instructions must be in
Pipeline.pipeline_run()-> PipelineResult (pipeline stages) and
replica.replica_run() -> BaseModel (pipeline's replica stages):
1. General preparation
2. Create replicas and their data (experiment variants to statistic analysis)
3. Run replicas using .pipeline_replica_run() launcher method
4. Analyze replica outputs
5. Analyse pipeline

Pipeline can be run locally - .run() or on remote container - .run_remote(),
you can use .remote_hook() to do something before create remote container with pipeline.
Experiments run using .pipeline_replica_run() only in pipeline instance.

Pipeline automatic create log.log, log.json files.
Implicit log variables:
- LOG_ID - uses for connect start and end function run
- LOG_STREAM_ID - uses to separate replica runs and pipeline run

Everything else is at your discretion (analyse, run structure too).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
import warnings

from pydantic import BaseModel
import pythonjsonlogger.jsonlogger as json_log

import simdel
from simdel import _log, _utils

from . import pools

_STDOUT_HANDLED: bool = False
"""Check handle_log_stdout called yet"""

_FORMATTER = logging.Formatter(
    "%(asctime)s |%(levelname)-7s|%(log_id)4s| %(message)-s", datefmt="%H:%M"
)
"""Basic log formatter for md lib."""

_JSON_FORMATTER = json_log.JsonFormatter(  # type: ignore
    "%(asctime)s %(levelname)-7s %(log_id)4s %(module)s %(message)-s %(desc)s %(log_stream_id)s"
)
"""Basic log.json formatter for md lib."""


class Pipeline(BaseModel, arbitrary_types_allowed=True, extra="allow"):
    """Simulation pipeline."""

    workdir: Path
    """Path to out files dir."""

    compress: bool = True
    """Compress workdir during pipelines working."""

    workers: int
    """N experiments running parallel."""

    n_replicas: int
    """N replicas for each experiment."""

    # TODO: to private
    progress_list: dict[int, int] = {}
    """Progress list."""

    @property
    def label(self) -> str:
        """Pipeline label."""
        return str(self.__class__)

    def pipeline_run(self) -> _utils.Table:
        """General run function."""
        ...

    def pipeline_replica_run(
        self, replica: Replica, n_replicas: int, replica_input: Any
    ) -> BaseModel | None:
        """Runner for pipeline's replica.

        :param replica: Replica
        :param n_replicas: Total replicas in pipeline
        :param replica_input: Replica input data
        :return: Calculation result or none (if fail)
        """
        # TODO: can combine?
        try:
            with _log.context(f"Replica {replica.label}", level=_log.Level.INFO):  # noqa: SIM117
                with _log.log_stream_context(str(replica.id)):
                    result = replica.run(replica_input)

        except Exception:
            self.progress_list[replica.id] = -1
            _progress_log(results_list=self.progress_list, n_replicas=n_replicas)
            return None

        else:
            self.progress_list[replica.id] = 1
            _progress_log(results_list=self.progress_list, n_replicas=n_replicas)
            return result

    def run(self) -> _utils.Table:
        """Run pipeline on local host.

        :return: Pipeline result table
        """
        self.workdir = self.workdir.resolve()
        if self.workdir.exists():
            msg = f"Pipeline folder is exists: {self.workdir}"
            _log.warning(msg)
        self.workdir.mkdir(parents=True, exist_ok=True)

        _log_methods(self)
        handle_log_file(self.workdir)
        handle_log_json(self.workdir)

        suppress_warnings()
        handle_stdout_log()

        pipe_data = dict(self)
        extra_data = {"lib_version": simdel.__version__}
        (self.workdir / "configs.json").write_text(
            json.dumps(
                pipe_data | extra_data,
                indent=4,
                default=str,
            )
        )

        return self.pipeline_run()


class PipelineRemote(Pipeline):
    """Remote version of pipeline. Can run itself on remote."""

    session: pools.Session
    """Session to run remote"""

    def pipeline_remote_hook(self):
        """Run additional functions before remote run."""
        ...

    def run_remote(self) -> str:
        """Run pipeline on remote host.

        :return: Container id
        """
        self.pipeline_remote_hook()
        with pools.RemotePool(self.__class__.run, session=self.session, cpu=4, memory=32) as f:
            future = f(self)
            return future.id


class Replica(BaseModel, arbitrary_types_allowed=True, extra="allow"):
    """Pipeline replica. Helper class - not use outside pipeline."""

    id: int
    """Replica id."""

    label: str
    """Replica name without id."""

    workdir: Path
    """Path to out files dir."""

    compress: bool = True
    """Compress workdir during pipelines working."""

    def replica_run(self, replica_input: Any) -> BaseModel:
        """General run function."""
        ...

    def run(self, replica_input: BaseModel) -> BaseModel:
        """Run replica on local host.

        :return: BaseModel result
        """
        self.workdir = self.workdir.resolve()
        if self.workdir.exists():
            msg = f"Replica folder is exists: {self.workdir}"
            _log.warning(msg)
        self.workdir.mkdir(parents=True, exist_ok=True)

        handle_log_file(self.workdir)
        _log_methods(self)

        return self.replica_run(replica_input)


def suppress_warnings():
    """Suppress all warnings in stdout."""
    logging.captureWarnings(capture=True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(_log.Level.WARNING.value)
    warnings.filterwarnings("ignore", category=UserWarning, module=".*site-packages.*")


def handle_stdout_log(level: _log.Level = _log.Level.INFO):
    """Handle all log messages >= Info to stdout at remote container.

    :param level: Logging level
    """
    global _STDOUT_HANDLED
    if not _STDOUT_HANDLED:
        handler = logging.StreamHandler()
        handler.setLevel(level.value)
        handler.setFormatter(_FORMATTER)
        handler.addFilter(_log.LogStreamFilter(default_stream=True))
        _log.logger.addHandler(handler)
        _STDOUT_HANDLED = True


def handle_log_file(folder: Path, name: str = "log"):
    """Handle all logs >= Debug to .log file.

    :param folder: Log .log file folder
    :param name: Log file, defaults to "log"
    """
    handler = logging.FileHandler(filename=folder / f"{name}.log")
    handler.setLevel(_log.Level.DEBUG.value)
    handler.setFormatter(_FORMATTER)
    handler.addFilter(_log.LogStreamFilter(default_stream=False))
    _log.logger.addHandler(handler)


def handle_log_json(folder: Path):
    """Handle all log messages to json at remote task.

    :param folder: Save dir path
    """
    handler = logging.FileHandler(filename=folder / "log.json")
    handler.setLevel(_log.Level.DEBUG.value)
    handler.setFormatter(_JSON_FORMATTER)
    logging.getLogger().addHandler(handler)


def _log_methods(self: BaseModel):
    """Log all public methods in class.

    pydantic.BaseModel works with extra='allow'
    :param self: Any class
    """
    for i in self.__dir__():
        if not hasattr(self, i):
            continue
        field = getattr(self, i)

        if (
            i.startswith(("_", "model_"))
            or (not callable(field))
            or (i == "run")
            or isinstance(field, property)
        ):
            continue
        logged = _log.context(i.replace("_", " ").upper(), level=_log.Level.INFO)(field)
        setattr(self, i, logged)


def _progress_log(results_list: dict[int, int], n_replicas: int):
    """Write process bar to log.

    :param results_list: Pipeline results list, array of [0 (in progress), 1(ok), -1 (not ok)]
    :param n_replicas: N replicas in pipeline
    """
    ok = sum(i for i in results_list.values() if i > 0)
    nok = -sum(i for i in results_list.values() if i < 0)
    _log.info(f"Progress: {ok + nok:>3d}/{n_replicas} fails:{nok:>3d}")
