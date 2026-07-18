"""High-level parallel running classes - pools."""

from __future__ import annotations

from concurrent import futures
from concurrent.futures import Future as LocalFuture
import contextvars
import io
import logging
from pathlib import Path
import pickle
import traceback
import typing

from simdel import _log

IN = typing.ParamSpec("IN")
OUT = typing.TypeVar("OUT")


class LocalPool(typing.Generic[IN, OUT]):
    """Pool for parallel launch local functions."""

    def __init__(self, f: typing.Callable[IN, OUT], /, max_workers: int | None = None):
        """Create local pool - wrapped non-parallel function to use it as parallel on local.

        :param f: Function
        :param max_workers: The maximum number of threads that can be used to
        execute the given calls, defaults to None
        """
        self._f: typing.Callable[IN, OUT] = f
        self._runner = futures.ThreadPoolExecutor(max_workers=max_workers)

    def __repr__(self):
        return f"<Parallelized pool: {self._f}>"

    def __call__(_self_, *args: IN.args, **kwargs: IN.kwargs) -> LocalFuture[OUT]:
        """Wrap function and make parallel variant to run it on local.

        Save outside context in threads.
        """
        context = contextvars.copy_context()

        def wrapper(*args: IN.args, **kwargs: IN.kwargs):
            tokens = {var: var.set(value) for var, value in context.items()}
            try:
                result = _self_._f(*args, **kwargs)
            finally:
                for var, token in tokens.items():
                    var.reset(token)
            return result

        return _self_._runner.submit(wrapper, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._runner.shutdown(wait=True)
        return False

    def shutdown(self):
        """Stop external runner."""
        self._runner.shutdown(wait=False)


# TODO: add no wait exit
class RemotePool(typing.Generic[IN, OUT]):
    """Abstract remote pool."""

    _f: typing.Callable[IN, OUT]
    """Wrapped function.
    Must be declared static without enclosure, locals, implicit context"""

    _session: Session
    """Function collection to run, push data, fetch results from remote."""

    _futures: list[RemoteFuture]
    """Started futures."""

    _options: dict
    """Additional parameters to RemoteRunner.submit"""

    def __init__(self, f: typing.Callable[IN, OUT], /, session: Session, **kwargs):
        self._f = f
        self._session = session
        self._futures = []
        self._options = kwargs

    def __repr__(self):
        return f"<Parallelized remote pool: {self._f}>"

    def __call__(_self_, *args: IN.args, **kwargs: IN.kwargs) -> RemoteFuture[OUT]:
        """Wrap function and make parallel variant to run it on local.

        Save outside context in threads.
        """
        id_ = _self_._session.push(
            pickle.dumps((_self_._f, args, kwargs)),
            **_self_._options,
        )
        future = RemoteFuture(session=_self_._session, id=id_)
        _self_._futures.append(future)
        return future

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        """Wait all external futures."""
        [f.result() for f in self._futures]


class RemoteFuture(typing.Generic[OUT]):
    """Remote future for remote pool."""

    id: str
    """Remote container id."""

    _session: Session
    """Remote session."""

    def __init__(self, id: str, session: Session):
        self.id = id
        self._session = session

    def __str__(self):
        return f"<Remote future ({self.id})>"

    def result(self) -> OUT:
        """Fetch result and load log from remote container data.

        :return: Function output
        """
        output_dump, log_dump = self._session.fetch_output(id=self.id)
        _load_log(log_dump=log_dump, id=self.id)
        return pickle.loads(output_dump)

    def input(
        self,
    ) -> tuple[typing.Callable[IN, OUT], tuple, dict]:
        """Get input data from remote container.

        :return RemoteInput: Function, args, kwargs
        """
        input_dump = self._session.fetch_input(id=self.id)
        f, f_args, f_kwargs = pickle.loads(input_dump)
        return f, f_args, f_kwargs

    def cancel(self) -> bool:
        """Cancel remote process."""
        return self._session.cancel(id=self.id)


class Session:
    """Helper class for push/pull dumps on/from remote container.

    Defines functions:
    - [de]serialize in/output data
    - run function from bytes (input data dump)

    Interface for:
    - submit function to run remote
    - stop remote container
    - get in/output data from remote container
    - standalone run function in container using external infrastructure
    """

    def push(
        self,
        input_dump: bytes,
        **options,
    ) -> str:
        """Push data to remote and start container.

        :param input_dump: Input data dump
        :param options: Options from Pool
        :return: Container id
        """
        ...

    def fetch_input(self, id: str) -> bytes:
        """Get input dump from remote container.

        :param id: Container id
        :return: Input dump bytes
        """
        ...

    def fetch_output(self, id: str) -> tuple[bytes, bytes]:
        """Get results from remote container.

        :param id: Container id
        :return: Function run results, log dump
        """
        ...

    def cancel(self, id: str) -> bool:
        """Shutdown certain container.

        :param id: Container id
        :return bool: Cancel result
        """
        ...

    def push_folder(
        self,
        session: Session,
        folder: Path,
        destination: Path,
    ) -> Path:
        """Push folder to remote.

        :param session: Remote session
        :param folder: Folder to push
        :param destination: Destination folder
        """
        ...


def remote_run(input_dump: bytes, log_file: Path) -> bytes:
    """Run function in the remote container.

    Wrap this function to run it standalone in container without arguments or outside context.
    Rebuild lib to run custom run functions.  Define files/data manipulations
    to create input data.

    EXAMPLE.
    simdel.my_addon.my_runner.py:
    ```
    import os
    from pathlib import Path
    from simdel.my_addon.pool import MyRunner

    if __name__ == "__main__":
        data_dir = Path(os.getenv("DATADIR"))
        input_dump = MyRunner.find_input(data_dir)
        log_file = MyRunner.find_log(data_dir)
        result = MyRunner.run(
            input_dump.read_bytes(),
            log_file,
        )
        out_file = MyRunner.get_out_file(data_dir)
        out_file.write_bytes(result)
    ```

    CLI in MyRunner.push :```mamba run -n env poetry run python -m simdel.my_addon.my_runner```

    :param input_dump: Input data dump
    :param log_file: Log file path to save log records
    :return: Output data or None (error case)
    """
    logger = logging.getLogger()
    logger.addHandler(_DumpHandler(log_file))

    result = None
    f, args, kwargs = pickle.loads(input_dump)
    try:
        result = f(*args, **kwargs)
    except Exception as e:
        _log.critical(msg="Remote running fail", exc_info=e)
    return pickle.dumps(result)


def _load_log(log_dump: bytes, id: str) -> typing.Any:
    log_records: list[bytes] = []
    record_list: list[bytes] = []
    with io.BytesIO(log_dump) as file:
        datas = file.readlines()
        for i in datas:
            if i == _LOG_DELIMITER[1:]:
                log_records.append(b"".join(record_list))
                record_list = []
            else:
                record_list.append(i)

    for record in log_records:
        log_dict = pickle.loads(record)
        if log_id := log_dict.get("log_id"):
            log_dict["log_id"] = f"{log_id}:{id}"
        if log_dict.get("log_stream_id"):
            log_dict["log_stream_id"] = _log.LOG_STREAM_ID.get()

        log_record = logging.makeLogRecord(log_dict)
        if log_dict.get("exc_type"):
            log_record.msg = log_record.msg + "\n" + log_record.exc_traceback[:-1]  # type: ignore
        logger = logging.getLogger(log_record.name)
        logger.handle(log_record)


_LOG_DELIMITER = b"\n###\n"
"""Delimiter for log records in log dump."""


# TODO: to pipe
class _DumpHandler(logging.Handler):
    """Log handler to dump log record on remote container."""

    def __init__(self, log_file: Path):
        super().__init__()
        self.filename = log_file
        self.file = log_file.open("ab")

    def emit(self, record):
        log_dump = pickle.dumps(_prepare_for_pickle(record))
        self.file.write(log_dump)
        self.file.write(_LOG_DELIMITER)
        self.file.flush()

    def close(self):
        self.file.close()
        super().close()


def _prepare_for_pickle(record: logging.LogRecord) -> dict:
    """Catch exception because exceptions are not serializable.

    :param record: LogRecord object
    :return: Log record data dict
    """
    data = record.__dict__ | dict(exc_info=None)
    if record.exc_info:
        data = data | dict(
            exc_type=record.exc_info[0].__name__,  # type: ignore
            exc_message=str(record.exc_info[1]),
            exc_traceback="".join(traceback.format_exception(*record.exc_info)),
        )
    return data
