"""Module for logging functions. Default logging Logger fields +:
- `desc` - description
- `log_id` - start-end link
- `log_stream_id` - additional int to separate log streams (global context variable).
"""

from __future__ import annotations

import contextlib
import contextvars
import enum
import logging
import threading
import traceback

logger = logging.getLogger("simdel")
"""MD library logger."""

logger.setLevel(logging.DEBUG)

_LOG_ID: int = -1
"""Global var to link the start and the end log notes together"""

_log_lock: threading.Lock = threading.Lock()
"""Log id increment lock"""

_LOG_STREAM_ID_DEFAULT: str = "-1"
"""LOG_STREAM_ID"""

LOG_STREAM_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_LOG_STREAM_ID", default=_LOG_STREAM_ID_DEFAULT
)
"""Context var to separate log streams."""


class Level(enum.Enum):
    """Logging level from .logging lib."""

    CRITICAL = logging.CRITICAL
    """`CRITICAL` logging level."""

    ERROR = logging.ERROR
    """`ERROR` logging level."""

    WARNING = logging.WARNING
    """`WARNING` logging level."""

    INFO = logging.INFO
    """`INFO` logging level."""

    DEBUG = logging.DEBUG
    """`DEBUG` logging level."""

    NOTSET = logging.NOTSET
    """`NOTSET` logging level."""


class LogStreamFilter(logging.Filter):
    """Filter log records only current LOG_STREAM_ID."""

    def __init__(self, default_stream: bool, name: str = ""):
        self.log_stream_id = _LOG_STREAM_ID_DEFAULT if default_stream else LOG_STREAM_ID.get()
        super().__init__(name)

    def filter(self, record):
        """Filter logging records.

        :param record: Logging record
        :return: Mask (bool)
        """
        if hasattr(record, "log_stream_id"):
            return record.log_stream_id == self.log_stream_id  # type: ignore
        return False


def critical(msg: str, exc_info: Exception, desc: str = ""):
    """Log message with severity 'CRITICAL'.

    :param msg: Message string
    :param exc_info: Exception info
    """
    _log(msg=msg, desc=desc, exc_info=exc_info, log_id=_get_log_id(), level=Level.CRITICAL)


def error(msg: str, exc_info: Exception, desc: str = ""):
    """Log message with severity 'ERROR'.

    :param msg: Message string
    :param exc_info: Exception info
    """
    _log(msg=msg, desc=desc, exc_info=exc_info, log_id=_get_log_id(), level=Level.ERROR)


def warning(msg: str, desc: str = "", exc_info: Exception | None = None):
    """Log message with severity 'WARNING'.

    :param msg: Message string
    """
    _log(msg=msg, desc=desc, exc_info=exc_info, log_id=_get_log_id(), level=Level.WARNING)


def info(msg: str, desc: str = ""):
    """Log message with severity 'INFO'.

    :param msg: Message string
    :param commands: CLI commands
    """
    _log(msg=msg, desc=desc, log_id=_get_log_id(), level=Level.INFO)


def debug(msg: str, desc: str = ""):
    """Log message with severity 'DEBUG'.

    :param msg: Message string
    :param commands: CLI commands
    """
    _log(msg=msg, desc=desc, log_id=_get_log_id(), level=Level.DEBUG)


@contextlib.contextmanager
def context(msg: str, level: Level, desc: str = ""):
    """Context for long process. Link start and finish log records.
    Exclude itself from traceback.

    :param msg: Logging message
    :param level: Logging level
    :param desc: Additional information for `desc` field in log record
    """
    log_id = _get_log_id()
    _log(msg="  " + msg, desc=desc, log_id=log_id, level=level)

    try:
        yield
    except Exception as e:
        informative_tb = traceback.format_exception(e)[2:]
        _log(
            msg="X " + msg,
            desc=desc + "".join(informative_tb),
            log_id=log_id,
            level=Level.ERROR,
        )
        raise
    else:
        _log(
            msg="V " + msg,
            desc=desc,
            log_id=log_id,
            level=level,
        )


@contextlib.contextmanager
def log_stream_context(log_stream_id: str = _LOG_STREAM_ID_DEFAULT):
    """Context to set LOG_STREAM_ID during process.

    :param log_stream_id: LOG_STREAM_ID, defaults to _LOG_STREAM_ID_DEFAULT
    """
    token = LOG_STREAM_ID.set(log_stream_id)
    try:
        yield
    finally:
        LOG_STREAM_ID.reset(token)


def _log(
    msg: str,
    desc: str,
    level: Level,
    log_id: str,
    exc_info: Exception | None = None,
):
    """General log function, write logging record.

    :param msg: Log message
    :param desc: Additional information
    :param level: Log level
    :param log_id: LOG_ID variable, connect start and finish log records together
    :param exc_info: Exception, defaults to None
    """
    logger.log(
        msg=msg,
        level=level.value,
        exc_info=exc_info,
        extra=dict(
            desc=desc,
            log_stream_id=str(LOG_STREAM_ID.get()),
            log_id=log_id,
        ),
    )


def _get_log_id() -> str:
    """Get current LOG ID.

    :return: LOG ID string
    """
    global _LOG_ID
    with _log_lock:
        _LOG_ID = _LOG_ID + 1
    return str(_LOG_ID)
