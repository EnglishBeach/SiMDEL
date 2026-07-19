"""Low-level base structures and run functions."""

from __future__ import annotations

from pathlib import Path
import subprocess
import typing

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from . import _log

T = typing.TypeVar("T", bound="Table")
STRICT: bool = False


class Table:
    """Immutable data table like pandas.DataFrame."""

    @property
    def index(self) -> pd.Index:
        """Table index."""
        return self._df.index

    def __init__(self, **kwargs):
        annotations = _get_annotations(self.__class__, {})

        if set(annotations) != set(kwargs.keys()):
            annotations_cols = set(annotations)
            df_cols = set(kwargs.keys())
            msg = (
                "Columns incorrect:\n"
                f"unnecessary: {df_cols - annotations_cols}\n"
                f"missing: {annotations_cols - df_cols}"
            )
            raise ValueError(msg)

        df = pd.DataFrame({key: kwargs[key] for key in annotations})

        # TODO: to_df better - change not allowed
        self._df = df.replace({np.nan: None})
        self._cols = annotations

    def __repr__(self) -> str:
        return repr(self._df)

    def _repr_html_(self):
        """HTML representation like in pandas.DataFrame."""
        return self._df._repr_html_()  # type: ignore

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, self.__class__):
            msg = "Compare different tables is not allowed"
            raise TypeError(msg)
        return hash(self) == hash(value)

    def __hash__(self) -> int:
        return hash(self._df.to_json())

    def __len__(self) -> int:
        return len(self._df)

    def __getattr__(self, attr: str) -> pd.Series:
        if (attr != "_cols") and (attr in self._cols):
            return self._df[attr]
        return self.__getattribute__(attr)

    def __setattr__(self, name: str, value: ...) -> None:
        if name in ["to_df", "_cols", "keys", "_df"]:
            return super().__setattr__(name, value)
        msg = "Table columns is immutable set"
        raise AttributeError(msg)

    @typing.overload
    def __getitem__(self: T, value: pd.Series[bool] | slice) -> T: ...

    @typing.overload
    def __getitem__(self, value: str) -> pd.Series: ...

    @typing.overload
    def __getitem__(self, value: int) -> pd.Series: ...

    def __getitem__(self, value):  # type: ignore
        if isinstance(value, (pd.Series, slice)):
            return self.__class__(**self._df[value])

        if isinstance(value, str):
            return self._df[value]

        if isinstance(value, int):
            return self._df.iloc[value]
        msg = f"Unsupported key type: {type(value)}"
        raise KeyError(msg)

    def __setitem__(self, index, value):
        self._df.loc[index] = value

    def to_df(self) -> pd.DataFrame:
        """Get pandas.DataFrame from Table.

        :return pd.DataFrame: andas.DataFrame
        """
        return self._df.copy()

    def keys(self) -> list[str]:
        """Table column names."""
        return list(self._cols.keys())


class PathContainer(BaseModel):
    """Container for output file paths to `simdel._wrappers` functions."""

    @model_validator(mode="before")
    def check(cls, values: dict) -> dict:
        """Check paths are exist, set None if don't exist."""
        for key, value in values.items():
            if isinstance(value, Path) and (not value.exists()):
                values[key] = None
        return values


def backup(file: Path) -> Path | None:
    """Check if file exists,
    if true - backup file to #<file name>
    if false - return None.

    :param file: File path
    :return: Backed up file path or None
    """
    if not file.exists():
        return None

    i = 0
    back_file = file.parent / f"#{file.name}"
    while back_file.exists():
        back_file = file.parent / f"#{file.name}{i}"
        i += 1
    msg = f"File backup {file.as_posix()} -> {back_file.name}"
    _log.warning(msg)
    file.rename(back_file)
    return back_file


def clear_backups(folder: Path):
    """Delete all backups in folder.

    :param folder: Folder to clear backups
    """
    for i in folder.iterdir():
        if i.name.startswith("#"):
            i.unlink()


def run(title: str, command: list[str], workdir: Path | None = None):
    """Run subprocess and log it.

    :param title: Process title
    :param command: Process cwd
    :param workdir: CWD path
    """
    with _log.context(msg=title, desc="\n".join([f"{workdir=}"] + command), level=_log.Level.DEBUG):
        cmd = " ".join(command)

        sp = subprocess.run(
            cmd,
            cwd=workdir,
            capture_output=True,
            check=False,
            shell=True,
        )
        if sp.returncode:
            raise RuntimeError("\n" + sp.stderr.decode())


def _get_annotations(cls: type, annots: dict[str, str]) -> dict[str, str]:
    if (cls == Table) or (not hasattr(cls, "__annotations__")):
        return annots

    annots = {
        i: val
        for i, val in cls.__annotations__.items()
        if not i.startswith("_") and (i not in annots)
    } | annots
    for base_cls in cls.__bases__:
        annots = _get_annotations(base_cls, annots)

    return annots
