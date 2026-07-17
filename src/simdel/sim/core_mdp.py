"""Base MDP structures."""

import enum
import functools
from pathlib import Path
import typing

from pydantic import BaseModel
from pydantic.functional_validators import BeforeValidator as _Validator

from simdel._misc import utils

# Field types
_custom_bool = enum.Flag


class _bool_tf(_custom_bool):
    true = True
    false = False

    def __bool__(self) -> bool:
        return bool(self.value)


class _bool_yn(_custom_bool):
    yes = True
    no = False

    def __bool__(self) -> bool:
        return bool(self.value)


class _bool_YN(_custom_bool):
    Y = True
    N = False

    def __bool__(self) -> bool:
        return bool(self.value)


# bool_yn = typing.Annotated[bool | _bool_yn, _Validator(_bool_yn)]
# """Bool, dumps to yes/no"""
# bool_tf = typing.Annotated[bool | _bool_tf, _Validator(_bool_tf)]
# """Bool, dumps to true/false"""
# bool_YN = typing.Annotated[bool | _bool_YN, _Validator(_bool_YN)]
# """Bool, dumps to Y/N"""

bool_yn = typing.Annotated[bool | _bool_yn, _Validator(bool)]
"""Bool, dumps to yes/no"""
bool_tf = typing.Annotated[bool | _bool_tf, _Validator(bool)]
"""Bool, dumps to true/false"""
bool_YN = typing.Annotated[bool | _bool_YN, _Validator(bool)]
"""Bool, dumps to Y/N"""


# MDP
class BaseMDP(BaseModel):
    """Representation of Gromacs MDP file."""

    name: str

    include: list[str] = []
    """Directories to include in your topology. NOT used in md library."""
    define: list[str] = []
    """Defines to pass to the preprocessor. NOT used in md library."""

    def __repr__(self) -> str:
        fields = " ".join(
            f"{key}={value}" for key, value in dict(self).items() if (value not in [None, []])
        )
        return f"<MDP {self.name}: {fields}>"

    def __str__(self) -> str:
        return self.__repr__()

    def save(self, save_dir: Path) -> Path:
        """Dump simulation parameters to file.

        :param save_dir: Save dir path
        :return: Dump .mdp file
        """
        config_file = save_dir / f"{self.name}.mdp"
        utils.backup(config_file)
        config_file.write_text(self.dump())
        return config_file

    def dump(self) -> str:
        """Dump simulation parameters.

        :return: Dump text
        """
        return "\n".join(
            [
                f"{key} = {self.dump_field(value)}"
                for key, value in dict(self).items()
                if (key != "name") and (value not in [None, []])
            ]
        )

    @functools.singledispatchmethod
    def dump_field(self, value: float | str | bool) -> str:
        """Dump mdp field polymorph function.

        :param value: Field value
        :return: Field dump
        """
        return str(value)

    @dump_field.register
    def _(self, value: bool):
        return "yes" if value else "no"

    @dump_field.register
    def _(self, value: _custom_bool):
        if not value.name:
            msg = f"{value} has incorrect name"
            raise ValueError(msg)
        return value.name

    @dump_field.register
    def _(self, value: enum.Enum):
        return value.value

    @dump_field.register
    def _(self, value: list):
        return " ".join(self.dump_field(i) for i in value)
