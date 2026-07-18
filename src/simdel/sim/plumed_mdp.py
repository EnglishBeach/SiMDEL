"""Plumed mdp parameters.

See documentation: https://www.plumed.org/doc-v2.9/user-doc/html/tutorials.html.
"""

from pathlib import Path
import shutil

from pydantic import BaseModel, model_validator

from simdel import _utils
from simdel._wrappers import plumed


# TODO: desc
@_utils.require(plumed)
class PlumedMDP(BaseModel):
    """REQUIRE PLUMED!
    Desc.
    """

    name: str = "plumed"
    """Config name."""

    text: str
    """Plumed text."""

    dt: float
    """Time delta, in `ps`."""

    out_data: list[str]
    """Output file names without suffix."""

    # TODO: refactor
    other_files: list[Path]
    """Files list used in plumed config."""

    @model_validator(mode="after")
    def _validate(self):
        """Validator."""
        names = [i.name for i in self.other_files]
        if len(set(names)) != len(names):
            msg = "Only unique plumed output files is allowed"
            raise ValueError(msg)
        return self

    def save(self, save_dir: Path) -> Path:
        """Dfgdsf.

        :param save_dir: _description_
        :return: _description_
        """
        config_file = save_dir / f"{self.name}.dat"
        _utils.backup(config_file)
        config_file.write_text(self.text)

        for file in self.other_files:
            if file.relative_to(save_dir):
                continue
            shutil.copy(file, save_dir / file.name)
        return config_file

    # TODO: rework
    def get_data(self, save_dir: Path) -> dict[str, Path]:
        """S.

        :param save_dir: _description_
        :return: _description_
        """
        return {i: save_dir / f"{i}.dat" for i in self.out_data}
