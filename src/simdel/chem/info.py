"""System classes and functions."""

from __future__ import annotations

from pydantic import BaseModel, model_validator

from . import ff_map


class SystemInfo(BaseModel, frozen=True, arbitrary_types_allowed=True):
    """System information container."""

    # TODO: replace to forcefield.type and validate water only in system
    ff_type: ff_map.FF | None
    """Forcefield type or None (unknown)"""

    water_type: ff_map.WaterType | None
    """Solvated info water type or None (non-solvated)."""

    water_flexibility: bool | None
    """Use flexible water topology or None (non-solvated)."""

    @model_validator(mode="after")
    def _validate(self):
        if self.water_type is not None and self.water_flexibility is None:
            msg = f"Water flexibility must be set if {self.water_type=}"
            raise ValueError(msg)
        elif self.water_type is None and self.water_flexibility is not None:
            msg = f"Water flexibility must be None if {self.water_type=}"
            raise ValueError(msg)

        if self.ff_type:
            self.ff_type.get_water_info(self.water_type)

        return self

    # TODO: desc
    def __add__(self, info: SystemInfo):  # noqa: ANN204
        """Mix info objects.

        :param info: Another info container
        """
        # Forcefield type
        mself = _mark_ff_type(self.ff_type)
        m = _mark_ff_type(info.ff_type)
        if mself == m and self.ff_type != info.ff_type:
            msg = f"Forcefield types can not mix together: {self.ff_type}, {info.ff_type}"
            raise ValueError(msg)
        elif mself >= m:
            new_ff_type = self.ff_type
        else:
            new_ff_type = info.ff_type

        # Water type
        mself = bool(self.water_type)
        m = bool(info.water_type)
        if mself == m and self.water_type != info.water_type:
            msg = (
                f"Different water types can not mix together: {self.water_type}, {info.water_type}"
            )
            raise ValueError(msg)
        elif mself >= m:
            new_water_type = self.water_type
        else:
            new_water_type = info.water_type

        # Water flexibility
        mself = _mark_trinary(self.water_flexibility)
        m = _mark_trinary(info.water_flexibility)
        if mself == m and self.water_flexibility != info.water_flexibility:
            msg = (
                f"Flexible and rigid waters can not mix together: {self.water_flexibility}, "
                f"{info.water_flexibility}"
            )
            raise ValueError(msg)
        elif mself >= m:
            new_water_flexibility = self.water_flexibility
        else:
            new_water_flexibility = info.water_flexibility

        return SystemInfo(
            ff_type=new_ff_type,
            water_type=new_water_type,
            water_flexibility=new_water_flexibility,
        )

    def __str__(self) -> str:
        ff_info = f"ff={self.ff_type.name if self.ff_type else 'unknown'}"
        if self.water_type:
            flex = "flexible" if self.water_flexibility else "rigid"
            water_info = f" {flex} {self.water_type.value}"
        else:
            water_info = ""
        return f"{ff_info}{water_info}"


def _mark_ff_type(ff_type: ff_map.FF | None):
    """Range forcefield types to corrext mix it.

    :param ff_type: Forcefield
    """
    if ff_type is None:
        return 0
    if isinstance(ff_type, ff_map.OpenFF):
        return 1
    return 2


def _mark_trinary(obj: bool | None):
    """Range water type info.

    Unknown < No water < Water

    :param obj: Description
    """
    if obj is None:
        return 0
    if obj is False:
        return 1
    return 2
