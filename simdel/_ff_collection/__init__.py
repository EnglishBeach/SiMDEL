"""Collection of standard GROMACS forcefield paths."""

from pathlib import Path

_file = Path(__file__).parent

AMBER94 = _file / "GROMACSamber94.ff"
AMBER96 = _file / "GROMACSamber96.ff"
AMBER99 = _file / "GROMACSamber99.ff"
AMBER99sb = _file / "GROMACSamber99sb.ff"
AMBER99sb_ildn = _file / "GROMACSamber99sb-ildn.ff"
AMBER03 = _file / "GROMACSamber03.ff"
AMBERGS = _file / "GROMACSamberGS.ff"
AMBER14sb_OL24 = _file / "GROMACSamber14sb_OL24.ff"
