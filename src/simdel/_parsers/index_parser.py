"""GROMACS index .ndx file parser."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def parse_index(index: Path, n_atoms: int) -> dict[str, pd.Series[bool]]:
    """Parse .ndx file.

    :param index: Index .ndx file path
    :param n_atoms: N atoms in geometry file
    :return: Selections dict
    """
    raw_indexes = [i.strip() for i in index.read_text().replace("\n", " ").split("[")]
    indexes = {}
    for i in raw_indexes:
        if i:
            name, mask_text = i.split("]")
            name = name.strip()
            index_set = {int(i) for i in mask_text.strip().split()}
            mask: pd.Series[bool] = pd.Series(
                [i in index_set for i in range(1, n_atoms + 1)],
            )
            indexes[name] = mask
    return indexes


def dump_index(name: str, mask: pd.Series[bool]) -> list[str]:
    """Dump selection to index .ndx file strings.

    :param name: Index name
    :param mask: Index mask
    :return: Index .ndx text
    """
    lines = []

    index = mask[mask].index + 1
    lines.append(f"[ {name} ]")
    lines.extend(
        " ".join(str(i) for i in chunk)
        for chunk in (list(index[i : i + 20]) for i in range(0, len(index), 20))
    )
    lines.append("\n")
    return lines
