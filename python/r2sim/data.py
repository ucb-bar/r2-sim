"""The loader of the --trace-out folder of r2sim."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Trace:
    """The files of one --trace-out folder."""

    resources: pd.DataFrame
    tasks: pd.DataFrame
    chunks: pd.DataFrame
    marks: pd.DataFrame
    phases: pd.DataFrame
    tree: list

    @property
    def bin(self) -> int:
        """The bin width of `resources`, in cycles."""
        starts = sorted(self.resources["bin_start"].unique())
        steps = [b - a for a, b in zip(starts, starts[1:]) if b > a]
        return int(min(steps)) if steps else 1


def load_trace(folder: Path) -> Trace:
    """Reads the files that `r2sim run --trace-out folder` wrote."""
    folder = Path(folder)
    return Trace(
        resources=pd.read_csv(folder / "resources.csv.gz"),
        tasks=pd.read_csv(folder / "tasks.csv.gz"),
        chunks=pd.read_csv(folder / "chunks.csv.gz"),
        marks=pd.read_csv(folder / "marks.csv"),
        phases=pd.read_csv(folder / "phases.csv"),
        tree=json.loads((folder / "tree.json").read_text()),
    )
