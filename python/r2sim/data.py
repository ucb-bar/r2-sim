"""Loaders for r2sim outputs: the --stats JSON and the --trace-out folder."""

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


def load_run(path: Path) -> dict:
    """Reads one --stats JSON."""
    return json.loads(Path(path).read_text())


def run_label(run: dict) -> tuple[str, str]:
    """(configuration file name, weight type) of a run."""
    config = Path(run["config"]).name
    params = (run.get("forward") or {}).get("params") or (run.get("prefill") or {}).get("params")
    if params is None:
        params = (run.get("report") or {}).get("params") or {}
    return config, params.get("weight_dtype", "bf16")


def acc_buffer(run: dict) -> int:
    """The Atlas accumulator buffer of a run, in bytes."""
    return int(run.get("acc_buffer", 0))
