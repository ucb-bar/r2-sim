"""Compares two golden runs (tools/golden.sh): the first line of each log
without the wall time, and each --stats JSON without the wall time, the
event rate and the functional worker's progress at each load."""

import json
import re
import sys
from pathlib import Path


def summary(path: Path) -> str:
    return re.sub(r"wall [0-9.]+ s, ", "", path.read_text().splitlines()[0])


def stats(path: Path) -> dict:
    d = json.loads(path.read_text())
    d.pop("wall_seconds", None)
    d.pop("events_per_second", None)
    for load in d.get("stats", {}).get("loads", []):
        load.pop("applied", None)
    return d


def main() -> int:
    ref, out = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0
    for log in sorted(ref.glob("*.log")):
        new = out / log.name
        if summary(log) != summary(new):
            print(f"DIFF {log.stem}: {summary(log)}\n     {' ' * len(log.stem)}  {summary(new)}")
            bad += 1
        elif stats(log.with_suffix(".json")) != stats(new.with_suffix(".json")):
            print(f"DIFF {log.stem}: JSON")
            bad += 1
        else:
            print(f"same {log.stem}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
