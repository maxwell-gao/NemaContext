"""Parse WormGUIDES 4D nuclei tracking data into structured developmental events.

Parses 360 timepoint files from WormGUIDES to extract:
- Cell trajectories (name -> sequence of (t, x, y, z))
- Division events (parent -> child1, child2, t_division)
- Death events (cell -> t_death)
- Embryo snapshots (t -> list of alive cells with positions)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CellState:
    """A cell at a single timepoint."""

    name: str
    x: float
    y: float
    z: float
    diameter: float = 0.0


@dataclass
class DivisionEvent:
    """A cell division: parent splits into two children."""

    parent: str
    child1: str
    child2: str
    t_division: int  # timepoint when parent last appears


@dataclass
class DeathEvent:
    """A programmed cell death."""

    cell: str
    t_last: int  # last timepoint the cell appears


@dataclass
class WormGUIDESData:
    """Complete parsed WormGUIDES dataset."""

    cell_trajectories: dict[str, list[tuple[int, float, float, float]]]
    division_events: list[DivisionEvent]
    death_events: list[DeathEvent]
    death_set: set[str]
    snapshots: dict[int, list[CellState]]
    all_cell_names: list[str]
    n_timepoints: int

    # Constants
    START_TIME_MIN: float = 20.0
    TIME_RESOLUTION_MIN: float = 1.0

    def timepoint_to_minutes(self, tp: int) -> float:
        return self.START_TIME_MIN + (tp - 1) * self.TIME_RESOLUTION_MIN

    def cell_lifetime(self, name: str) -> tuple[int, int]:
        """Return (first_tp, last_tp) for a cell."""
        traj = self.cell_trajectories[name]
        return traj[0][0], traj[-1][0]

    def cells_at(self, tp: int) -> list[CellState]:
        """All named cells alive at timepoint tp."""
        return self.snapshots.get(tp, [])

    def cell_count_at(self, tp: int) -> int:
        return len(self.snapshots.get(tp, []))

    def get_founder(self, name: str) -> str:
        """Extract founder lineage from cell name."""
        if name.startswith("AB"):
            return "AB"
        for prefix in ("MS", "E", "C", "D", "P4", "P3", "P2", "P1", "P0", "Z2", "Z3"):
            if name.startswith(prefix):
                return prefix
        return "unknown"


# Child suffix pairs used in C. elegans lineage naming
_SUFFIX_PAIRS = [("a", "p"), ("l", "r"), ("d", "v")]


def parse_wormguides(
    nuclei_dir: str | Path,
    deaths_csv: str | Path | None = None,
) -> WormGUIDESData:
    """Parse WormGUIDES nuclei files into structured developmental data.

    Args:
        nuclei_dir: Path to directory containing t001-nuclei through t360-nuclei.
        deaths_csv: Path to CellDeaths.csv (optional).

    Returns:
        WormGUIDESData with all parsed trajectories, divisions, deaths, snapshots.
    """
    nuclei_dir = Path(nuclei_dir)

    # Load death list
    death_set: set[str] = set()
    if deaths_csv is not None:
        deaths_path = Path(deaths_csv)
        if deaths_path.exists():
            with open(deaths_path) as f:
                for line in f:
                    name = line.strip()
                    if name:
                        death_set.add(name)

    # Parse all timepoints
    cell_trajectories: dict[str, list[tuple[int, float, float, float]]] = {}
    snapshots: dict[int, list[CellState]] = {}

    n_timepoints = 0
    for tp in range(1, 500):
        fpath = nuclei_dir / f"t{tp:03d}-nuclei"
        if not fpath.exists():
            break
        n_timepoints = tp

        cells_at_tp: list[CellState] = []
        with open(fpath) as f:
            for line in f:
                parts = line.strip().rstrip(",").split(",")
                if len(parts) < 10:
                    continue
                name = parts[9].strip().strip('"')
                if not name or name == "Nuc":
                    continue
                try:
                    x = float(parts[5].strip())
                    y = float(parts[6].strip())
                    z = float(parts[7].strip())
                    diam = float(parts[8].strip()) if len(parts) > 8 else 0.0
                except (ValueError, IndexError):
                    continue

                cells_at_tp.append(CellState(name=name, x=x, y=y, z=z, diameter=diam))

                if name not in cell_trajectories:
                    cell_trajectories[name] = []
                cell_trajectories[name].append((tp, x, y, z))

        snapshots[tp] = cells_at_tp

    # Detect division events
    division_events: list[DivisionEvent] = []
    cell_names_set = set(cell_trajectories.keys())

    for name in sorted(cell_names_set):
        traj = cell_trajectories[name]
        last_tp = traj[-1][0]

        for s1, s2 in _SUFFIX_PAIRS:
            child1 = name + s1
            child2 = name + s2
            if child1 in cell_names_set and child2 in cell_names_set:
                division_events.append(
                    DivisionEvent(
                        parent=name,
                        child1=child1,
                        child2=child2,
                        t_division=last_tp,
                    )
                )
                break

    # Detect death events
    death_events: list[DeathEvent] = []
    for name in sorted(death_set):
        if name in cell_trajectories:
            traj = cell_trajectories[name]
            death_events.append(DeathEvent(cell=name, t_last=traj[-1][0]))

    all_names = sorted(cell_names_set)

    return WormGUIDESData(
        cell_trajectories=cell_trajectories,
        division_events=division_events,
        death_events=death_events,
        death_set=death_set,
        snapshots=snapshots,
        all_cell_names=all_names,
        n_timepoints=n_timepoints,
    )
