"""WormGUIDES dataset for BROT training with real developmental dynamics.

Unlike NemaDataset (which assembles pseudo-embryos from cells of different
embryos), WormGUIDESDataset provides REAL embryo snapshots from a single
tracked embryo across 360 timepoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .states import SampleState
from .wormguides_parser import WormGUIDESData, parse_wormguides

_FOUNDERS = ("AB", "MS", "E", "C", "D", "P4")
_FOUNDER_TO_IDX = {f: i for i, f in enumerate(_FOUNDERS)}
_K = len(_FOUNDERS) + 1


class WormGUIDESDataset:
    """Dataset of real embryo snapshots from WormGUIDES 4D tracking."""

    def __init__(
        self,
        nuclei_dir: str | Path,
        deaths_csv: str | Path | None = None,
        min_cells: int = 4,
        stride: int = 5,
        include_velocity: bool = False,
    ):
        self.data = parse_wormguides(nuclei_dir, deaths_csv)
        self.include_velocity = include_velocity
        self.K = _K

        valid_tps = [
            tp for tp in range(1, self.data.n_timepoints + 1)
            if self.data.cell_count_at(tp) >= min_cells
        ]
        self.timepoints = valid_tps[::stride]

        all_coords = []
        for traj in self.data.cell_trajectories.values():
            for _, x, y, z in traj:
                all_coords.append((x, y, z))
        coords_arr = np.array(all_coords, dtype=np.float32)
        self._spatial_min = coords_arr.min(axis=0)
        self._spatial_max = coords_arr.max(axis=0)
        self._spatial_range = (self._spatial_max - self._spatial_min).clip(min=1e-6)

        self.continuous_dim = 6 if include_velocity else 3

    def __len__(self):
        return len(self.timepoints)

    def __getitem__(self, idx):
        tp = self.timepoints[idx]
        cells = self.data.cells_at(tp)

        elements = []
        del_flags = []
        ids = []

        for i, cell in enumerate(cells):
            xyz = np.array([cell.x, cell.y, cell.z], dtype=np.float32)
            xyz_norm = (xyz - self._spatial_min) / self._spatial_range

            if self.include_velocity:
                vel = self._compute_velocity(cell.name, tp)
                continuous = torch.from_numpy(np.concatenate([xyz_norm, vel]))
            else:
                continuous = torch.from_numpy(xyz_norm)

            founder = self.data.get_founder(cell.name)
            discrete = _FOUNDER_TO_IDX.get(founder, _K - 1)

            elements.append((continuous, discrete))
            del_flags.append(cell.name in self.data.death_set)
            ids.append(i + 1)

        n = len(elements)
        return SampleState(
            elements=elements,
            groupings=[0] * n,
            del_flags=del_flags,
            ids=ids,
            branchmask=[True] * n,
            flowmask=[True] * n,
        )

    def x0_sampler(self, root):
        center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        noise = np.random.randn(3).astype(np.float32) * 0.1
        xyz = center + noise
        if self.include_velocity:
            vel = np.zeros(3, dtype=np.float32)
            return (torch.from_numpy(np.concatenate([xyz, vel])), _K - 1)
        return (torch.from_numpy(xyz), _K - 1)

    def get_division_events(self):
        return self.data.division_events

    def get_death_set(self):
        return self.data.death_set

    def get_cell_names_at(self, idx):
        tp = self.timepoints[idx]
        return [c.name for c in self.data.cells_at(tp)]

    def _compute_velocity(self, name, tp):
        traj = self.data.cell_trajectories.get(name)
        if traj is None or len(traj) < 2:
            return np.zeros(3, dtype=np.float32)
        positions = {t: (x, y, z) for t, x, y, z in traj}
        if tp not in positions:
            return np.zeros(3, dtype=np.float32)
        cur = np.array(positions[tp], dtype=np.float32)
        if tp + 1 in positions:
            nxt = np.array(positions[tp + 1], dtype=np.float32)
            vel = (nxt - cur) / self._spatial_range
        elif tp - 1 in positions:
            prv = np.array(positions[tp - 1], dtype=np.float32)
            vel = (cur - prv) / self._spatial_range
        else:
            vel = np.zeros(3, dtype=np.float32)
        return vel.astype(np.float32)
