"""Dynamic cell management for autoregressive development.

Handles cell division and deletion with variable sequence lengths.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .states import BranchingState


@dataclass
class EventDecision:
    """Decision for cell events."""

    split_probs: torch.Tensor  # [B, L] probability of division
    del_probs: torch.Tensor  # [B, L] probability of deletion
    split_samples: torch.Tensor  # [B, L] binary decisions (0 or 1)
    del_samples: torch.Tensor  # [B, L] binary decisions (0 or 1)


class DynamicCellManager:
    """Manage dynamic cell population with division and deletion.

    Handles:
    - Cell division: 1 cell → 2 daughter cells
    - Cell deletion: mark and remove cells
    - Variable sequence lengths across batches
    - Differentiable event sampling for training
    """

    def __init__(
        self,
        split_threshold: float = 0.5,
        del_threshold: float = 0.5,
        max_cells: int = 1024,
        use_gumbel: bool = True,
        tau: float = 0.1,
    ):
        self.split_threshold = split_threshold
        self.del_threshold = del_threshold
        self.max_cells = max_cells
        self.use_gumbel = use_gumbel
        self.tau = tau  # Temperature for Gumbel-Softmax

    def sample_events(
        self,
        split_logits: torch.Tensor,
        del_logits: torch.Tensor,
        deterministic: bool = False,
    ) -> EventDecision:
        """Sample division and deletion events.

        Args:
            split_logits: [B, L, 1] logits for cell division
            del_logits: [B, L, 1] logits for cell deletion
            deterministic: If True, use threshold; else sample

        Returns:
            EventDecision with probabilities and samples
        """
        # Squeeze last dimension
        split_logits = split_logits.squeeze(-1)  # [B, L]
        del_logits = del_logits.squeeze(-1)  # [B, L]

        # Compute probabilities
        split_probs = torch.sigmoid(split_logits)
        del_probs = torch.sigmoid(del_logits)

        if deterministic:
            # Hard thresholding for inference
            split_samples = (split_probs > self.split_threshold).float()
            del_samples = (del_probs > self.del_threshold).float()
        else:
            # Differentiable sampling for training
            if self.use_gumbel:
                split_samples = self._gumbel_sigmoid(split_logits)
                del_samples = self._gumbel_sigmoid(del_logits)
            else:
                # Straight-through estimator
                split_hard = (split_probs > self.split_threshold).float()
                split_soft = split_probs
                split_samples = split_hard - split_soft.detach() + split_soft

                del_hard = (del_probs > self.del_threshold).float()
                del_soft = del_probs
                del_samples = del_hard - del_soft.detach() + del_soft

        return EventDecision(
            split_probs=split_probs,
            del_probs=del_probs,
            split_samples=split_samples,
            del_samples=del_samples,
        )

    def _gumbel_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        """Differentiable sigmoid sampling using Gumbel-Softmax trick.

        Args:
            logits: [B, L] logits

        Returns:
            [B, L] soft samples in [0, 1]
        """
        # For binary case, we can use Gumbel-Softmax with 2 classes
        # Then take the probability of class 1

        # Add Gumbel noise
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-10) + 1e-10)

        # Apply temperature
        z = (logits + gumbel_noise) / self.tau

        # Sigmoid = soft version of threshold
        return torch.sigmoid(z)

    def apply_events(
        self,
        state: BranchingState,
        events: EventDecision,
    ) -> BranchingState:
        """Apply division and deletion events to create new state.

        Args:
            state: Current BranchingState
            events: EventDecision with split/del samples

        Returns:
            New BranchingState with updated cell population
        """
        B = state.states[0].shape[0]
        device = state.states[0].device

        new_states_list = []

        for b in range(B):
            # Get current cells for this batch item
            n_cells = state.padmask[b].sum().item()

            if n_cells == 0:
                # Empty batch item
                new_states_list.append(None)
                continue

            # Get event decisions for valid cells
            split_decisions = events.split_samples[b, :n_cells]  # [n_cells]
            del_decisions = events.del_samples[b, :n_cells]  # [n_cells]

            # Determine which cells survive (not deleted)
            survive_mask = del_decisions < 0.5

            # Determine which cells divide
            divide_mask = split_decisions > 0.5

            # Collect new cells
            new_continuous = []
            new_discrete = []
            new_groupings = []
            new_ids = []

            cell_id_counter = 1

            for i in range(n_cells):
                if not survive_mask[i]:
                    # Cell is deleted
                    continue

                # Get current cell state
                cont = state.states[0][b, i]  # [gene_dim + spatial_dim]
                disc = state.states[1][b, i] if state.states[1] is not None else 0
                group = state.groupings[b, i]

                if divide_mask[i]:
                    # Cell divides: create two daughters with small perturbation
                    noise_scale = 0.01
                    daughter1 = cont + torch.randn_like(cont) * noise_scale
                    daughter2 = cont + torch.randn_like(cont) * noise_scale

                    new_continuous.append(daughter1)
                    new_continuous.append(daughter2)
                    new_discrete.extend([disc, disc])
                    new_groupings.extend([group, group])
                    new_ids.extend([cell_id_counter, cell_id_counter + 1])
                    cell_id_counter += 2
                else:
                    # Cell survives without division
                    new_continuous.append(cont)
                    new_discrete.append(disc)
                    new_groupings.append(group)
                    new_ids.append(cell_id_counter)
                    cell_id_counter += 1

            # Pad to max_cells if needed
            n_new = len(new_continuous)
            if n_new > self.max_cells:
                # Truncate if too many
                new_continuous = new_continuous[: self.max_cells]
                new_discrete = new_discrete[: self.max_cells]
                new_groupings = new_groupings[: self.max_cells]
                new_ids = new_ids[: self.max_cells]
                n_new = self.max_cells

            # Create tensors
            if n_new > 0:
                new_cont_tensor = torch.stack(new_continuous).unsqueeze(
                    0
                )  # [1, n_new, D]
                new_disc_tensor = torch.tensor(
                    new_discrete, dtype=torch.long, device=device
                ).unsqueeze(0)
                new_group_tensor = torch.tensor(
                    new_groupings, dtype=torch.long, device=device
                ).unsqueeze(0)
                new_id_tensor = torch.tensor(
                    new_ids, dtype=torch.long, device=device
                ).unsqueeze(0)

                # Padding
                if n_new < self.max_cells:
                    pad_size = self.max_cells - n_new
                    new_cont_tensor = F.pad(new_cont_tensor, (0, 0, 0, pad_size))
                    new_disc_tensor = F.pad(new_disc_tensor, (0, pad_size), value=0)
                    new_group_tensor = F.pad(new_group_tensor, (0, pad_size), value=0)
                    new_id_tensor = F.pad(new_id_tensor, (0, pad_size), value=0)

                    # Create mask
                    pad_mask = torch.cat(
                        [
                            torch.ones(1, n_new, dtype=torch.bool, device=device),
                            torch.zeros(1, pad_size, dtype=torch.bool, device=device),
                        ],
                        dim=1,
                    )
                else:
                    pad_mask = torch.ones(1, n_new, dtype=torch.bool, device=device)
            else:
                # All cells deleted - keep at least one placeholder
                new_cont_tensor = torch.zeros(
                    1, 1, state.states[0].shape[-1], device=device
                )
                new_disc_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
                new_group_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
                new_id_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
                pad_mask = torch.ones(
                    1, 1, dtype=torch.bool, device=device
                )  # Keep as valid but empty

            new_states_list.append(
                {
                    "continuous": new_cont_tensor,
                    "discrete": new_disc_tensor,
                    "groupings": new_group_tensor,
                    "ids": new_id_tensor,
                    "padmask": pad_mask,
                    "n_cells": n_new,
                }
            )

        # Batch the results
        max_n = max([s["n_cells"] if s is not None else 0 for s in new_states_list])

        batch_continuous = []
        batch_discrete = []
        batch_groupings = []
        batch_del_flags = []
        batch_ids = []
        batch_padmask = []
        batch_flowmask = []
        batch_branchmask = []

        for s in new_states_list:
            if s is None or s["n_cells"] == 0:
                # Empty batch item
                batch_continuous.append(
                    torch.zeros(1, max_n, state.states[0].shape[-1], device=device)
                )
                batch_discrete.append(
                    torch.zeros(1, max_n, dtype=torch.long, device=device)
                )
                batch_groupings.append(
                    torch.zeros(1, max_n, dtype=torch.long, device=device)
                )
                batch_del_flags.append(
                    torch.zeros(1, max_n, dtype=torch.bool, device=device)
                )
                batch_ids.append(torch.zeros(1, max_n, dtype=torch.long, device=device))
                batch_padmask.append(
                    torch.zeros(1, max_n, dtype=torch.bool, device=device)
                )
                batch_flowmask.append(
                    torch.zeros(1, max_n, dtype=torch.bool, device=device)
                )
                batch_branchmask.append(
                    torch.zeros(1, max_n, dtype=torch.bool, device=device)
                )
            else:
                # Extract and pad to max_n
                n = s["n_cells"]
                if n < max_n:
                    pad_size = max_n - n
                    batch_continuous.append(F.pad(s["continuous"], (0, 0, 0, pad_size)))
                    batch_discrete.append(F.pad(s["discrete"], (0, pad_size)))
                    batch_groupings.append(F.pad(s["groupings"], (0, pad_size)))
                    batch_del_flags.append(
                        torch.zeros(1, max_n, dtype=torch.bool, device=device)
                    )
                    batch_ids.append(F.pad(s["ids"], (0, pad_size)))
                    batch_padmask.append(
                        torch.cat(
                            [
                                torch.ones(1, n, dtype=torch.bool, device=device),
                                torch.zeros(
                                    1, pad_size, dtype=torch.bool, device=device
                                ),
                            ],
                            dim=1,
                        )
                    )
                    batch_flowmask.append(
                        torch.cat(
                            [
                                torch.ones(1, n, dtype=torch.bool, device=device),
                                torch.zeros(
                                    1, pad_size, dtype=torch.bool, device=device
                                ),
                            ],
                            dim=1,
                        )
                    )
                    batch_branchmask.append(
                        torch.cat(
                            [
                                torch.ones(1, n, dtype=torch.bool, device=device),
                                torch.zeros(
                                    1, pad_size, dtype=torch.bool, device=device
                                ),
                            ],
                            dim=1,
                        )
                    )
                else:
                    batch_continuous.append(s["continuous"])
                    batch_discrete.append(s["discrete"])
                    batch_groupings.append(s["groupings"])
                    batch_del_flags.append(
                        torch.zeros(1, max_n, dtype=torch.bool, device=device)
                    )
                    batch_ids.append(s["ids"])
                    batch_padmask.append(s["padmask"])
                    batch_flowmask.append(
                        torch.ones(1, max_n, dtype=torch.bool, device=device)
                    )
                    batch_branchmask.append(
                        torch.ones(1, max_n, dtype=torch.bool, device=device)
                    )

        # Stack into batch
        new_state = BranchingState(
            states=(
                torch.cat(batch_continuous, dim=0),
                torch.cat(batch_discrete, dim=0),
            ),
            groupings=torch.cat(batch_groupings, dim=0),
            del_flags=torch.cat(batch_del_flags, dim=0),
            ids=torch.cat(batch_ids, dim=0),
            padmask=torch.cat(batch_padmask, dim=0),
            flowmask=torch.cat(batch_flowmask, dim=0),
            branchmask=torch.cat(batch_branchmask, dim=0),
        )

        return new_state

    def compute_event_loss(
        self,
        events: EventDecision,
        target_splits: torch.Tensor | None = None,
        target_dels: torch.Tensor | None = None,
    ) -> dict:
        """Compute event prediction losses.

        Args:
            events: EventDecision with predictions
            target_splits: [B, L] ground truth split indicators
            target_dels: [B, L] ground truth deletion indicators

        Returns:
            Dictionary of losses
        """
        losses = {}

        if target_splits is not None:
            losses["split"] = F.binary_cross_entropy(
                events.split_probs,
                target_splits.float(),
            )
        else:
            losses["split"] = torch.tensor(0.0, device=events.split_probs.device)

        if target_dels is not None:
            losses["del"] = F.binary_cross_entropy(
                events.del_probs,
                target_dels.float(),
            )
        else:
            losses["del"] = torch.tensor(0.0, device=events.del_probs.device)

        return losses
