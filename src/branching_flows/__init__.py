"""Branching Flows: variable-length generative modeling with splits and deletions.

Python/PyTorch port of BranchingFlows.jl -- a generator-matching framework
that extends flow matching to variable-length sequences by modeling
branching/coalescent processes over binary forests.
"""

from .flow import CoalescentFlow, branching_bridge, sample_forest
from .loss import (
    bregman_poisson_loss,
    deletion_loss,
    logit_bce_loss,
    loss_scale,
    split_loss,
)
from .merging import canonical_anchor_merge, select_anchor_merge
from .policies import BalancedSequential, SequentialUniform
from .processes import BrownianBridge, DiscreteInterpolatingFlow, OUFlow
from .sampling import Tracker, generate, step
from .states import (
    BranchingState,
    BridgeOutput,
    SampleState,
    fixedcount_del_insertions,
    group_fixedcount_del_insertions,
    uniform_del_insertions,
)
from .trees import FlowNode, add_child, merge_nodes
from .emergent_loss import (
    emergent_context_loss,
    sinkhorn_divergence,
    cell_count_loss,
    diversity_loss,
    weak_anchor_loss,
)

__all__ = [
    # Core
    "CoalescentFlow",
    "branching_bridge",
    "sample_forest",
    # States
    "BranchingState",
    "BridgeOutput",
    "SampleState",
    "FlowNode",
    # Processes
    "BrownianBridge",
    "OUFlow",
    "DiscreteInterpolatingFlow",
    # Policies
    "SequentialUniform",
    "BalancedSequential",
    # Losses
    "split_loss",
    "deletion_loss",
    "bregman_poisson_loss",
    "logit_bce_loss",
    "loss_scale",
    # Merging
    "canonical_anchor_merge",
    "select_anchor_merge",
    # Deletion augmentation
    "uniform_del_insertions",
    "fixedcount_del_insertions",
    "group_fixedcount_del_insertions",
    # Generation
    "step",
    "generate",
    "Tracker",
    # Trees
    "add_child",
    "merge_nodes",
    # Emergent Loss
    "emergent_context_loss",
    "sinkhorn_divergence",
    "cell_count_loss",
    "diversity_loss",
    "weak_anchor_loss",
]
