"""Archived trimodal and lineage-biased modeling modules.

These modules are retained for historical reference and legacy experiments.
Active development should import from `src.branching_flows`, not from this
subpackage.
"""

from .cross_lineage_probe import CrossLineageProbe, run_cross_lineage_discovery
from .crossmodal_model import CrossModalFusion, CrossModalNemaModel
from .lineage import (
    apply_lineage_bias_to_attention,
    batch_lineage_bias,
    compute_lineage_bias,
    lineage_distance,
    parse_lineage_name,
)
from .nema_model import NemaFlowModel
from .trimodal_loss import (
    curriculum_trimodal_loss,
    masked_mse_loss,
    trimodal_context_loss,
    weak_anchor_loss_masked,
)

__all__ = [
    "CrossLineageProbe",
    "run_cross_lineage_discovery",
    "CrossModalFusion",
    "CrossModalNemaModel",
    "apply_lineage_bias_to_attention",
    "batch_lineage_bias",
    "compute_lineage_bias",
    "lineage_distance",
    "parse_lineage_name",
    "NemaFlowModel",
    "curriculum_trimodal_loss",
    "masked_mse_loss",
    "trimodal_context_loss",
    "weak_anchor_loss_masked",
]
