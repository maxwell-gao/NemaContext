"""
Lineage encoder for data processing.

This module re-exports the LineageEncoder from the builder module
for convenience when used in data processing pipelines.
"""

from src.data.builder.lineage_encoder import (
    FOUNDER_CELLS,
    FOUNDER_PREFIXES,
    LineageEncoder,
)

__all__ = [
    "LineageEncoder",
    "FOUNDER_CELLS",
    "FOUNDER_PREFIXES",
]
