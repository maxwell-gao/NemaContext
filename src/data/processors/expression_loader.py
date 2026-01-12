"""
Expression loader for NemaContext data processors.

This module re-exports the ExpressionLoader from the builder module
for convenience when working with the processors API.
"""

from src.data.builder.expression_loader import ExpressionLoader

__all__ = ["ExpressionLoader"]
