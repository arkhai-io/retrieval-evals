"""Grounded evaluations (with gold standard answers).

This module contains evaluations that compare generated answers against
gold standard answers. Common metrics include:
- Exact match
- F1 score
- BLEU score
- ROUGE score
- Semantic similarity
"""

from retrieval_evals.evals.grounded.base import evaluate_grounded

__all__ = ["evaluate_grounded"]
