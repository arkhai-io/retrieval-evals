"""Evaluation modules for retrieval systems.

This package contains two types of evaluations:
- grounded: Evaluations that compare against gold standard answers
- ungrounded: Evaluations that don't require gold standard answers
"""

from retrieval_evals.evals.grounded import evaluate_grounded
from retrieval_evals.evals.ungrounded import evaluate_ungrounded

__all__ = [
    "evaluate_grounded",
    "evaluate_ungrounded",
]
