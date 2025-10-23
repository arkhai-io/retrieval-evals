"""Ungrounded evaluations (without gold standard answers).

This module contains evaluations that don't require gold standard answers.
Common metrics include:
- Answer relevance
- Faithfulness to context
- Coherence
- Consistency
- LLM-as-judge scoring
"""

from retrieval_evals.evals.ungrounded.base import evaluate_ungrounded

__all__ = ["evaluate_ungrounded"]
