"""ROUGE metric for lexical overlap evaluation.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures recall-based
n-gram overlap. Unlike BLEU's precision focus, ROUGE emphasizes how much of the
reference answer is captured by the generated answer.

Key characteristics:
- Recall-oriented: penalizes missing information from reference
- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap
- ROUGE-L: longest common subsequence (captures word order)
- Porter stemmer optional (matches word stems, e.g., "running" = "run")

Score range: 0.0 (no overlap) to 1.0 (perfect recall)
Best for: Ensuring generated answer covers reference content
"""

from rouge_score import rouge_scorer

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class ROUGE(Metric):
    """ROUGE score metric measuring recall-oriented n-gram overlap."""

    def __init__(self, rouge_type: str = "rougeL", use_stemmer: bool = True):
        """Initialize ROUGE metric.

        Args:
            rouge_type: ROUGE variant (rouge1, rouge2, rougeL, rougeLsum)
            use_stemmer: Apply Porter stemmer (default: True)
        """
        super().__init__(rouge_type=rouge_type, use_stemmer=use_stemmer)
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=use_stemmer)

    @property
    def name(self) -> str:
        """Return metric name."""
        return self.rouge_type.lower()

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute ROUGE scores.

        Args:
            qa_pairs: List of Q&A pairs with gold standard answers

        Returns:
            List with single evaluation result
        """
        if not qa_pairs:
            return [
                {
                    "metric_name": self.name,
                    "score": 0.0,
                    "individual_scores": [],
                    "metadata": {"num_pairs": 0, "config": self.config},
                }
            ]

        individual_scores = []
        for pair in qa_pairs:
            scores = self.scorer.score(pair["gold_answer"], pair["answer"])
            # Use F1 score as primary metric
            f1_score = scores[self.rouge_type].fmeasure
            individual_scores.append(float(f1_score))

        avg_score = sum(individual_scores) / len(individual_scores)

        return [
            {
                "metric_name": self.name,
                "score": avg_score,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "rouge_type": self.rouge_type,
                    "config": self.config,
                },
            }
        ]
