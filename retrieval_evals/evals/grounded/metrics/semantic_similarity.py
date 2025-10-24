"""Semantic similarity metric using sentence transformers."""

from sentence_transformers import CrossEncoder, SentenceTransformer

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class SemanticSimilarity(Metric):
    """Semantic similarity using bi-encoder or cross-encoder models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_type: str = "bi-encoder",
    ):
        """Initialize semantic similarity metric.

        Args:
            model_name: Sentence transformer model name
            model_type: Either "bi-encoder" or "cross-encoder"
        """
        super().__init__(model_name=model_name, model_type=model_type)
        if model_type not in ["bi-encoder", "cross-encoder"]:
            raise ValueError("model_type must be 'bi-encoder' or 'cross-encoder'")

        self.model_name = model_name
        self.model_type = model_type
        self.model: SentenceTransformer | CrossEncoder = (
            SentenceTransformer(model_name)
            if model_type == "bi-encoder"
            else CrossEncoder(model_name)
        )

    @property
    def name(self) -> str:
        """Return metric name."""
        return f"semantic_similarity_{self.model_type}"

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute semantic similarity scores.

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
        if self.model_type == "bi-encoder" and isinstance(self.model, SentenceTransformer):
            # Bi-encoder: compute embeddings and cosine similarity
            answers = [p["answer"] for p in qa_pairs]
            gold_answers = [p["gold_answer"] for p in qa_pairs]

            answer_embeddings = self.model.encode(answers, convert_to_tensor=True)
            gold_embeddings = self.model.encode(gold_answers, convert_to_tensor=True)

            similarities = self.model.similarity(answer_embeddings, gold_embeddings)
            individual_scores = [float(similarities[i][i]) for i in range(len(qa_pairs))]
        elif isinstance(self.model, CrossEncoder):
            # Cross-encoder: score pairs directly
            pairs = [(p["answer"], p["gold_answer"]) for p in qa_pairs]
            scores = self.model.predict(pairs)
            individual_scores = [
                float(s.item()) if hasattr(s, "item") else float(s) for s in scores
            ]

        avg_score = sum(individual_scores) / len(individual_scores)

        return [
            {
                "metric_name": self.name,
                "score": avg_score,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "model_name": self.model_name,
                    "config": self.config,
                },
            }
        ]
