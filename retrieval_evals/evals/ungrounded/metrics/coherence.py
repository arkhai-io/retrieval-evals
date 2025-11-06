"""Semantic coherence metric using sentence embeddings.

Evaluates the semantic consistency and flow within an answer by measuring
how well consecutive sentences relate to each other.

Methodology:
1. Splits answer into individual sentences
2. Encodes each sentence into a dense vector embedding
3. Computes cosine similarity between consecutive sentence pairs
4. Returns statistics on inter-sentence coherence

Metrics Computed:
- **coherence_score**: Average similarity between consecutive sentences (primary metric)
- **min_coherence**: Lowest similarity (detects jarring transitions)
- **coherence_variance**: Consistency of flow throughout answer
- **topic_consistency**: Similarity of all sentences to the overall answer embedding

High coherence indicates:
- Smooth transitions between ideas
- Consistent topic throughout answer
- Logical progression of thought

Low coherence may indicate:
- Abrupt topic changes
- Disjointed or rambling responses
- Multiple unrelated ideas mixed together
"""

import re
from statistics import mean, variance
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval_evals.evals.ungrounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPair


class Coherence(Metric):
    """Semantic coherence metric using sentence embeddings.

    Measures how well consecutive sentences in an answer relate to each other
    semantically, indicating smooth flow and topical consistency.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize coherence metric.

        Args:
            embedding_model: SentenceTransformer model name
        """
        super().__init__(embedding_model=embedding_model)
        self.embedding_model_name = embedding_model
        self.encoder = SentenceTransformer(embedding_model)

    @property
    def name(self) -> str:
        """Return metric name."""
        return "coherence"

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r"[.!?]+", text)
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        return sentences

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norm_product == 0:
            return 0.0

        similarity = dot_product / norm_product
        return float(similarity)

    def _analyze_coherence(self, text: str) -> dict[str, Any]:
        """Analyze semantic coherence of text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with coherence metrics
        """
        if not text.strip():
            return {
                "coherence_score": 0.0,
                "min_coherence": 0.0,
                "coherence_variance": 0.0,
                "topic_consistency": 0.0,
                "num_sentences": 0,
                "consecutive_similarities": [],
            }

        sentences = self._split_sentences(text)

        # Need at least 2 sentences for coherence
        if len(sentences) < 2:
            return {
                "coherence_score": 1.0,  # Single sentence is perfectly coherent with itself
                "min_coherence": 1.0,
                "coherence_variance": 0.0,
                "topic_consistency": 1.0,
                "num_sentences": len(sentences),
                "consecutive_similarities": [],
            }

        # Encode all sentences
        embeddings = self.encoder.encode(sentences)

        # Encode full text for topic consistency
        full_text_embedding = self.encoder.encode([text])[0]

        # Compute consecutive sentence similarities
        consecutive_sims = []
        for i in range(len(embeddings) - 1):
            sim = self._compute_cosine_similarity(embeddings[i], embeddings[i + 1])
            consecutive_sims.append(sim)

        # Compute topic consistency (each sentence vs. full text)
        topic_sims = []
        for emb in embeddings:
            sim = self._compute_cosine_similarity(emb, full_text_embedding)
            topic_sims.append(sim)

        # Compute metrics
        coherence_score = mean(consecutive_sims) if consecutive_sims else 0.0
        min_coherence = min(consecutive_sims) if consecutive_sims else 0.0
        coherence_var = variance(consecutive_sims) if len(consecutive_sims) > 1 else 0.0
        topic_consistency = mean(topic_sims) if topic_sims else 0.0

        return {
            "coherence_score": coherence_score,
            "min_coherence": min_coherence,
            "coherence_variance": coherence_var,
            "topic_consistency": topic_consistency,
            "num_sentences": len(sentences),
            "consecutive_similarities": consecutive_sims,
        }

    def compute(self, qa_pairs: list[QAPair]) -> list[EvalResult]:
        """Compute coherence scores for Q&A pairs.

        Args:
            qa_pairs: List of Q&A pairs

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
        coherence_scores = []

        for pair in qa_pairs:
            analysis = self._analyze_coherence(pair["answer"])
            individual_scores.append(analysis)
            coherence_scores.append(analysis["coherence_score"])

        # Use average coherence score as overall score
        avg_coherence = mean(coherence_scores) if coherence_scores else 0.0

        return [
            {
                "metric_name": self.name,
                "score": avg_coherence,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "embedding_model": self.embedding_model_name,
                    "config": self.config,
                    "interpretation": {
                        "0.8-1.0": "Excellent coherence - smooth, logical flow",
                        "0.6-0.8": "Good coherence - generally consistent",
                        "0.4-0.6": "Moderate coherence - some disjointed sections",
                        "0.2-0.4": "Poor coherence - frequent topic jumps",
                        "0.0-0.2": "Very poor coherence - rambling or incoherent",
                    },
                },
            }
        ]
