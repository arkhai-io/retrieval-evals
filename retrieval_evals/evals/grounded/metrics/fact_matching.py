"""LLM-based fact extraction with semantic similarity matching.

This metric evaluates answer quality by comparing atomic facts extracted from both
the model's answer and a gold standard reference answer.

Methodology:
1. **Fact Extraction (LLM-based):**
   - Uses an LLM to decompose both model and gold answers into atomic facts
   - Each fact is a discrete, verifiable statement
   - Facts are self-contained and independent of context

2. **Semantic Encoding:**
   - Converts all extracted facts into dense vector embeddings using SentenceTransformers
   - Embeddings capture semantic meaning beyond surface-level text similarity

3. **Similarity Matrix Computation:**
   - Computes cosine similarity between all pairs of model and gold facts
   - Creates an N×M matrix where entry (i,j) is similarity between model_fact[i] and gold_fact[j]

4. **1:1 Bipartite Matching:**
   - Greedy: Iteratively matches highest similarity pairs above threshold
   - Optimal: Uses Hungarian algorithm for globally optimal assignment
   - Each fact can match to at most one fact from the other set

5. **Metrics Calculation:**
   - Precision: What fraction of model facts are correct (matched to gold)?
   - Recall: What fraction of gold facts are covered by model?
   - F1 Score: Harmonic mean of precision and recall (primary metric)

Output Structure:
- Overall F1 score averaged across all Q&A pairs
- Individual results per pair with:
  * Extracted facts from model and gold answers
  * Matched fact pairs with similarity scores
  * Unmatched facts from both sides
  * Per-pair precision, recall, F1
  * Full similarity matrix for analysis
"""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class FactMatching(Metric):
    """LLM-based fact extraction with embedding-based matching.

    This metric provides a comprehensive evaluation of answer quality by extracting
    atomic facts from both the model's generated answer and a reference (gold) answer,
    then performing semantic similarity-based matching to compute precision, recall, and F1.

    The metric is particularly useful when:
    - You want to understand which specific facts are correct vs. incorrect
    - You need fine-grained analysis beyond simple text similarity
    - You want to identify both missing information (low recall) and hallucinations (low precision)
    - You have reference answers that may be phrased differently than model outputs

    Key Features:
    - Uses LLM for robust fact extraction (handles paraphrasing, inference)
    - Semantic similarity matching (not exact string matching)
    - 1:1 bipartite matching ensures fair comparison
    - Detailed output shows which facts matched and which didn't
    - Configurable similarity threshold and matching strategy

    Example:
        >>> metric = FactMatching(
        ...     api_key="your-openrouter-key",
        ...     similarity_threshold=0.75,
        ...     matching_strategy="greedy"
        ... )
        >>> results = metric.compute(qa_pairs)
        >>> print(f"F1 Score: {results[0]['score']:.3f}")
    """

    def __init__(
        self,
        api_key: str,
        llm_model: str = "anthropic/claude-3.5-sonnet",
        embedding_model: str = "all-MiniLM-L6-v2",
        base_url: str = "https://openrouter.ai/api/v1",
        similarity_threshold: float = 0.75,
        matching_strategy: Literal["greedy", "optimal"] = "greedy",
    ):
        """Initialize fact matching metric.

        Args:
            api_key: OpenRouter API key for LLM
            llm_model: Model identifier on OpenRouter for fact extraction
            embedding_model: SentenceTransformer model for embeddings
            base_url: OpenRouter API base URL
            similarity_threshold: Minimum cosine similarity to consider a match
            matching_strategy: 'greedy' or 'optimal' (Hungarian algorithm)
        """
        super().__init__(
            llm_model=llm_model,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            matching_strategy=matching_strategy,
        )
        self.api_key = api_key
        self.llm_model = llm_model
        self.base_url = base_url
        self.similarity_threshold = similarity_threshold
        self.matching_strategy = matching_strategy

        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "fact_extraction.txt"
        self.prompt_template = prompt_path.read_text()

        # Initialize embedding model
        self.encoder = SentenceTransformer(embedding_model)

    @property
    def name(self) -> str:
        """Return metric name."""
        return "fact_matching"

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM via OpenRouter API to extract facts.

        Args:
            prompt: Formatted prompt string

        Returns:
            Dictionary with 'facts' list
        """
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=60,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed: dict[str, Any] = json.loads(content)
        return parsed

    def _extract_facts(self, question: str, answer: str, answer_type: str) -> list[str]:
        """Extract atomic facts from an answer using LLM.

        Args:
            question: The question being answered
            answer: The answer text to extract facts from
            answer_type: 'model' or 'gold' for better prompting

        Returns:
            List of atomic fact strings
        """
        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            answer_type=answer_type,
        )

        try:
            result = self._call_llm(prompt)
            facts: list[str] = result.get("facts", [])
            return facts
        except Exception as e:
            print(f"Error extracting facts from {answer_type} answer: {e}")
            return []

    def _compute_similarity_matrix(self, facts1: list[str], facts2: list[str]) -> np.ndarray:
        """Compute cosine similarity matrix between two fact sets.

        Args:
            facts1: First list of facts
            facts2: Second list of facts

        Returns:
            Matrix of shape (len(facts1), len(facts2)) with cosine similarities
        """
        if not facts1 or not facts2:
            return np.array([])

        # Encode facts to embeddings
        embeddings1 = self.encoder.encode(facts1)
        embeddings2 = self.encoder.encode(facts2)

        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)

        normalized1 = embeddings1 / norm1
        normalized2 = embeddings2 / norm2

        # Compute cosine similarity matrix
        similarity_matrix = np.dot(normalized1, normalized2.T)
        return similarity_matrix

    def _greedy_matching(
        self, similarity_matrix: np.ndarray
    ) -> tuple[list[dict[str, Any]], set[int]]:
        """Perform greedy matching based on similarity matrix.

        Args:
            similarity_matrix: Matrix of similarities

        Returns:
            Tuple of (matches, used_gold_indices)
        """
        matches = []
        used_model = set()
        used_gold = set()

        # Flatten and sort by similarity (highest first)
        pairs = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    pairs.append((i, j, similarity_matrix[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)

        # Greedily match
        for model_idx, gold_idx, similarity in pairs:
            if model_idx not in used_model and gold_idx not in used_gold:
                matches.append(
                    {
                        "model_idx": model_idx,
                        "gold_idx": gold_idx,
                        "similarity": float(similarity),
                    }
                )
                used_model.add(model_idx)
                used_gold.add(gold_idx)

        return matches, used_gold

    def _optimal_matching(
        self, similarity_matrix: np.ndarray
    ) -> tuple[list[dict[str, Any]], set[int]]:
        """Perform optimal bipartite matching using Hungarian algorithm.

        Args:
            similarity_matrix: Matrix of similarities

        Returns:
            Tuple of (matches, used_gold_indices)
        """
        from scipy.optimize import linear_sum_assignment

        # Convert similarity to cost (higher similarity = lower cost)
        cost_matrix = 1 - similarity_matrix

        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        used_gold = set()

        for model_idx, gold_idx in zip(row_ind, col_ind, strict=False):
            similarity = similarity_matrix[model_idx, gold_idx]
            if similarity >= self.similarity_threshold:
                matches.append(
                    {
                        "model_idx": model_idx,
                        "gold_idx": gold_idx,
                        "similarity": float(similarity),
                    }
                )
                used_gold.add(gold_idx)

        return matches, used_gold

    def _match_facts(self, model_facts: list[str], gold_facts: list[str]) -> dict[str, Any]:
        """Match facts between model and gold answers.

        Args:
            model_facts: Facts from model answer
            gold_facts: Facts from gold answer

        Returns:
            Dictionary with matching results and metrics
        """
        if not model_facts or not gold_facts:
            return {
                "matches": [],
                "unmatched_model": model_facts,
                "unmatched_gold": gold_facts,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "similarity_matrix": [],
            }

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(model_facts, gold_facts)

        # Find matches
        if self.matching_strategy == "greedy":
            matches, used_gold_indices = self._greedy_matching(similarity_matrix)
        else:
            matches, used_gold_indices = self._optimal_matching(similarity_matrix)

        # Identify unmatched facts
        used_model_indices = {m["model_idx"] for m in matches}
        unmatched_model = [
            model_facts[i] for i in range(len(model_facts)) if i not in used_model_indices
        ]
        unmatched_gold = [
            gold_facts[j] for j in range(len(gold_facts)) if j not in used_gold_indices
        ]

        # Compute metrics
        num_matches = len(matches)
        precision = num_matches / len(model_facts) if model_facts else 0.0
        recall = num_matches / len(gold_facts) if gold_facts else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "matches": matches,
            "unmatched_model": unmatched_model,
            "unmatched_gold": unmatched_gold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "similarity_matrix": similarity_matrix.tolist(),
        }

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute fact matching scores.

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
        f1_scores = []

        for pair in qa_pairs:
            try:
                # Extract facts from both answers
                model_facts = self._extract_facts(pair["question"], pair["answer"], "model")
                gold_facts = self._extract_facts(pair["question"], pair["gold_answer"], "gold")

                # Match facts
                matching_result = self._match_facts(model_facts, gold_facts)

                # Store detailed results
                individual_scores.append(
                    {
                        "question": pair["question"],
                        "model_facts": model_facts,
                        "gold_facts": gold_facts,
                        "matching": matching_result,
                    }
                )

                f1_scores.append(matching_result["f1"])

            except Exception as e:
                print(f"Error processing pair: {e}")
                individual_scores.append(
                    {
                        "question": pair.get("question", ""),
                        "model_facts": [],
                        "gold_facts": [],
                        "matching": {
                            "matches": [],
                            "unmatched_model": [],
                            "unmatched_gold": [],
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "error": str(e),
                        },
                    }
                )
                f1_scores.append(0.0)

        # Overall F1 score (average across all pairs)
        overall_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return [
            {
                "metric_name": self.name,
                "score": overall_f1,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "llm_model": self.llm_model,
                    "embedding_model": self.config.get("embedding_model"),
                    "similarity_threshold": self.similarity_threshold,
                    "matching_strategy": self.matching_strategy,
                    "config": self.config,
                },
            }
        ]
