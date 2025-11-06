"""LLM-as-a-judge metric for answer completeness and correctness.

This metric uses an LLM to evaluate two critical dimensions of answer quality:

1. **Completeness**: Does the answer cover all important aspects of the gold answer?
   - Evaluates information coverage, not just fact count
   - Considers whether key topics and concepts are addressed
   - Penalizes missing critical information
   - Range: 1-5 (1=Very Incomplete, 5=Fully Complete)

2. **Correctness**: Is the information in the answer factually accurate?
   - Identifies factual errors and contradictions
   - Checks accuracy against the gold standard
   - Detects hallucinations and misinformation
   - Range: 1-5 (1=Many Errors, 5=Fully Accurate)

Best Practices Implemented:
- Clear, structured prompts with examples
- Granular 1-5 Likert scale for nuance
- Explicit evaluation criteria and rubric
- Chain-of-thought reasoning required
- JSON output for structured parsing
- Temperature=0 for consistency
- Separate scoring for each dimension
- Detailed reasoning for interpretability

The metric returns both individual dimension scores and an overall quality score.
"""

import json
from pathlib import Path
from typing import Any

import requests

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class AnswerQuality(Metric):
    """LLM-based evaluation of answer completeness and correctness.

    This metric provides a comprehensive assessment of answer quality by evaluating:
    1. Completeness: Coverage of important information from gold answer
    2. Correctness: Factual accuracy compared to gold standard

    The metric uses a carefully designed prompt with explicit rubrics and requires
    the LLM to provide reasoning for its scores, improving reliability and interpretability.

    Example:
        >>> metric = AnswerQuality(api_key="your-openrouter-key")
        >>> results = metric.compute(qa_pairs)
        >>> print(f"Overall Quality: {results[0]['score']:.2f}/5")
        >>> print(f"Completeness: {results[0]['metadata']['avg_completeness']:.2f}/5")
        >>> print(f"Correctness: {results[0]['metadata']['avg_correctness']:.2f}/5")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize answer quality metric.

        Args:
            api_key: OpenRouter API key for LLM
            model: Model identifier on OpenRouter
            base_url: OpenRouter API base URL
        """
        super().__init__(model=model)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "answer_quality.txt"
        self.prompt_template = prompt_path.read_text()

    @property
    def name(self) -> str:
        """Return metric name."""
        return "answer_quality"

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM via OpenRouter API.

        Args:
            prompt: Formatted prompt string

        Returns:
            Dictionary with evaluation results
        """
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic for consistency
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

    def _evaluate_answer(
        self, question: str, model_answer: str, gold_answer: str
    ) -> dict[str, Any]:
        """Evaluate a single answer for completeness and correctness.

        Args:
            question: The question being answered
            model_answer: The model's generated answer
            gold_answer: The reference (gold) answer

        Returns:
            Dictionary with scores and reasoning
        """
        prompt = self.prompt_template.format(
            question=question,
            model_answer=model_answer,
            gold_answer=gold_answer,
        )

        try:
            result = self._call_llm(prompt)

            # Validate scores are in range
            completeness = max(1, min(5, result.get("completeness_score", 3)))
            correctness = max(1, min(5, result.get("correctness_score", 3)))

            # Overall quality is average of the two dimensions
            overall = (completeness + correctness) / 2.0

            return {
                "completeness_score": completeness,
                "correctness_score": correctness,
                "overall_score": overall,
                "completeness_reasoning": result.get("completeness_reasoning", ""),
                "correctness_reasoning": result.get("correctness_reasoning", ""),
                "key_missing_info": result.get("key_missing_info", []),
                "factual_errors": result.get("factual_errors", []),
            }
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {
                "completeness_score": 0,
                "correctness_score": 0,
                "overall_score": 0.0,
                "completeness_reasoning": f"Error: {str(e)}",
                "correctness_reasoning": f"Error: {str(e)}",
                "key_missing_info": [],
                "factual_errors": [],
            }

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute answer quality scores.

        Args:
            qa_pairs: List of Q&A pairs with gold standard answers

        Returns:
            List with single evaluation result containing overall quality score
        """
        if not qa_pairs:
            return [
                {
                    "metric_name": self.name,
                    "score": 0.0,
                    "individual_scores": [],
                    "metadata": {
                        "num_pairs": 0,
                        "avg_completeness": 0.0,
                        "avg_correctness": 0.0,
                        "config": self.config,
                    },
                }
            ]

        individual_scores = []
        completeness_scores = []
        correctness_scores = []
        overall_scores = []

        for pair in qa_pairs:
            try:
                result = self._evaluate_answer(
                    pair["question"],
                    pair["answer"],
                    pair["gold_answer"],
                )

                individual_scores.append(
                    {
                        "question": pair["question"],
                        **result,
                    }
                )

                completeness_scores.append(result["completeness_score"])
                correctness_scores.append(result["correctness_score"])
                overall_scores.append(result["overall_score"])

            except Exception as e:
                print(f"Error processing pair: {e}")
                individual_scores.append(
                    {
                        "question": pair.get("question", ""),
                        "completeness_score": 0,
                        "correctness_score": 0,
                        "overall_score": 0.0,
                        "error": str(e),
                    }
                )
                completeness_scores.append(0)
                correctness_scores.append(0)
                overall_scores.append(0.0)

        # Compute averages
        avg_completeness = (
            sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        )
        avg_correctness = (
            sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        )
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        return [
            {
                "metric_name": self.name,
                "score": avg_overall,  # Overall quality score (1-5)
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "avg_completeness": avg_completeness,
                    "avg_correctness": avg_correctness,
                    "model": self.model,
                    "config": self.config,
                },
            }
        ]
