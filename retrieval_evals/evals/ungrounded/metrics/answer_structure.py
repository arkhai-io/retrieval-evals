"""LLM-as-judge metric for answer structure and organization evaluation.

Evaluates how well an answer is structured and organized, independent of
factual correctness. Assesses:

1. **Organization** (1-5): Clear introduction, body, conclusion; logical flow
2. **Formatting** (1-5): Appropriate use of paragraphs, lists, headers, emphasis
3. **Information Hierarchy** (1-5): Most important info first, good progression
4. **Clarity of Expression** (1-5): Easy to follow, good transitions, no confusion

Scores: 5=Excellent, 4=Good, 3=Acceptable, 2=Poor, 1=Unacceptable

The overall score is the average across all four dimensions, scaled to 0-1.
"""

import json
from pathlib import Path
from typing import Any

import requests

from retrieval_evals.evals.ungrounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPair


class AnswerStructure(Metric):
    """LLM-as-judge for evaluating answer structure and organization.

    Evaluates:
    1. Organization (logical flow, intro/body/conclusion)
    2. Formatting (paragraphs, lists, emphasis)
    3. Information hierarchy (importance ordering)
    4. Clarity of expression (transitions, comprehensibility)

    Uses OpenRouter API for LLM evaluation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize answer structure evaluation metric.

        Args:
            api_key: OpenRouter API key
            model: Model identifier on OpenRouter
            base_url: OpenRouter API base URL
        """
        super().__init__(model=model)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "answer_structure.txt"
        self.prompt_template = prompt_path.read_text()

    @property
    def name(self) -> str:
        """Return metric name."""
        return "answer_structure"

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM via OpenRouter API.

        Args:
            prompt: Formatted prompt string

        Returns:
            Dictionary with scores and explanations
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

    def compute(self, qa_pairs: list[QAPair]) -> list[EvalResult]:
        """Compute answer structure scores.

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
        avg_scores = []

        for pair in qa_pairs:
            prompt = self.prompt_template.format(
                question=pair["question"],
                answer=pair["answer"],
            )

            try:
                result = self._call_llm(prompt)

                # Compute average score (1-5 scale, convert to 0-1)
                dimensions = ["organization", "formatting", "hierarchy", "clarity"]
                dimension_scores = [result.get(dim, 3) for dim in dimensions]
                avg_score = sum(dimension_scores) / len(dimensions)
                normalized_score = (avg_score - 1) / 4  # Convert 1-5 to 0-1

                individual_scores.append(result)
                avg_scores.append(normalized_score)

            except Exception as e:
                print(f"Error evaluating pair: {e}")
                individual_scores.append(
                    {
                        "organization": 0,
                        "formatting": 0,
                        "hierarchy": 0,
                        "clarity": 0,
                        "summary": "Evaluation error",
                    }
                )
                avg_scores.append(0.0)

        overall_avg = sum(avg_scores) / len(avg_scores) if avg_scores else 0.0

        return [
            {
                "metric_name": self.name,
                "score": overall_avg,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "model": self.model,
                    "config": self.config,
                    "dimensions": [
                        "organization",
                        "formatting",
                        "hierarchy",
                        "clarity",
                    ],
                },
            }
        ]
