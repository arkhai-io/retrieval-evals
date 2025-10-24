"""LLM-as-judge metric for factual faithfulness evaluation.

Inspired by MORQA: Benchmarking Evaluation Metrics for Medical Open-Ended
Question Answering (Yim et al., 2025).

Prompt Behavior:
The LLM extracts up to 10 atomic facts from the model answer and verifies each
against the reference answer. Each fact is labeled as:
- supported_by_reference: Fact is confirmed by reference
- partially_supported: Fact is somewhat aligned but incomplete/imprecise
- contradicted: Fact directly conflicts with reference
- not_in_reference: Fact is not mentioned in reference

Computes atomic_faithfulness score:
(#supported + 0.5*#partially_supported) / max(1, #facts)

Also identifies critical errors (wrong diagnosis, wrong drug/dose, missed red-flags).
Returns JSON with extracted facts, critical errors, faithfulness score, and summary.
"""

import json
from pathlib import Path
from typing import Any

import requests

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class MORQAFaithfulness(Metric):
    """LLM-as-judge for evaluating factual faithfulness against reference.

    Extracts atomic facts and verifies alignment with gold standard answer.
    Uses OpenRouter API for LLM evaluation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize MORQA faithfulness evaluation metric.

        Args:
            api_key: OpenRouter API key
            model: Model identifier on OpenRouter
            base_url: OpenRouter API base URL
        """
        super().__init__(api_key=api_key, model=model, base_url=base_url)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url

        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "morqa_faithfulness.txt"
        self.prompt_template = prompt_path.read_text()

    @property
    def name(self) -> str:
        """Return metric name."""
        return "morqa_faithfulness"

    def _call_llm(self, prompt: str) -> dict[str, Any]:
        """Call LLM via OpenRouter API.

        Args:
            prompt: Formatted prompt string

        Returns:
            Dictionary with facts, critical_errors, atomic_faithfulness, summary
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

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute factual faithfulness scores.

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
        avg_scores = []

        for pair in qa_pairs:
            prompt = self.prompt_template.format(
                question=pair["question"],
                answer=pair["answer"],
                gold_answer=pair["gold_answer"],
            )

            try:
                result = self._call_llm(prompt)
                individual_scores.append(result)
                avg_scores.append(result["atomic_faithfulness"])
            except Exception as e:
                # On error, append empty result and 0
                print(f"Error evaluating pair: {e}")
                individual_scores.append(
                    {
                        "facts": [],
                        "critical_errors": [],
                        "atomic_faithfulness": 0.0,
                        "summary": "Evaluation error",
                    }
                )
                avg_scores.append(0.0)

        overall_avg = sum(avg_scores) / len(avg_scores)

        return [
            {
                "metric_name": self.name,
                "score": overall_avg,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "model": self.model,
                    "config": self.config,
                },
            }
        ]
