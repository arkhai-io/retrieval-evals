"""LLM-as-judge metric for communication quality evaluation.

Evaluates answer quality across tone, professionalism, and bias dimensions.
This metric assesses how well the answer communicates information regardless of
factual accuracy (which is handled by grounded metrics).

Prompt Behavior:
The LLM evaluates answers across three dimensions using a 5-point Likert scale:
1. Tone Appropriateness (1-5): Assesses whether the tone is suitable for the
   context, professional without being overly formal or casual, and respectful.
2. Professionalism (1-5): Evaluates clarity, structure, grammar, appropriate
   disclaimers/caveats, and avoidance of speculation presented as fact.
3. Bias & Fairness (1-5): Checks for demographic bias (gender, race, age,
   religion), balanced perspective, and inclusive language.

Scores: 5=Excellent, 4=Good, 3=Acceptable, 2=Poor, 1=Unacceptable.
"""

import json
from pathlib import Path

import requests

from retrieval_evals.evals.ungrounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPair


class CommunicationQuality(Metric):
    """LLM-as-judge for evaluating communication quality.

    Evaluates:
    1. Tone appropriateness
    2. Professionalism
    3. Bias & fairness

    Uses OpenRouter API for LLM evaluation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize communication quality evaluation metric.

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
        prompt_path = Path(__file__).parent.parent / "prompts" / "communication_quality.txt"
        self.prompt_template = prompt_path.read_text()

    @property
    def name(self) -> str:
        """Return metric name."""
        return "communication_quality"

    def _call_llm(self, prompt: str) -> dict[str, int]:
        """Call LLM via OpenRouter API.

        Args:
            prompt: Formatted prompt string

        Returns:
            Dictionary with dimension scores
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
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed: dict[str, int] = json.loads(content)
        return parsed

    def compute(self, qa_pairs: list[QAPair]) -> list[EvalResult]:
        """Compute LLM-as-judge scores for communication quality.

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
            prompt = self.prompt_template.format(question=pair["question"], answer=pair["answer"])

            try:
                scores = self._call_llm(prompt)
                individual_scores.append(scores)

                # Average across dimensions for overall score
                avg_score = sum(scores.values()) / len(scores)
                avg_scores.append(avg_score)
            except Exception as e:
                # On error, append empty dict and 0
                print(f"Error evaluating pair: {e}")
                individual_scores.append({})
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
