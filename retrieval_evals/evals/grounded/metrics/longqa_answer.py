"""LLM-as-judge metric for answer quality evaluation.

Inspired by LONGQAEval: Designing Reliable Evaluations of Long-Form Clinical QA
under Resource Constraints (Bologna et al., 2025).
https://arxiv.org/abs/2501.00000

Prompt Behavior:
The LLM evaluates answers across three dimensions using a 5-point Likert scale:
1. Medical Knowledge Alignment (1-5): Assesses factual accuracy, evidence-based
   claims, appropriate uncertainty expression, and absence of contradictions.
2. Question Addressing (1-5): Evaluates relevance, completeness of response,
   inclusion of requested details, and lack of digressions.
3. Risk Communication (1-5): Checks for clear explanation of contraindications,
   side effects, and potential consequences in accessible language.

The prompt provides scoring guidelines and instructs the LLM to return only JSON
with numeric scores (no explanation), making it suitable for automated evaluation.
Scores: 5=Agree, 4=Partially Agree, 3=Neutral, 2=Partially Disagree, 1=Disagree.
"""

import json
from pathlib import Path

import requests

from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold


class LongQAAnswer(Metric):
    """LLM-as-judge for evaluating answer quality across three dimensions.

    Evaluates:
    1. Alignment with current medical knowledge
    2. Addressing the specific question
    3. Communicating contraindications or risks

    Uses OpenRouter API for LLM evaluation.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """Initialize LongQA answer evaluation metric.

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
        prompt_path = Path(__file__).parent.parent / "prompts" / "longqa_answer.txt"
        self.prompt_template = prompt_path.read_text()

    @property
    def name(self) -> str:
        """Return metric name."""
        return "longqa_answer"

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

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute LLM-as-judge scores.

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
