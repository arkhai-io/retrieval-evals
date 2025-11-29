"""LLM-as-judge metric for evaluating style consistency.

Compares new answers against reference examples to ensure consistent style, tone,
format, and communication patterns across responses.

Prompt Behavior:
The LLM evaluates answers across five dimensions using a 5-point Likert scale:
1. Tone & Formality Consistency (1-5): Assesses whether formality level, warmth,
   pronoun usage, and engagement match the reference style.
2. Structure & Format Similarity (1-5): Evaluates answer organization, use of
   lists/paragraphs, level of detail, and opening/closing patterns.
3. Length & Verbosity Alignment (1-5): Checks answer length comparability,
   conciseness vs elaboration, and sentence length patterns.
4. Vocabulary & Complexity Level (1-5): Assesses vocabulary sophistication,
   technical vs layman terms, and use of examples/analogies.
5. Response Pattern Matching (1-5): Evaluates approach to answering, use of
   disclaimers, certainty level, and directness.

Scores: 5=Perfect match, 4=Very similar, 3=Somewhat similar, 2=Noticeably different,
1=Completely different.
"""

import json
from pathlib import Path

import requests

from retrieval_evals.evals.ungrounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPair


class StyleConsistency(Metric):
    """LLM-as-judge for evaluating style consistency against reference examples.

    Evaluates:
    1. Tone & formality consistency
    2. Structure & format similarity
    3. Length & verbosity alignment
    4. Vocabulary & complexity level
    5. Response pattern matching

    Uses OpenRouter API for LLM evaluation.
    """

    def __init__(
        self,
        api_key: str,
        reference_pairs: list[dict[str, str]],
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
        max_reference_examples: int = 5,
    ):
        """Initialize style consistency evaluation metric.

        Args:
            api_key: OpenRouter API key
            reference_pairs: List of reference Q&A pairs showing desired style
            model: Model identifier on OpenRouter
            base_url: OpenRouter API base URL
            max_reference_examples: Maximum number of reference examples to include
        """
        super().__init__(
            reference_pairs=reference_pairs,
            model=model,
            max_reference_examples=max_reference_examples,
        )
        self.api_key = api_key
        self.reference_pairs = reference_pairs[:max_reference_examples]
        self.model = model
        self.base_url = base_url
        self.max_reference_examples = max_reference_examples

        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "style_consistency.txt"
        self.prompt_template = prompt_path.read_text()

    @property
    def name(self) -> str:
        """Return metric name."""
        return "style_consistency"

    def _format_reference_examples(self) -> str:
        """Format reference examples for the prompt.

        Returns:
            Formatted string with all reference examples
        """
        formatted_examples = []
        for idx, pair in enumerate(self.reference_pairs, start=1):
            example = f"""EXAMPLE {idx}:
QUESTION: {pair['question']}
ANSWER: {pair['answer']}"""
            formatted_examples.append(example)

        return "\n\n".join(formatted_examples)

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
        """Compute LLM-as-judge scores for style consistency.

        Args:
            qa_pairs: List of Q&A pairs to evaluate

        Returns:
            List with single evaluation result
        """
        if not qa_pairs:
            return [
                {
                    "metric_name": self.name,
                    "score": 0.0,
                    "individual_scores": [],
                    "metadata": {
                        "num_pairs": 0,
                        "num_references": len(self.reference_pairs),
                        "model": self.model,
                        "config": self.config,
                    },
                }
            ]

        reference_examples = self._format_reference_examples()
        individual_scores = []
        avg_scores = []

        for pair in qa_pairs:
            prompt = self.prompt_template.format(
                reference_examples=reference_examples,
                question=pair["question"],
                answer=pair["answer"],
            )

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
                    "num_references": len(self.reference_pairs),
                    "model": self.model,
                    "config": self.config,
                },
            }
        ]
