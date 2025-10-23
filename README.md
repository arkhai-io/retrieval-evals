# retrieval-evals

Evaluation framework for question-answering systems.

## Installation

```bash
poetry install
```

## Usage

### Grounded Evaluations

Compare generated answers against gold standard answers.

```python
from retrieval_evals.evals import evaluate_grounded

# From file
results = evaluate_grounded("data.json")

# From data
qa_pairs = [
    {
        "question": "What is Python?",
        "answer": "A programming language",
        "gold_answer": "A high-level programming language"
    }
]
results = evaluate_grounded(qa_pairs)

# Custom metric configuration
from retrieval_evals.evals.grounded.metrics import ExactMatch

results = evaluate_grounded(
    qa_pairs,
    metrics=[ExactMatch(case_sensitive=True)]
)
```

### Ungrounded Evaluations

Evaluate without gold standard answers.

```python
from retrieval_evals.evals import evaluate_ungrounded
from retrieval_evals.evals.ungrounded.metrics import AnswerLength

qa_pairs = [
    {
        "question": "What is Python?",
        "answer": "A programming language"
    }
]

results = evaluate_ungrounded(
    qa_pairs,
    metrics=[AnswerLength(unit="words")]
)
```

## Output Schema

```python
[
    {
        "metric_name": str,              # Metric identifier
        "score": float,                  # Overall average score
        "individual_scores": [float],    # Score per Q&A pair
        "metadata": {                    # Additional context
            "num_pairs": int,
            "config": dict,
            ...
        }
    }
]
```

## Available Metrics

**Grounded:**
- `ExactMatch(case_sensitive=False, normalize_whitespace=True)`

**Ungrounded:**
- `AnswerLength(unit="words")`

## Adding Metrics

Create a new metric class in the appropriate directory:

```python
# retrieval_evals/evals/grounded/metrics/my_metric.py
from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold

class MyMetric(Metric):
    def __init__(self, param: str = "default"):
        super().__init__(param=param)
        self.param = param

    @property
    def name(self) -> str:
        return "my_metric"

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        individual_scores = [1.0 for _ in qa_pairs]  # Your logic here

        return [{
            "metric_name": self.name,
            "score": sum(individual_scores) / len(individual_scores),
            "individual_scores": individual_scores,
            "metadata": {"num_pairs": len(qa_pairs), "config": self.config}
        }]
```

Register in `__init__.py` and add to `DEFAULT_METRICS` in `base.py`.

## Development

**Install pre-commit hooks:**
```bash
poetry run pre-commit install
```

**Run tests:**
```bash
poetry run pytest
```

**Lint and format:**
```bash
poetry run ruff format .
poetry run ruff check --fix .
poetry run mypy retrieval_evals
```

## Structure

```
retrieval_evals/
├── types/
│   └── __init__.py          # QAPair, QAPairWithGold, EvalResult
├── evals/
│   ├── grounded/
│   │   ├── base.py          # evaluate_grounded()
│   │   └── metrics/
│   │       ├── base.py      # Metric base class
│   │       └── exact_match.py
│   └── ungrounded/
│       ├── base.py          # evaluate_ungrounded()
│       └── metrics/
│           ├── base.py      # Metric base class
│           └── answer_length.py
└── ...
```
