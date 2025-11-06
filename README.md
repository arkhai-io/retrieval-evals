<div align="center">
  <img src="assets/logo.jpg" alt="Retrieval Evals Logo" width="200"/>

  # Retrieval Evals

  Evaluation framework for question-answering systems

  ![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
  ![License MIT](https://img.shields.io/badge/license-MIT-green.svg)
</div>

---

## Overview

Retrieval Evals is a comprehensive evaluation framework for assessing question-answering systems. It supports both grounded evaluations (with gold standard answers) and ungrounded evaluations (without reference answers), providing flexible metric configurations and extensible architecture.

## Key Features

- **Grounded Evaluations**: Compare generated answers against gold standard references using metrics like ExactMatch, SemanticSimilarity, BLEU, ROUGE, and METEOR
- **Ungrounded Evaluations**: Assess answers without references using AnswerLength, Coherence, Readability, and Communication Quality
- **Extensible Architecture**: Easy-to-implement custom metrics with standardized interfaces
- **LLM-based Metrics**: Advanced evaluation using language models for faithfulness, answer quality, and structure analysis
- **Comprehensive Output**: Detailed scoring with individual and aggregate results plus metadata

## Installation

```bash
poetry install
```

## Quick Start

### Grounded Evaluation

```python
from retrieval_evals.evals import evaluate_grounded

qa_pairs = [
    {
        "question": "What is Python?",
        "answer": "A programming language",
        "gold_answer": "A high-level programming language"
    }
]

results = evaluate_grounded(qa_pairs)
```

### Ungrounded Evaluation

```python
from retrieval_evals.evals import evaluate_ungrounded

qa_pairs = [
    {
        "question": "What is Python?",
        "answer": "A programming language"
    }
]

results = evaluate_ungrounded(qa_pairs)
```

### Custom Metric Configuration

```python
from retrieval_evals.evals.grounded.metrics import ExactMatch, SemanticSimilarity

results = evaluate_grounded(
    qa_pairs,
    metrics=[
        ExactMatch(case_sensitive=True),
        SemanticSimilarity(model_name="all-MiniLM-L6-v2")
    ]
)
```

## Available Metrics

### Grounded Metrics
- **ExactMatch**: Binary match between answer and gold standard
- **SemanticSimilarity**: Cosine similarity using sentence embeddings
- **BLEU**: Precision-based n-gram overlap
- **ROUGE**: Recall-based n-gram overlap
- **METEOR**: Alignment-based matching with synonyms
- **FactMatching**: LLM-based fact extraction and comparison
- **AnswerQuality**: LLM-based quality assessment
- **LongQAAnswer**: Specialized evaluation for long-form answers
- **MORQAFaithfulness**: Context-based faithfulness evaluation

### Ungrounded Metrics
- **AnswerLength**: Token or word count analysis
- **Coherence**: LLM-based coherence scoring
- **Readability**: Flesch Reading Ease and Grade Level
- **CommunicationQuality**: LLM-based communication assessment
- **AnswerStructure**: Format and organization evaluation

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

## Creating Custom Metrics

Implement the `Metric` base class in the appropriate directory:

```python
from retrieval_evals.evals.grounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPairWithGold

class CustomMetric(Metric):
    def __init__(self, param: str = "default"):
        super().__init__(param=param)
        self.param = param

    @property
    def name(self) -> str:
        return "custom_metric"

    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        individual_scores = [self._score(pair) for pair in qa_pairs]

        return [{
            "metric_name": self.name,
            "score": sum(individual_scores) / len(individual_scores),
            "individual_scores": individual_scores,
            "metadata": {"num_pairs": len(qa_pairs), "config": self.config}
        }]

    def _score(self, pair: QAPairWithGold) -> float:
        # Custom scoring logic
        return 1.0
```

Register the metric in `__init__.py` and add to `DEFAULT_METRICS` in `base.py`.

## Development

**Install dependencies:**
```bash
poetry install
```

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

## Project Structure

```
retrieval_evals/
├── types/                      # Type definitions
│   └── __init__.py
├── evals/
│   ├── grounded/
│   │   ├── base.py            # evaluate_grounded()
│   │   ├── metrics/           # Grounded metrics
│   │   └── prompts/           # LLM prompts
│   └── ungrounded/
│       ├── base.py            # evaluate_ungrounded()
│       ├── metrics/           # Ungrounded metrics
│       └── prompts/           # LLM prompts
└── py.typed
```

## License

MIT
