# retrieval-evals

Retrieval evaluation framework for assessing question-answering systems.

## Installation

Install the package with Poetry:

```bash
poetry install
```

## Usage

The framework supports two types of evaluations:

### 1. Grounded Evaluations (with gold standard)

Evaluations that compare generated answers against gold standard answers.

```python
from retrieval_evals.evals import evaluate_grounded

results = evaluate_grounded("examples/grounded.json")
```

Example JSON format:
```json
[
  {
    "question": "What is Python?",
    "answer": "Python is a programming language.",
    "gold_answer": "Python is a high-level programming language."
  }
]
```

### 2. Ungrounded Evaluations (without gold standard)

Evaluations that don't require gold standard answers, using metrics like coherence, relevance, and consistency.

```python
from retrieval_evals.evals import evaluate_ungrounded

results = evaluate_ungrounded("examples/ungrounded.json")
```

Example JSON format:
```json
[
  {
    "question": "What is Python?",
    "answer": "Python is a programming language."
  }
]
```

## Development

### Pre-commit Hooks

Install pre-commit hooks (runs linting, formatting, and type checking automatically):
```bash
poetry run pre-commit install
```

Run hooks manually:
```bash
poetry run pre-commit run --all-files
```

### Running Tests

```bash
poetry run pytest
```

### Linting

Format code with ruff:
```bash
poetry run ruff format .
```

Lint code:
```bash
poetry run ruff check .
```

Auto-fix linting issues:
```bash
poetry run ruff check --fix .
```

### Type Checking

```bash
poetry run mypy retrieval_evals
```

## Project Structure

```
retrieval-evals/
├── retrieval_evals/              # Main package
│   ├── __init__.py
│   ├── types/                    # Type definitions
│   │   └── __init__.py
│   ├── py.typed                  # PEP 561 marker
│   └── evals/                    # Evaluation modules
│       ├── grounded/             # With gold standard
│       │   ├── __init__.py
│       │   └── base.py
│       └── ungrounded/           # Without gold standard
│           ├── __init__.py
│           └── base.py
├── tests/                        # Test files
├── examples/                     # Example JSON files
│   ├── grounded.json
│   └── ungrounded.json
├── pyproject.toml                # Project configuration
├── .pre-commit-config.yaml       # Pre-commit hooks
└── README.md
```

## Adding New Evaluations

To add a new evaluation method:

1. For grounded evaluations, add a new module in `retrieval_evals/evals/grounded/`
2. For ungrounded evaluations, add a new module in `retrieval_evals/evals/ungrounded/`
3. Implement your evaluation logic following the existing patterns
4. Add tests in the `tests/` directory
