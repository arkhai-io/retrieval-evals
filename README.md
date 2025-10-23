# retrieval-evals

Retrieval evaluation framework

## Installation

Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

## Development

### Running Tests

```bash
pytest
```

### Linting

Format code with ruff:
```bash
ruff format .
```

Lint code:
```bash
ruff check .
```

Auto-fix linting issues:
```bash
ruff check --fix .
```

### Type Checking

```bash
mypy retrieval_evals
```

### Run All Checks

```bash
# Format
ruff format .

# Lint
ruff check --fix .

# Type check
mypy retrieval_evals

# Test
pytest
```

## Project Structure

```
retrieval-evals/
├── retrieval_evals/     # Main package
│   ├── __init__.py
│   └── py.typed         # PEP 561 marker for type checking
├── tests/               # Test files
│   ├── __init__.py
│   └── test_example.py
├── pyproject.toml       # Project configuration
├── README.md
└── .gitignore
```
