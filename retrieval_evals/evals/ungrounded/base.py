"""Base module for ungrounded evaluations.

This module handles evaluations where gold standard answers are not available.
It uses alternative metrics like consistency, coherence, or model-based scoring.
"""

import json
from pathlib import Path

from retrieval_evals.types import EvalResult, QAPair


def load_data(json_path: Path | str) -> list[QAPair]:
    """Load Q&A pairs from JSON file.

    Expected JSON format:
    [
        {
            "question": "What is Python?",
            "answer": "Python is a programming language."
        },
        ...
    ]

    Args:
        json_path: Path to JSON file containing Q&A pairs

    Returns:
        List of Q&A pairs

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If file is not a JSON file or contains invalid JSON
    """
    # Convert to Path object for consistent handling
    path = Path(json_path)

    # Check if file exists
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    # Check if it's a file (not a directory)
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    # Check file extension
    if path.suffix.lower() != ".json":
        raise ValueError(f"File must be a JSON file, got: {path.suffix}")

    # Load and validate JSON syntax
    try:
        with open(path) as f:
            data: list[QAPair] = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {path}: {e}") from e

    return data


def evaluate_ungrounded(json_path: Path | str) -> list[EvalResult]:
    """Run ungrounded evaluation without gold standard answers.

    Args:
        json_path: Path to JSON file containing Q&A pairs

    Returns:
        List of evaluation results with scores
    """
    qa_pairs = load_data(json_path)

    # TODO: Implement actual evaluation logic
    # For now, just return placeholder results
    results: list[EvalResult] = [
        {
            "metric_name": "placeholder",
            "score": 0.0,
            "metadata": {"num_pairs": len(qa_pairs)},
        }
    ]

    return results
