"""Shared test utilities and fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Any


def create_temp_json(data: list[dict[str, Any]]) -> Path:
    """Create a temporary JSON file with test data.

    Args:
        data: List of dictionaries to write to JSON

    Returns:
        Path to temporary JSON file (caller should delete after use)
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return Path(f.name)


def sample_grounded_data() -> list[dict[str, str]]:
    """Generate sample grounded evaluation data.

    Returns:
        List of Q&A pairs with gold standard answers
    """
    return [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "gold_answer": "4",
        },
        {
            "question": "What is Python?",
            "answer": "programming language",
            "gold_answer": "Programming Language",
        },
    ]


def sample_ungrounded_data() -> list[dict[str, str]]:
    """Generate sample ungrounded evaluation data.

    Returns:
        List of Q&A pairs without gold standard answers
    """
    return [
        {
            "question": "What is 2+2?",
            "answer": "The answer is four",
        },
        {
            "question": "What is Python?",
            "answer": "A programming language",
        },
    ]
