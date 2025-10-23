"""Tests for evaluation modules."""

import json
import tempfile
from pathlib import Path

from retrieval_evals.evals import evaluate_grounded, evaluate_ungrounded


def test_evaluate_grounded() -> None:
    """Test grounded evaluation with gold standard."""
    # Create temporary test data
    test_data = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "gold_answer": "Four",
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        results = evaluate_grounded(temp_path)
        assert len(results) > 0
        assert "metric_name" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]
    finally:
        temp_path.unlink()


def test_evaluate_ungrounded() -> None:
    """Test ungrounded evaluation without gold standard."""
    # Create temporary test data
    test_data = [
        {
            "question": "What is 2+2?",
            "answer": "4",
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        results = evaluate_ungrounded(temp_path)
        assert len(results) > 0
        assert "metric_name" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]
    finally:
        temp_path.unlink()
