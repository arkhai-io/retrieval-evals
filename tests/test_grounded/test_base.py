"""Tests for grounded evaluation base orchestrator."""

from retrieval_evals.evals import evaluate_grounded
from retrieval_evals.evals.grounded.metrics import ExactMatch
from tests.utils import create_temp_json, sample_grounded_data


def test_evaluate_grounded_with_file() -> None:
    """Test grounded evaluation loading data from file."""
    test_data = sample_grounded_data()
    temp_path = create_temp_json(test_data)

    try:
        results = evaluate_grounded(temp_path)
        assert len(results) > 0
        assert "metric_name" in results[0]
        assert "score" in results[0]
        assert "individual_scores" in results[0]
        assert "metadata" in results[0]
        assert results[0]["metric_name"] == "exact_match"
    finally:
        temp_path.unlink()


def test_evaluate_grounded_with_data() -> None:
    """Test grounded evaluation with data passed directly."""
    test_data = [
        {
            "question": "What is 2+2?",
            "answer": "4",
            "gold_answer": "4",
        },
        {
            "question": "What is 3+3?",
            "answer": "6",
            "gold_answer": "six",
        },
    ]

    results = evaluate_grounded(test_data)
    assert len(results) == 1
    assert results[0]["metric_name"] == "exact_match"
    assert results[0]["score"] == 0.5
    assert results[0]["individual_scores"] == [1.0, 0.0]
    assert len(results[0]["individual_scores"]) == 2


def test_evaluate_grounded_custom_metrics() -> None:
    """Test grounded evaluation with custom metric instances."""
    test_data = [
        {
            "question": "What is 2+2?",
            "answer": "Four",
            "gold_answer": "four",
        }
    ]

    # Case sensitive - should not match
    results = evaluate_grounded(test_data, metrics=[ExactMatch(case_sensitive=True)])
    assert results[0]["score"] == 0.0

    # Case insensitive - should match
    results = evaluate_grounded(test_data, metrics=[ExactMatch(case_sensitive=False)])
    assert results[0]["score"] == 1.0


def test_evaluate_grounded_multiple_metrics() -> None:
    """Test grounded evaluation with multiple metrics."""
    test_data = sample_grounded_data()

    results = evaluate_grounded(
        test_data,
        metrics=[
            ExactMatch(case_sensitive=False),
            ExactMatch(case_sensitive=True),
        ],
    )

    assert len(results) == 2
    assert all(r["metric_name"] == "exact_match" for r in results)
    # Different configs should yield potentially different scores
    assert "case_sensitive" in results[0]["metadata"]["config"]
