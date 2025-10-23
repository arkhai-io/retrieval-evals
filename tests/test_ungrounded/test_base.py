"""Tests for ungrounded evaluation base orchestrator."""

from retrieval_evals.evals import evaluate_ungrounded
from retrieval_evals.evals.ungrounded.metrics import AnswerLength
from tests.utils import create_temp_json, sample_ungrounded_data


def test_evaluate_ungrounded_with_file() -> None:
    """Test ungrounded evaluation loading data from file."""
    test_data = sample_ungrounded_data()
    temp_path = create_temp_json(test_data)

    try:
        results = evaluate_ungrounded(temp_path)
        assert len(results) > 0
        assert "metric_name" in results[0]
        assert "score" in results[0]
        assert "individual_scores" in results[0]
        assert "metadata" in results[0]
        assert results[0]["metric_name"] == "answer_length_words"
    finally:
        temp_path.unlink()


def test_evaluate_ungrounded_with_data() -> None:
    """Test ungrounded evaluation with data passed directly."""
    test_data = [
        {
            "question": "What is 2+2?",
            "answer": "The answer is four",
        },
        {
            "question": "What is Python?",
            "answer": "A programming language",
        },
    ]

    results = evaluate_ungrounded(test_data)
    assert len(results) == 1
    assert results[0]["metric_name"] == "answer_length_words"
    assert results[0]["score"] == 3.5
    assert results[0]["individual_scores"] == [4.0, 3.0]
    assert len(results[0]["individual_scores"]) == 2


def test_evaluate_ungrounded_custom_metrics() -> None:
    """Test ungrounded evaluation with custom metric instances."""
    test_data = [
        {
            "question": "What is 2+2?",
            "answer": "Four",
        }
    ]

    results = evaluate_ungrounded(test_data, metrics=[AnswerLength(unit="characters")])
    assert results[0]["metric_name"] == "answer_length_characters"
    assert results[0]["score"] == 4.0


def test_evaluate_ungrounded_multiple_metrics() -> None:
    """Test ungrounded evaluation with multiple metrics."""
    test_data = sample_ungrounded_data()

    results = evaluate_ungrounded(
        test_data,
        metrics=[
            AnswerLength(unit="words"),
            AnswerLength(unit="characters"),
        ],
    )

    assert len(results) == 2
    metric_names = [r["metric_name"] for r in results]
    assert "answer_length_words" in metric_names
    assert "answer_length_characters" in metric_names
