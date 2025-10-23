"""Tests for ExactMatch metric."""

from retrieval_evals.evals.grounded.metrics import ExactMatch
from tests.utils import create_temp_json


def test_exact_match_basic() -> None:
    """Test basic exact match functionality."""
    metric = ExactMatch()
    qa_pairs = [
        {"question": "Q1", "answer": "yes", "gold_answer": "yes"},
        {"question": "Q2", "answer": "no", "gold_answer": "yes"},
    ]

    results = metric.compute(qa_pairs)
    assert len(results) == 1
    assert results[0]["metric_name"] == "exact_match"
    assert results[0]["score"] == 0.5
    assert results[0]["individual_scores"] == [1.0, 0.0]


def test_exact_match_case_insensitive() -> None:
    """Test case insensitive matching."""
    metric = ExactMatch(case_sensitive=False)
    qa_pairs = [
        {"question": "Q1", "answer": "Yes", "gold_answer": "yes"},
        {"question": "Q2", "answer": "NO", "gold_answer": "no"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 1.0
    assert results[0]["individual_scores"] == [1.0, 1.0]


def test_exact_match_case_sensitive() -> None:
    """Test case sensitive matching."""
    metric = ExactMatch(case_sensitive=True)
    qa_pairs = [
        {"question": "Q1", "answer": "Yes", "gold_answer": "yes"},
        {"question": "Q2", "answer": "no", "gold_answer": "no"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 0.5
    assert results[0]["individual_scores"] == [0.0, 1.0]


def test_exact_match_whitespace_normalization() -> None:
    """Test whitespace normalization."""
    metric = ExactMatch(normalize_whitespace=True)
    qa_pairs = [
        {"question": "Q1", "answer": "  hello  world  ", "gold_answer": "hello world"},
        {"question": "Q2", "answer": "foo\n\nbar", "gold_answer": "foo bar"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 1.0
    assert results[0]["individual_scores"] == [1.0, 1.0]


def test_exact_match_no_whitespace_normalization() -> None:
    """Test without whitespace normalization."""
    metric = ExactMatch(normalize_whitespace=False)
    qa_pairs = [
        {"question": "Q1", "answer": "hello  world", "gold_answer": "hello world"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 0.0  # Extra spaces not normalized


def test_exact_match_empty_list() -> None:
    """Test with empty Q&A pairs list."""
    metric = ExactMatch()
    results = metric.compute([])

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"] == []
    assert results[0]["metadata"]["num_pairs"] == 0


def test_exact_match_from_file() -> None:
    """Test loading data from file."""
    metric = ExactMatch()
    test_data = [
        {"question": "Q1", "answer": "yes", "gold_answer": "yes"},
    ]
    temp_path = create_temp_json(test_data)

    try:
        results = metric.evaluate(temp_path)
        assert results[0]["score"] == 1.0
    finally:
        temp_path.unlink()


def test_exact_match_metadata() -> None:
    """Test metadata in results."""
    metric = ExactMatch(case_sensitive=True, normalize_whitespace=False)
    qa_pairs = [
        {"question": "Q1", "answer": "yes", "gold_answer": "yes"},
        {"question": "Q2", "answer": "no", "gold_answer": "no"},
    ]

    results = metric.compute(qa_pairs)
    metadata = results[0]["metadata"]

    assert metadata["num_pairs"] == 2
    assert metadata["num_matches"] == 2
    assert metadata["config"]["case_sensitive"] is True
    assert metadata["config"]["normalize_whitespace"] is False
