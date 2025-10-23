"""Tests for AnswerLength metric."""

import pytest

from retrieval_evals.evals.ungrounded.metrics import AnswerLength
from tests.utils import create_temp_json


def test_answer_length_words() -> None:
    """Test answer length in words."""
    metric = AnswerLength(unit="words")
    qa_pairs = [
        {"question": "Q1", "answer": "one two three"},
        {"question": "Q2", "answer": "a b c d"},
    ]

    results = metric.compute(qa_pairs)
    assert len(results) == 1
    assert results[0]["metric_name"] == "answer_length_words"
    assert results[0]["score"] == 3.5
    assert results[0]["individual_scores"] == [3.0, 4.0]


def test_answer_length_characters() -> None:
    """Test answer length in characters."""
    metric = AnswerLength(unit="characters")
    qa_pairs = [
        {"question": "Q1", "answer": "abc"},
        {"question": "Q2", "answer": "12345"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["metric_name"] == "answer_length_characters"
    assert results[0]["score"] == 4.0
    assert results[0]["individual_scores"] == [3.0, 5.0]


def test_answer_length_invalid_unit() -> None:
    """Test that invalid unit raises error."""
    with pytest.raises(ValueError, match="Unit must be 'words' or 'characters'"):
        AnswerLength(unit="invalid")


def test_answer_length_empty_answer() -> None:
    """Test with empty answer."""
    metric = AnswerLength(unit="words")
    qa_pairs = [
        {"question": "Q1", "answer": ""},
        {"question": "Q2", "answer": "word"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 0.5
    assert results[0]["individual_scores"] == [0.0, 1.0]


def test_answer_length_empty_list() -> None:
    """Test with empty Q&A pairs list."""
    metric = AnswerLength(unit="words")
    results = metric.compute([])

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"] == []
    assert results[0]["metadata"]["num_pairs"] == 0


def test_answer_length_from_file() -> None:
    """Test loading data from file."""
    metric = AnswerLength(unit="words")
    test_data = [
        {"question": "Q1", "answer": "one two three"},
    ]
    temp_path = create_temp_json(test_data)

    try:
        results = metric.evaluate(temp_path)
        assert results[0]["score"] == 3.0
    finally:
        temp_path.unlink()


def test_answer_length_metadata() -> None:
    """Test metadata in results."""
    metric = AnswerLength(unit="words")
    qa_pairs = [
        {"question": "Q1", "answer": "a b"},
        {"question": "Q2", "answer": "one two three four"},
    ]

    results = metric.compute(qa_pairs)
    metadata = results[0]["metadata"]

    assert metadata["num_pairs"] == 2
    assert metadata["min"] == 2.0
    assert metadata["max"] == 4.0
    assert metadata["unit"] == "words"
    assert metadata["config"]["unit"] == "words"


def test_answer_length_single_word() -> None:
    """Test with single word answers."""
    metric = AnswerLength(unit="words")
    qa_pairs = [
        {"question": "Q1", "answer": "yes"},
        {"question": "Q2", "answer": "no"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 1.0
    assert results[0]["individual_scores"] == [1.0, 1.0]


def test_answer_length_whitespace_handling() -> None:
    """Test handling of extra whitespace."""
    metric = AnswerLength(unit="words")
    qa_pairs = [
        {"question": "Q1", "answer": "  one  two  "},
        {"question": "Q2", "answer": "a\n\nb\tc"},
    ]

    results = metric.compute(qa_pairs)
    # split() handles all whitespace: ["one", "two"] = 2 words, ["a", "b", "c"] = 3 words
    assert results[0]["score"] == 2.5
    assert results[0]["individual_scores"] == [2.0, 3.0]
