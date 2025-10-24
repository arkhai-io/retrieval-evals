"""Tests for LongQAAnswer metric."""

import json
from unittest.mock import Mock, patch

import pytest

from retrieval_evals.evals.grounded.metrics import LongQAAnswer


@patch("retrieval_evals.evals.grounded.metrics.longqa_answer.requests.post")
def test_longqa_answer_basic(mock_post: Mock) -> None:
    """Test basic LongQA answer evaluation."""
    # Mock API response
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "The answer aligns with current medical knowledge": 5,
                            "The answer addresses the specific medical question": 4,
                            "The answer communicates contraindications or risks": 3,
                        }
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    metric = LongQAAnswer(api_key="test-key")
    qa_pairs = [
        {
            "question": "What is the treatment for headache?",
            "answer": "Ibuprofen or acetaminophen",
            "gold_answer": "NSAIDs or acetaminophen as first-line treatment",
        }
    ]

    results = metric.compute(qa_pairs)
    assert len(results) == 1
    assert results[0]["metric_name"] == "longqa_answer"
    assert results[0]["score"] == 4.0  # (5 + 4 + 3) / 3
    assert len(results[0]["individual_scores"]) == 1
    assert results[0]["individual_scores"][0] == {
        "The answer aligns with current medical knowledge": 5,
        "The answer addresses the specific medical question": 4,
        "The answer communicates contraindications or risks": 3,
    }


@patch("retrieval_evals.evals.grounded.metrics.longqa_answer.requests.post")
def test_longqa_answer_multiple_pairs(mock_post: Mock) -> None:
    """Test with multiple Q&A pairs."""
    # Mock API responses for two calls
    mock_responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "The answer aligns with current medical knowledge": 5,
                                "The answer addresses the specific medical question": 5,
                                "The answer communicates contraindications or risks": 5,
                            }
                        )
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "The answer aligns with current medical knowledge": 2,
                                "The answer addresses the specific medical question": 2,
                                "The answer communicates contraindications or risks": 1,
                            }
                        )
                    }
                }
            ]
        },
    ]

    mock_response = Mock()
    mock_response.json.side_effect = mock_responses
    mock_post.return_value = mock_response

    metric = LongQAAnswer(api_key="test-key")
    qa_pairs = [
        {
            "question": "Q1",
            "answer": "Good answer",
            "gold_answer": "Reference answer",
        },
        {
            "question": "Q2",
            "answer": "Bad answer",
            "gold_answer": "Reference answer",
        },
    ]

    results = metric.compute(qa_pairs)
    # First pair: (5+5+5)/3 = 5.0, Second pair: (2+2+1)/3 = 1.67, Average: (5.0+1.67)/2 = 3.33
    assert results[0]["score"] == pytest.approx(3.33, abs=0.1)
    assert len(results[0]["individual_scores"]) == 2


@patch("retrieval_evals.evals.grounded.metrics.longqa_answer.requests.post")
def test_longqa_answer_with_markdown(mock_post: Mock) -> None:
    """Test handling markdown code blocks in response."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '```json\n{"The answer aligns with current medical knowledge": 4, "The answer addresses the specific medical question": 4, "The answer communicates contraindications or risks": 4}\n```'
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    metric = LongQAAnswer(api_key="test-key")
    qa_pairs = [{"question": "Q", "answer": "A", "gold_answer": "G"}]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 4.0


def test_longqa_answer_empty_list() -> None:
    """Test with empty Q&A pairs list."""
    metric = LongQAAnswer(api_key="test-key")
    results = metric.compute([])

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"] == []
    assert results[0]["metadata"]["num_pairs"] == 0


@patch("retrieval_evals.evals.grounded.metrics.longqa_answer.requests.post")
def test_longqa_answer_metadata(mock_post: Mock) -> None:
    """Test metadata in results."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "The answer aligns with current medical knowledge": 5,
                            "The answer addresses the specific medical question": 5,
                            "The answer communicates contraindications or risks": 5,
                        }
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    model_name = "anthropic/claude-3.5-sonnet"
    metric = LongQAAnswer(api_key="test-key", model=model_name)
    qa_pairs = [{"question": "Q", "answer": "A", "gold_answer": "G"}]

    results = metric.compute(qa_pairs)
    metadata = results[0]["metadata"]

    assert metadata["num_pairs"] == 1
    assert metadata["model"] == model_name
    assert len(results[0]["individual_scores"]) == 1
    assert isinstance(results[0]["individual_scores"][0], dict)
