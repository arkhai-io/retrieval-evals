"""Tests for MORQAFaithfulness metric."""

import json
from unittest.mock import Mock, patch

from retrieval_evals.evals.grounded.metrics import MORQAFaithfulness


@patch("retrieval_evals.evals.grounded.metrics.morqa_faithfulness.requests.post")
def test_morqa_faithfulness_basic(mock_post: Mock) -> None:
    """Test basic MORQA faithfulness evaluation."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "facts": [
                                {
                                    "text": "Ibuprofen is an NSAID",
                                    "label": "supported_by_reference",
                                },
                                {
                                    "text": "Take 200mg every 4-6 hours",
                                    "label": "partially_supported",
                                },
                            ],
                            "critical_errors": [],
                            "atomic_faithfulness": 0.75,
                            "summary": "Answer is largely accurate with minor dosing imprecision",
                        }
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    metric = MORQAFaithfulness(api_key="test-key")
    qa_pairs = [
        {
            "question": "What is the treatment for headache?",
            "answer": "Ibuprofen 200mg every 4-6 hours",
            "gold_answer": "NSAIDs like ibuprofen 400-600mg every 6 hours",
        }
    ]

    results = metric.compute(qa_pairs)
    assert len(results) == 1
    assert results[0]["metric_name"] == "morqa_faithfulness"
    assert results[0]["score"] == 0.75
    assert len(results[0]["individual_scores"]) == 1
    assert results[0]["individual_scores"][0]["atomic_faithfulness"] == 0.75
    assert len(results[0]["individual_scores"][0]["facts"]) == 2


@patch("retrieval_evals.evals.grounded.metrics.morqa_faithfulness.requests.post")
def test_morqa_faithfulness_with_errors(mock_post: Mock) -> None:
    """Test with critical errors detected."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "facts": [
                                {"text": "Aspirin for heart attack", "label": "contradicted"},
                                {"text": "Take 100mg", "label": "not_in_reference"},
                            ],
                            "critical_errors": ["wrong_drug"],
                            "atomic_faithfulness": 0.0,
                            "summary": "Wrong medication recommended",
                        }
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    metric = MORQAFaithfulness(api_key="test-key")
    qa_pairs = [{"question": "Q", "answer": "A", "gold_answer": "G"}]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 0.0
    assert len(results[0]["individual_scores"][0]["critical_errors"]) == 1
    assert "wrong_drug" in results[0]["individual_scores"][0]["critical_errors"]


@patch("retrieval_evals.evals.grounded.metrics.morqa_faithfulness.requests.post")
def test_morqa_faithfulness_multiple_pairs(mock_post: Mock) -> None:
    """Test with multiple Q&A pairs."""
    mock_responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "facts": [{"text": "Fact 1", "label": "supported_by_reference"}],
                                "critical_errors": [],
                                "atomic_faithfulness": 1.0,
                                "summary": "Perfect alignment",
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
                                "facts": [{"text": "Fact 1", "label": "contradicted"}],
                                "critical_errors": ["wrong_diagnosis"],
                                "atomic_faithfulness": 0.0,
                                "summary": "Contradicted reference",
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

    metric = MORQAFaithfulness(api_key="test-key")
    qa_pairs = [
        {"question": "Q1", "answer": "A1", "gold_answer": "G1"},
        {"question": "Q2", "answer": "A2", "gold_answer": "G2"},
    ]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 0.5  # (1.0 + 0.0) / 2
    assert len(results[0]["individual_scores"]) == 2


def test_morqa_faithfulness_empty_list() -> None:
    """Test with empty Q&A pairs list."""
    metric = MORQAFaithfulness(api_key="test-key")
    results = metric.compute([])

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"] == []
    assert results[0]["metadata"]["num_pairs"] == 0


@patch("retrieval_evals.evals.grounded.metrics.morqa_faithfulness.requests.post")
def test_morqa_faithfulness_metadata(mock_post: Mock) -> None:
    """Test metadata in results."""
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "facts": [],
                            "critical_errors": [],
                            "atomic_faithfulness": 1.0,
                            "summary": "Test",
                        }
                    )
                }
            }
        ]
    }
    mock_post.return_value = mock_response

    model_name = "anthropic/claude-3.5-sonnet"
    metric = MORQAFaithfulness(api_key="test-key", model=model_name)
    qa_pairs = [{"question": "Q", "answer": "A", "gold_answer": "G"}]

    results = metric.compute(qa_pairs)
    metadata = results[0]["metadata"]

    assert metadata["num_pairs"] == 1
    assert metadata["model"] == model_name
    assert len(results[0]["individual_scores"]) == 1
