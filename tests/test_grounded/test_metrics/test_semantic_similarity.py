"""Tests for SemanticSimilarity metric."""

from unittest.mock import Mock, patch

import pytest

from retrieval_evals.evals.grounded.metrics import SemanticSimilarity


@patch("retrieval_evals.evals.grounded.metrics.semantic_similarity.SentenceTransformer")
def test_semantic_similarity_bi_encoder(mock_st: Mock) -> None:
    """Test semantic similarity with bi-encoder."""
    # Mock the model
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    # similarity returns 2D array where diagonal contains pair similarities
    mock_model.similarity.return_value = [[0.9, 0.1], [0.2, 0.7]]
    mock_st.return_value = mock_model

    metric = SemanticSimilarity(model_type="bi-encoder")
    qa_pairs = [
        {"question": "Q1", "answer": "yes", "gold_answer": "yes"},
        {"question": "Q2", "answer": "maybe", "gold_answer": "yes"},
    ]

    results = metric.compute(qa_pairs)
    assert len(results) == 1
    assert results[0]["metric_name"] == "semantic_similarity_bi-encoder"
    assert results[0]["score"] == 0.8
    assert results[0]["individual_scores"] == [0.9, 0.7]


@patch("retrieval_evals.evals.grounded.metrics.semantic_similarity.CrossEncoder")
def test_semantic_similarity_cross_encoder(mock_ce: Mock) -> None:
    """Test semantic similarity with cross-encoder."""
    # Mock the model
    mock_model = Mock()
    mock_model.predict.return_value = [0.95, 0.85]
    mock_ce.return_value = mock_model

    metric = SemanticSimilarity(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model_type="cross-encoder"
    )
    qa_pairs = [
        {"question": "Q1", "answer": "yes", "gold_answer": "yes"},
        {"question": "Q2", "answer": "maybe", "gold_answer": "yes"},
    ]

    results = metric.compute(qa_pairs)
    assert len(results) == 1
    assert results[0]["metric_name"] == "semantic_similarity_cross-encoder"
    assert abs(results[0]["score"] - 0.9) < 0.001
    assert results[0]["individual_scores"] == [0.95, 0.85]


def test_semantic_similarity_invalid_model_type() -> None:
    """Test that invalid model type raises error."""
    with pytest.raises(ValueError, match="model_type must be"):
        SemanticSimilarity(model_type="invalid")


@patch("retrieval_evals.evals.grounded.metrics.semantic_similarity.SentenceTransformer")
def test_semantic_similarity_empty_list(mock_st: Mock) -> None:
    """Test with empty Q&A pairs list."""
    mock_st.return_value = Mock()

    metric = SemanticSimilarity()
    results = metric.compute([])

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"] == []
    assert results[0]["metadata"]["num_pairs"] == 0


@patch("retrieval_evals.evals.grounded.metrics.semantic_similarity.SentenceTransformer")
def test_semantic_similarity_metadata(mock_st: Mock) -> None:
    """Test metadata in results."""
    mock_model = Mock()
    mock_model.encode.return_value = Mock()
    mock_model.similarity.return_value = [[0.9]]
    mock_st.return_value = mock_model

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    metric = SemanticSimilarity(model_name=model_name, model_type="bi-encoder")
    qa_pairs = [{"question": "Q1", "answer": "yes", "gold_answer": "yes"}]

    results = metric.compute(qa_pairs)
    metadata = results[0]["metadata"]

    assert metadata["num_pairs"] == 1
    assert metadata["model_name"] == model_name
    assert metadata["config"]["model_name"] == model_name
    assert metadata["config"]["model_type"] == "bi-encoder"


@patch("retrieval_evals.evals.grounded.metrics.semantic_similarity.CrossEncoder")
def test_semantic_similarity_cross_encoder_single_pair(mock_ce: Mock) -> None:
    """Test cross-encoder with single pair."""
    mock_model = Mock()
    mock_model.predict.return_value = [1.0]
    mock_ce.return_value = mock_model

    metric = SemanticSimilarity(model_type="cross-encoder")
    qa_pairs = [{"question": "Q1", "answer": "identical", "gold_answer": "identical"}]

    results = metric.compute(qa_pairs)
    assert results[0]["score"] == 1.0
    assert results[0]["individual_scores"] == [1.0]
    assert len(results[0]["individual_scores"]) == 1
