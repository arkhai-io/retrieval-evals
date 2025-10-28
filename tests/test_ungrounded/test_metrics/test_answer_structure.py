"""Tests for answer structure metric."""

from unittest.mock import Mock, patch

import pytest

from retrieval_evals.evals.ungrounded.metrics.answer_structure import AnswerStructure
from retrieval_evals.types import QAPair


@pytest.fixture
def mock_api_key() -> str:
    """Return a mock API key for testing."""
    return "test-api-key"


@pytest.fixture
def answer_structure_metric(mock_api_key: str) -> AnswerStructure:
    """Create an AnswerStructure metric instance for testing."""
    return AnswerStructure(api_key=mock_api_key, model="test-model")


class TestAnswerStructureInit:
    """Tests for AnswerStructure initialization."""

    def test_init_with_defaults(self, mock_api_key: str) -> None:
        """Test initialization with default parameters."""
        metric = AnswerStructure(api_key=mock_api_key)

        assert metric.api_key == mock_api_key
        assert metric.model == "anthropic/claude-3.5-sonnet"

    def test_init_with_custom_params(self, mock_api_key: str) -> None:
        """Test initialization with custom parameters."""
        metric = AnswerStructure(
            api_key=mock_api_key,
            model="custom-model",
            base_url="https://custom.api",
        )

        assert metric.model == "custom-model"
        assert metric.base_url == "https://custom.api"

    def test_name_property(self, answer_structure_metric: AnswerStructure) -> None:
        """Test that name property returns correct value."""
        assert answer_structure_metric.name == "answer_structure"


class TestLLMCalls:
    """Tests for LLM API calls."""

    def test_call_llm_success(self, answer_structure_metric: AnswerStructure) -> None:
        """Test successful LLM call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"organization": 5, "formatting": 4, '
                            '"hierarchy": 5, "clarity": 4, "summary": "Well structured"}'
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            result = answer_structure_metric._call_llm("test prompt")

        assert result["organization"] == 5
        assert result["formatting"] == 4
        assert "summary" in result

    def test_call_llm_with_json_markers(self, answer_structure_metric: AnswerStructure) -> None:
        """Test LLM call when response has JSON code markers."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '```json\n{"organization": 3, "formatting": 3, '
                            '"hierarchy": 3, "clarity": 3, "summary": "OK"}\n```'
                        )
                    }
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            result = answer_structure_metric._call_llm("test prompt")

        assert result["organization"] == 3


class TestCompute:
    """Tests for compute method."""

    def test_compute_empty_pairs(self, answer_structure_metric: AnswerStructure) -> None:
        """Test compute with empty QA pairs."""
        qa_pairs: list[QAPair] = []

        results = answer_structure_metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "answer_structure"
        assert results[0]["score"] == 0.0
        assert results[0]["individual_scores"] == []

    def test_compute_single_pair(self, answer_structure_metric: AnswerStructure) -> None:
        """Test compute with single QA pair."""
        qa_pairs: list[QAPair] = [
            {
                "question": "What is good structure?",
                "answer": "Good structure has clear organization.",
            }
        ]

        mock_llm_response = {
            "organization": 4,
            "formatting": 4,
            "hierarchy": 4,
            "clarity": 4,
            "summary": "Well structured answer",
        }

        with patch.object(answer_structure_metric, "_call_llm", return_value=mock_llm_response):
            results = answer_structure_metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "answer_structure"
        # Score should be (4+4+4+4)/4 = 4, normalized to 0-1: (4-1)/4 = 0.75
        assert results[0]["score"] == pytest.approx(0.75, abs=0.01)
        assert len(results[0]["individual_scores"]) == 1

    def test_compute_multiple_pairs(self, answer_structure_metric: AnswerStructure) -> None:
        """Test compute with multiple QA pairs."""
        qa_pairs: list[QAPair] = [
            {"question": "Q1?", "answer": "A1"},
            {"question": "Q2?", "answer": "A2"},
        ]

        mock_llm_response = {
            "organization": 3,
            "formatting": 3,
            "hierarchy": 3,
            "clarity": 3,
            "summary": "Average",
        }

        with patch.object(answer_structure_metric, "_call_llm", return_value=mock_llm_response):
            results = answer_structure_metric.compute(qa_pairs)

        assert len(results[0]["individual_scores"]) == 2
        assert results[0]["metadata"]["num_pairs"] == 2

    def test_compute_with_error(self, answer_structure_metric: AnswerStructure) -> None:
        """Test compute handles errors gracefully."""
        qa_pairs: list[QAPair] = [
            {"question": "Q?", "answer": "A"},
        ]

        with patch.object(answer_structure_metric, "_call_llm", side_effect=Exception("API Error")):
            results = answer_structure_metric.compute(qa_pairs)

        assert len(results) == 1
        # Should still return a result, but with 0 scores
        assert results[0]["score"] == 0.0
        assert results[0]["individual_scores"][0]["organization"] == 0

    def test_score_normalization(self, answer_structure_metric: AnswerStructure) -> None:
        """Test that scores are correctly normalized from 1-5 to 0-1."""
        qa_pairs: list[QAPair] = [
            {"question": "Q?", "answer": "A"},
        ]

        # Test with perfect score (5 on all dimensions)
        perfect_response = {
            "organization": 5,
            "formatting": 5,
            "hierarchy": 5,
            "clarity": 5,
            "summary": "Perfect",
        }

        with patch.object(answer_structure_metric, "_call_llm", return_value=perfect_response):
            results = answer_structure_metric.compute(qa_pairs)

        # (5-1)/4 = 1.0
        assert results[0]["score"] == 1.0

        # Test with minimum score (1 on all dimensions)
        poor_response = {
            "organization": 1,
            "formatting": 1,
            "hierarchy": 1,
            "clarity": 1,
            "summary": "Poor",
        }

        with patch.object(answer_structure_metric, "_call_llm", return_value=poor_response):
            results = answer_structure_metric.compute(qa_pairs)

        # (1-1)/4 = 0.0
        assert results[0]["score"] == 0.0

    def test_metadata_includes_dimensions(self, answer_structure_metric: AnswerStructure) -> None:
        """Test that metadata includes dimension names."""
        qa_pairs: list[QAPair] = [{"question": "Q?", "answer": "A"}]

        mock_response = {
            "organization": 3,
            "formatting": 3,
            "hierarchy": 3,
            "clarity": 3,
            "summary": "OK",
        }

        with patch.object(answer_structure_metric, "_call_llm", return_value=mock_response):
            results = answer_structure_metric.compute(qa_pairs)

        dimensions = results[0]["metadata"]["dimensions"]
        assert "organization" in dimensions
        assert "formatting" in dimensions
        assert "hierarchy" in dimensions
        assert "clarity" in dimensions
