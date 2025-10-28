"""Tests for fact matching metric."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from retrieval_evals.evals.grounded.metrics.fact_matching import FactMatching
from retrieval_evals.types import QAPairWithGold


@pytest.fixture
def mock_api_key() -> str:
    """Return a mock API key for testing."""
    return "test-api-key"


@pytest.fixture
def fact_matching_metric(mock_api_key: str) -> FactMatching:
    """Create a FactMatching metric instance for testing."""
    return FactMatching(
        api_key=mock_api_key,
        llm_model="test-model",
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.75,
    )


class TestFactMatchingInit:
    """Tests for FactMatching initialization."""

    def test_init_with_defaults(self, mock_api_key: str) -> None:
        """Test initialization with default parameters."""
        metric = FactMatching(api_key=mock_api_key)

        assert metric.api_key == mock_api_key
        assert metric.llm_model == "anthropic/claude-3.5-sonnet"
        assert metric.similarity_threshold == 0.75
        assert metric.matching_strategy == "greedy"

    def test_init_with_custom_params(self, mock_api_key: str) -> None:
        """Test initialization with custom parameters."""
        # Use a real embedding model for this test
        metric = FactMatching(
            api_key=mock_api_key,
            llm_model="custom-model",
            embedding_model="all-MiniLM-L6-v2",  # Use real model
            similarity_threshold=0.85,
            matching_strategy="optimal",
        )

        assert metric.llm_model == "custom-model"
        assert metric.similarity_threshold == 0.85
        assert metric.matching_strategy == "optimal"

    def test_name_property(self, fact_matching_metric: FactMatching) -> None:
        """Test that name property returns correct value."""
        assert fact_matching_metric.name == "fact_matching"


class TestSimilarityMatrix:
    """Tests for similarity matrix computation."""

    def test_compute_similarity_matrix_basic(self, fact_matching_metric: FactMatching) -> None:
        """Test basic similarity matrix computation."""
        facts1 = ["Pneumonia is treated with antibiotics"]
        facts2 = ["Antibiotics treat bacterial pneumonia"]

        matrix = fact_matching_metric._compute_similarity_matrix(facts1, facts2)

        assert matrix.shape == (1, 1)
        assert 0.0 <= matrix[0, 0] <= 1.0
        # These should be similar
        assert matrix[0, 0] > 0.7

    def test_compute_similarity_matrix_multiple(self, fact_matching_metric: FactMatching) -> None:
        """Test similarity matrix with multiple facts."""
        facts1 = [
            "Pneumonia is treated with antibiotics",
            "Patient should rest",
            "Drink plenty of water",
        ]
        facts2 = [
            "Antibiotics are used for pneumonia",
            "Rest is important for recovery",
        ]

        matrix = fact_matching_metric._compute_similarity_matrix(facts1, facts2)

        assert matrix.shape == (3, 2)
        # All values should be between 0 and 1
        assert np.all((matrix >= 0) & (matrix <= 1))
        # First fact should match best with first gold fact
        assert matrix[0, 0] > matrix[0, 1]
        # Second fact should match best with second gold fact
        assert matrix[1, 1] > matrix[1, 0]

    def test_compute_similarity_matrix_empty(self, fact_matching_metric: FactMatching) -> None:
        """Test similarity matrix with empty inputs."""
        facts1: list[str] = []
        facts2 = ["Some fact"]

        matrix = fact_matching_metric._compute_similarity_matrix(facts1, facts2)

        assert matrix.size == 0


class TestGreedyMatching:
    """Tests for greedy matching algorithm."""

    def test_greedy_matching_perfect(self, fact_matching_metric: FactMatching) -> None:
        """Test greedy matching with clear 1:1 matches."""
        # High similarity on diagonal
        similarity_matrix = np.array(
            [
                [0.95, 0.20, 0.15],
                [0.25, 0.90, 0.18],
                [0.22, 0.30, 0.88],
            ]
        )

        matches, used_gold = fact_matching_metric._greedy_matching(similarity_matrix)

        assert len(matches) == 3
        assert len(used_gold) == 3
        # Check that we got the diagonal matches
        assert {(m["model_idx"], m["gold_idx"]) for m in matches} == {
            (0, 0),
            (1, 1),
            (2, 2),
        }

    def test_greedy_matching_below_threshold(self, fact_matching_metric: FactMatching) -> None:
        """Test greedy matching when similarities are below threshold."""
        # All similarities below 0.75 threshold
        similarity_matrix = np.array(
            [
                [0.60, 0.55],
                [0.50, 0.65],
            ]
        )

        matches, used_gold = fact_matching_metric._greedy_matching(similarity_matrix)

        assert len(matches) == 0
        assert len(used_gold) == 0

    def test_greedy_matching_partial(self, fact_matching_metric: FactMatching) -> None:
        """Test greedy matching with only some matches above threshold."""
        similarity_matrix = np.array(
            [
                [0.85, 0.30],
                [0.40, 0.60],  # Below threshold
            ]
        )

        matches, used_gold = fact_matching_metric._greedy_matching(similarity_matrix)

        assert len(matches) == 1
        assert matches[0]["model_idx"] == 0
        assert matches[0]["gold_idx"] == 0


class TestOptimalMatching:
    """Tests for optimal matching algorithm."""

    def test_optimal_matching_perfect(self, fact_matching_metric: FactMatching) -> None:
        """Test optimal matching with clear 1:1 matches."""
        similarity_matrix = np.array(
            [
                [0.95, 0.20],
                [0.25, 0.90],
            ]
        )

        matches, used_gold = fact_matching_metric._optimal_matching(similarity_matrix)

        assert len(matches) == 2
        assert {(m["model_idx"], m["gold_idx"]) for m in matches} == {(0, 0), (1, 1)}

    def test_optimal_matching_below_threshold(self, fact_matching_metric: FactMatching) -> None:
        """Test optimal matching when similarities are below threshold."""
        similarity_matrix = np.array(
            [
                [0.60, 0.55],
                [0.50, 0.65],
            ]
        )

        matches, used_gold = fact_matching_metric._optimal_matching(similarity_matrix)

        assert len(matches) == 0


class TestMatchFacts:
    """Tests for fact matching."""

    def test_match_facts_perfect_match(self, fact_matching_metric: FactMatching) -> None:
        """Test matching when facts align perfectly."""
        model_facts = ["Fact A", "Fact B"]
        gold_facts = ["Fact A", "Fact B"]

        result = fact_matching_metric._match_facts(model_facts, gold_facts)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert len(result["matches"]) == 2
        assert len(result["unmatched_model"]) == 0
        assert len(result["unmatched_gold"]) == 0

    def test_match_facts_partial_match(self, fact_matching_metric: FactMatching) -> None:
        """Test matching with partial overlap."""
        model_facts = [
            "Pneumonia is treated with antibiotics",
            "Patient should get rest",  # More similar to "Rest is recommended"
            "Extra unmatched fact completely different",
        ]
        gold_facts = [
            "Antibiotics treat pneumonia",
            "Rest is recommended for recovery",
        ]

        result = fact_matching_metric._match_facts(model_facts, gold_facts)

        # Should match at least 1 out of 3 model facts (first one definitely matches)
        assert result["precision"] >= 1 / 3
        # Check we got some matches
        assert len(result["matches"]) >= 1
        assert len(result["unmatched_model"]) >= 1
        # We should have at least 1 unmatched gold if not all matched
        if len(result["matches"]) < 2:
            assert len(result["unmatched_gold"]) >= 1

    def test_match_facts_empty_model(self, fact_matching_metric: FactMatching) -> None:
        """Test matching with empty model facts."""
        model_facts: list[str] = []
        gold_facts = ["Some fact"]

        result = fact_matching_metric._match_facts(model_facts, gold_facts)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_match_facts_empty_gold(self, fact_matching_metric: FactMatching) -> None:
        """Test matching with empty gold facts."""
        model_facts = ["Some fact"]
        gold_facts: list[str] = []

        result = fact_matching_metric._match_facts(model_facts, gold_facts)

        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0


class TestLLMCalls:
    """Tests for LLM API calls."""

    def test_call_llm_success(self, fact_matching_metric: FactMatching) -> None:
        """Test successful LLM call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"facts": ["Fact 1", "Fact 2"]}'}}]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            result = fact_matching_metric._call_llm("test prompt")

        assert result == {"facts": ["Fact 1", "Fact 2"]}

    def test_call_llm_with_json_markers(self, fact_matching_metric: FactMatching) -> None:
        """Test LLM call when response has JSON code markers."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '```json\n{"facts": ["Fact 1"]}\n```'}}]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            result = fact_matching_metric._call_llm("test prompt")

        assert result == {"facts": ["Fact 1"]}

    def test_extract_facts_success(self, fact_matching_metric: FactMatching) -> None:
        """Test successful fact extraction."""
        with patch.object(
            fact_matching_metric,
            "_call_llm",
            return_value={"facts": ["Fact A", "Fact B"]},
        ):
            facts = fact_matching_metric._extract_facts("Question?", "Answer text", "model")

        assert facts == ["Fact A", "Fact B"]

    def test_extract_facts_error(self, fact_matching_metric: FactMatching) -> None:
        """Test fact extraction with error."""
        with patch.object(fact_matching_metric, "_call_llm", side_effect=Exception("API Error")):
            facts = fact_matching_metric._extract_facts("Question?", "Answer text", "model")

        assert facts == []


class TestCompute:
    """Tests for compute method."""

    def test_compute_empty_pairs(self, fact_matching_metric: FactMatching) -> None:
        """Test compute with empty QA pairs."""
        qa_pairs: list[QAPairWithGold] = []

        results = fact_matching_metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "fact_matching"
        assert results[0]["score"] == 0.0
        assert results[0]["individual_scores"] == []

    def test_compute_single_pair(self, fact_matching_metric: FactMatching) -> None:
        """Test compute with single QA pair."""
        qa_pairs: list[QAPairWithGold] = [
            {
                "question": "What treats pneumonia?",
                "answer": "Antibiotics treat pneumonia effectively.",
                "gold_answer": "Pneumonia is treated with antibiotics.",
            }
        ]

        # Mock the extraction and matching
        with patch.object(
            fact_matching_metric,
            "_extract_facts",
            side_effect=[
                ["Antibiotics treat pneumonia"],  # model facts
                ["Pneumonia is treated with antibiotics"],  # gold facts
            ],
        ):
            results = fact_matching_metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "fact_matching"
        assert 0.0 <= results[0]["score"] <= 1.0
        assert len(results[0]["individual_scores"]) == 1

        individual = results[0]["individual_scores"][0]
        assert "model_facts" in individual
        assert "gold_facts" in individual
        assert "matching" in individual

    def test_compute_multiple_pairs(self, fact_matching_metric: FactMatching) -> None:
        """Test compute with multiple QA pairs."""
        qa_pairs: list[QAPairWithGold] = [
            {
                "question": "Q1?",
                "answer": "A1",
                "gold_answer": "Gold A1",
            },
            {
                "question": "Q2?",
                "answer": "A2",
                "gold_answer": "Gold A2",
            },
        ]

        with patch.object(
            fact_matching_metric,
            "_extract_facts",
            return_value=["Fact"],
        ):
            results = fact_matching_metric.compute(qa_pairs)

        assert len(results[0]["individual_scores"]) == 2
        assert results[0]["metadata"]["num_pairs"] == 2

    def test_compute_with_error(self, fact_matching_metric: FactMatching) -> None:
        """Test compute handles errors gracefully."""
        qa_pairs: list[QAPairWithGold] = [
            {
                "question": "Q?",
                "answer": "A",
                "gold_answer": "Gold",
            }
        ]

        with patch.object(
            fact_matching_metric,
            "_extract_facts",
            side_effect=Exception("Test error"),
        ):
            results = fact_matching_metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["score"] == 0.0
        assert "error" in results[0]["individual_scores"][0]["matching"]
