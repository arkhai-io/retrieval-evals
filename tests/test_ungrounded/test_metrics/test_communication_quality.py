"""Tests for CommunicationQuality metric."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from retrieval_evals.evals.ungrounded.metrics import CommunicationQuality
from tests.utils import create_temp_json


@pytest.fixture
def mock_llm_response() -> dict[str, int]:
    """Mock LLM response for testing."""
    return {
        "tone_appropriateness": 5,
        "professionalism": 4,
        "bias_and_fairness": 5,
    }


@pytest.fixture
def communication_quality_metric(mock_llm_response: dict[str, int]) -> CommunicationQuality:
    """Create CommunicationQuality metric with mocked LLM calls."""
    metric = CommunicationQuality(api_key="test-key", model="test-model")
    metric._call_llm = MagicMock(return_value=mock_llm_response)
    return metric


def test_communication_quality_basic(
    communication_quality_metric: CommunicationQuality,
) -> None:
    """Test basic communication quality computation."""
    qa_pairs = [
        {
            "question": "What is the capital of France?",
            "answer": "The capital of France is Paris.",
        },
        {
            "question": "How do vaccines work?",
            "answer": "Vaccines train your immune system to recognize pathogens.",
        },
    ]

    results = communication_quality_metric.compute(qa_pairs)

    assert len(results) == 1
    assert results[0]["metric_name"] == "communication_quality"
    # Average of (5+4+5)/3 = 4.67 for each pair
    assert results[0]["score"] == pytest.approx(4.67, abs=0.01)
    assert len(results[0]["individual_scores"]) == 2


def test_communication_quality_empty_list(
    communication_quality_metric: CommunicationQuality,
) -> None:
    """Test with empty Q&A pairs list."""
    results = communication_quality_metric.compute([])

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"] == []
    assert results[0]["metadata"]["num_pairs"] == 0


def test_communication_quality_individual_scores(
    communication_quality_metric: CommunicationQuality,
) -> None:
    """Test individual score structure."""
    qa_pairs = [
        {"question": "Test question?", "answer": "Test answer."},
    ]

    results = communication_quality_metric.compute(qa_pairs)
    individual = results[0]["individual_scores"][0]

    assert "tone_appropriateness" in individual
    assert "professionalism" in individual
    assert "bias_and_fairness" in individual
    assert all(1 <= v <= 5 for v in individual.values())


def test_communication_quality_metadata(
    communication_quality_metric: CommunicationQuality,
) -> None:
    """Test metadata in results."""
    qa_pairs = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]

    results = communication_quality_metric.compute(qa_pairs)
    metadata = results[0]["metadata"]

    assert metadata["num_pairs"] == 2
    assert metadata["model"] == "test-model"
    assert "config" in metadata
    # Should not contain api_key
    assert "api_key" not in metadata["config"]


def test_communication_quality_api_call() -> None:
    """Test that API is called correctly."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"tone_appropriateness": 5, "professionalism": 4, "bias_and_fairness": 5}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        metric = CommunicationQuality(
            api_key="test-key",
            model="anthropic/claude-3.5-sonnet",
            base_url="https://openrouter.ai/api/v1",
        )

        qa_pairs = [
            {"question": "What is AI?", "answer": "AI is artificial intelligence."},
        ]

        results = metric.compute(qa_pairs)

        # Verify API was called
        assert mock_post.called
        call_args = mock_post.call_args

        # Check URL
        assert call_args[0][0] == "https://openrouter.ai/api/v1/chat/completions"

        # Check headers
        assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"

        # Check JSON payload
        json_data = call_args[1]["json"]
        assert json_data["model"] == "anthropic/claude-3.5-sonnet"
        assert json_data["temperature"] == 0.0

        # Check results
        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(4.67, abs=0.01)


def test_communication_quality_error_handling() -> None:
    """Test error handling when LLM call fails."""
    metric = CommunicationQuality(api_key="test-key")

    def mock_error(*_args: str, **_kwargs: str) -> None:
        raise Exception("API error")

    metric._call_llm = mock_error

    qa_pairs = [
        {"question": "Test question", "answer": "Test answer"},
    ]

    results = metric.compute(qa_pairs)

    # Should still return results with 0 score for failed evaluation
    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"][0] == {}


def test_communication_quality_json_with_markdown() -> None:
    """Test parsing JSON from markdown code blocks."""
    with patch("requests.post") as mock_post:
        # Test with ```json blocks
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"tone_appropriateness": 5, "professionalism": 4, "bias_and_fairness": 5}\n```'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        metric = CommunicationQuality(api_key="test-key")
        qa_pairs = [{"question": "Q", "answer": "A"}]

        results = metric.compute(qa_pairs)
        assert results[0]["score"] == pytest.approx(4.67, abs=0.01)


def test_communication_quality_from_file() -> None:
    """Test loading data from file."""
    metric = CommunicationQuality(api_key="test-key")
    metric._call_llm = MagicMock(
        return_value={
            "tone_appropriateness": 5,
            "professionalism": 5,
            "bias_and_fairness": 5,
        }
    )

    test_data = [
        {"question": "Q1", "answer": "A1"},
    ]
    temp_path = create_temp_json(test_data)

    try:
        results = metric.evaluate(temp_path)
        assert results[0]["score"] == 5.0
    finally:
        temp_path.unlink()


def test_communication_quality_varying_scores() -> None:
    """Test with varying scores across dimensions."""
    metric = CommunicationQuality(api_key="test-key")

    # Mock different scores for different pairs
    mock_responses = [
        {"tone_appropriateness": 5, "professionalism": 5, "bias_and_fairness": 5},
        {"tone_appropriateness": 3, "professionalism": 2, "bias_and_fairness": 4},
        {"tone_appropriateness": 4, "professionalism": 4, "bias_and_fairness": 3},
    ]

    metric._call_llm = MagicMock(side_effect=mock_responses)

    qa_pairs = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
        {"question": "Q3", "answer": "A3"},
    ]

    results = metric.compute(qa_pairs)

    # First: (5+5+5)/3 = 5.0
    # Second: (3+2+4)/3 = 3.0
    # Third: (4+4+3)/3 = 3.67
    # Average: (5.0 + 3.0 + 3.67) / 3 = 3.89
    assert results[0]["score"] == pytest.approx(3.89, abs=0.01)
    assert len(results[0]["individual_scores"]) == 3


def test_communication_quality_metric_name() -> None:
    """Test that metric name is correct."""
    metric = CommunicationQuality(api_key="test-key")
    assert metric.name == "communication_quality"


def test_communication_quality_prompt_formatting() -> None:
    """Test that prompt is formatted correctly."""
    metric = CommunicationQuality(api_key="test-key")

    # Mock to capture the prompt
    called_prompts = []

    def capture_prompt(prompt: str) -> dict[str, int]:
        called_prompts.append(prompt)
        return {
            "tone_appropriateness": 5,
            "professionalism": 5,
            "bias_and_fairness": 5,
        }

    metric._call_llm = capture_prompt

    qa_pairs = [
        {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence."},
    ]

    metric.compute(qa_pairs)

    # Check that prompt contains question and answer
    assert len(called_prompts) == 1
    assert "What is AI?" in called_prompts[0]
    assert "AI stands for Artificial Intelligence." in called_prompts[0]
