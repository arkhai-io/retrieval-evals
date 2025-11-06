"""Tests for StyleConsistency metric."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from retrieval_evals.evals.ungrounded.metrics import StyleConsistency


@pytest.fixture
def reference_pairs() -> list[dict[str, str]]:
    """Reference Q&A pairs showing desired style."""
    return [
        {
            "question": "What is Python?",
            "answer": "Python is a high-level programming language. It's known for its simplicity and readability.",
        },
        {
            "question": "How do I install packages?",
            "answer": "You can install packages using pip. Simply run 'pip install package-name' in your terminal.",
        },
        {
            "question": "What are decorators?",
            "answer": "Decorators are functions that modify other functions. They're a powerful feature for code reuse.",
        },
    ]


@pytest.fixture
def mock_llm_response() -> dict[str, int]:
    """Mock LLM response for testing."""
    return {
        "tone_and_formality": 5,
        "structure_and_format": 4,
        "length_and_verbosity": 5,
        "vocabulary_and_complexity": 4,
        "response_pattern": 5,
    }


@pytest.fixture
def style_consistency_metric(
    reference_pairs: list[dict[str, str]], mock_llm_response: dict[str, int]
) -> StyleConsistency:
    """Create StyleConsistency metric with mocked LLM calls."""
    metric = StyleConsistency(
        api_key="test-key", reference_pairs=reference_pairs, model="test-model"
    )
    metric._call_llm = MagicMock(return_value=mock_llm_response)
    return metric


def test_style_consistency_basic(
    style_consistency_metric: StyleConsistency,
) -> None:
    """Test basic style consistency computation."""
    qa_pairs = [
        {
            "question": "What is JavaScript?",
            "answer": "JavaScript is a programming language. It's commonly used for web development.",
        },
        {
            "question": "How do classes work?",
            "answer": "Classes are blueprints for objects. They define properties and methods.",
        },
    ]

    results = style_consistency_metric.compute(qa_pairs)

    assert len(results) == 1
    assert results[0]["metric_name"] == "style_consistency"
    # Average of (5+4+5+4+5)/5 = 4.6 for each pair
    assert results[0]["score"] == pytest.approx(4.6, abs=0.01)
    assert len(results[0]["individual_scores"]) == 2


def test_style_consistency_empty_list(
    style_consistency_metric: StyleConsistency,
) -> None:
    """Test with empty Q&A pairs list."""
    results = style_consistency_metric.compute([])

    assert len(results) == 1
    assert results[0]["score"] == 0.0
    assert results[0]["individual_scores"] == []
    assert results[0]["metadata"]["num_pairs"] == 0
    assert results[0]["metadata"]["num_references"] == 3


def test_style_consistency_individual_scores(
    style_consistency_metric: StyleConsistency,
) -> None:
    """Test individual score structure."""
    qa_pairs = [
        {"question": "Test question?", "answer": "Test answer."},
    ]

    results = style_consistency_metric.compute(qa_pairs)
    individual = results[0]["individual_scores"][0]

    assert "tone_and_formality" in individual
    assert "structure_and_format" in individual
    assert "length_and_verbosity" in individual
    assert "vocabulary_and_complexity" in individual
    assert "response_pattern" in individual
    assert all(1 <= v <= 5 for v in individual.values())


def test_style_consistency_metadata(
    style_consistency_metric: StyleConsistency,
) -> None:
    """Test metadata in results."""
    qa_pairs = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
    ]

    results = style_consistency_metric.compute(qa_pairs)
    metadata = results[0]["metadata"]

    assert metadata["num_pairs"] == 2
    assert metadata["num_references"] == 3
    assert metadata["model"] == "test-model"
    assert "config" in metadata
    # Should not contain api_key
    assert "api_key" not in metadata["config"]


def test_style_consistency_reference_formatting(
    reference_pairs: list[dict[str, str]],
) -> None:
    """Test that references are formatted correctly."""
    metric = StyleConsistency(api_key="test-key", reference_pairs=reference_pairs)

    formatted = metric._format_reference_examples()

    # Check all references are included
    assert "EXAMPLE 1:" in formatted
    assert "EXAMPLE 2:" in formatted
    assert "EXAMPLE 3:" in formatted

    # Check content is included
    assert "What is Python?" in formatted
    assert "Python is a high-level" in formatted


def test_style_consistency_max_references(
    reference_pairs: list[dict[str, str]],
) -> None:
    """Test max_reference_examples parameter."""
    metric = StyleConsistency(
        api_key="test-key", reference_pairs=reference_pairs, max_reference_examples=2
    )

    assert len(metric.reference_pairs) == 2

    formatted = metric._format_reference_examples()
    assert "EXAMPLE 1:" in formatted
    assert "EXAMPLE 2:" in formatted
    assert "EXAMPLE 3:" not in formatted


def test_style_consistency_api_call(reference_pairs: list[dict[str, str]]) -> None:
    """Test that API is called correctly."""
    with patch("requests.post") as mock_post:
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"tone_and_formality": 5, "structure_and_format": 4, "length_and_verbosity": 5, "vocabulary_and_complexity": 4, "response_pattern": 5}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        metric = StyleConsistency(
            api_key="test-key",
            reference_pairs=reference_pairs,
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
        assert results[0]["score"] == pytest.approx(4.6, abs=0.01)


def test_style_consistency_error_handling(
    reference_pairs: list[dict[str, str]],
) -> None:
    """Test error handling when LLM call fails."""
    metric = StyleConsistency(api_key="test-key", reference_pairs=reference_pairs)

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


def test_style_consistency_json_with_markdown(
    reference_pairs: list[dict[str, str]],
) -> None:
    """Test parsing JSON from markdown code blocks."""
    with patch("requests.post") as mock_post:
        # Test with ```json blocks
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"tone_and_formality": 5, "structure_and_format": 4, "length_and_verbosity": 5, "vocabulary_and_complexity": 4, "response_pattern": 5}\n```'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        metric = StyleConsistency(api_key="test-key", reference_pairs=reference_pairs)
        qa_pairs = [{"question": "Q", "answer": "A"}]

        results = metric.compute(qa_pairs)
        assert results[0]["score"] == pytest.approx(4.6, abs=0.01)


def test_style_consistency_varying_scores(
    reference_pairs: list[dict[str, str]],
) -> None:
    """Test with varying scores across evaluations."""
    metric = StyleConsistency(api_key="test-key", reference_pairs=reference_pairs)

    # Mock different scores for different pairs
    mock_responses = [
        {
            "tone_and_formality": 5,
            "structure_and_format": 5,
            "length_and_verbosity": 5,
            "vocabulary_and_complexity": 5,
            "response_pattern": 5,
        },
        {
            "tone_and_formality": 3,
            "structure_and_format": 2,
            "length_and_verbosity": 4,
            "vocabulary_and_complexity": 3,
            "response_pattern": 3,
        },
        {
            "tone_and_formality": 4,
            "structure_and_format": 4,
            "length_and_verbosity": 3,
            "vocabulary_and_complexity": 4,
            "response_pattern": 4,
        },
    ]

    metric._call_llm = MagicMock(side_effect=mock_responses)

    qa_pairs = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
        {"question": "Q3", "answer": "A3"},
    ]

    results = metric.compute(qa_pairs)

    # First: (5+5+5+5+5)/5 = 5.0
    # Second: (3+2+4+3+3)/5 = 3.0
    # Third: (4+4+3+4+4)/5 = 3.8
    # Average: (5.0 + 3.0 + 3.8) / 3 = 3.93
    assert results[0]["score"] == pytest.approx(3.93, abs=0.01)
    assert len(results[0]["individual_scores"]) == 3


def test_style_consistency_metric_name(reference_pairs: list[dict[str, str]]) -> None:
    """Test that metric name is correct."""
    metric = StyleConsistency(api_key="test-key", reference_pairs=reference_pairs)
    assert metric.name == "style_consistency"


def test_style_consistency_prompt_includes_references(
    reference_pairs: list[dict[str, str]],
) -> None:
    """Test that prompt includes reference examples."""
    metric = StyleConsistency(api_key="test-key", reference_pairs=reference_pairs)

    # Mock to capture the prompt
    called_prompts = []

    def capture_prompt(prompt: str) -> dict[str, int]:
        called_prompts.append(prompt)
        return {
            "tone_and_formality": 5,
            "structure_and_format": 5,
            "length_and_verbosity": 5,
            "vocabulary_and_complexity": 5,
            "response_pattern": 5,
        }

    metric._call_llm = capture_prompt

    qa_pairs = [
        {"question": "New question?", "answer": "New answer."},
    ]

    metric.compute(qa_pairs)

    # Check that prompt contains reference examples and new Q&A
    assert len(called_prompts) == 1
    prompt = called_prompts[0]
    assert "What is Python?" in prompt  # From reference
    assert "New question?" in prompt  # New question
    assert "New answer." in prompt  # New answer
