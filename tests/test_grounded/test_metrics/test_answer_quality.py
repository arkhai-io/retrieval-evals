"""Tests for AnswerQuality metric."""

from unittest.mock import Mock, patch

import pytest

from retrieval_evals.evals.grounded.metrics.answer_quality import AnswerQuality


@pytest.fixture
def answer_quality():
    """Create AnswerQuality metric with test API key."""
    return AnswerQuality(api_key="test-key")


class TestAnswerQualityInit:
    """Test AnswerQuality initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        metric = AnswerQuality(api_key="test-key")
        assert metric.api_key == "test-key"
        assert metric.model == "anthropic/claude-3.5-sonnet"
        assert metric.base_url == "https://openrouter.ai/api/v1"
        assert metric.name == "answer_quality"

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        metric = AnswerQuality(
            api_key="custom-key",
            model="anthropic/claude-3-opus",
            base_url="https://custom.api.com",
        )
        assert metric.api_key == "custom-key"
        assert metric.model == "anthropic/claude-3-opus"
        assert metric.base_url == "https://custom.api.com"


class TestLLMCalls:
    """Test LLM API calls."""

    def test_call_llm_success(self, answer_quality):
        """Test successful LLM API call."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "completeness_score": 4,
                            "completeness_reasoning": "Test reasoning",
                            "key_missing_info": ["Missing item"],
                            "correctness_score": 5,
                            "correctness_reasoning": "All correct",
                            "factual_errors": []
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = answer_quality._call_llm("test prompt")

            # Verify API call
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            assert call_kwargs["json"]["model"] == "anthropic/claude-3.5-sonnet"
            assert call_kwargs["json"]["temperature"] == 0.0
            assert call_kwargs["timeout"] == 60

            # Verify result
            assert result["completeness_score"] == 4
            assert result["correctness_score"] == 5

    def test_call_llm_with_json_markdown(self, answer_quality):
        """Test LLM call with JSON in markdown code blocks."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """```json
                        {
                            "completeness_score": 3,
                            "correctness_score": 4
                        }
                        ```"""
                    }
                }
            ]
        }
        mock_response.raise_for_status.return_value = None

        with patch("requests.post", return_value=mock_response):
            result = answer_quality._call_llm("test prompt")
            assert result["completeness_score"] == 3
            assert result["correctness_score"] == 4


class TestEvaluateAnswer:
    """Test single answer evaluation."""

    def test_evaluate_complete_and_correct(self, answer_quality):
        """Test evaluation of a complete and correct answer."""
        mock_llm_result = {
            "completeness_score": 5,
            "completeness_reasoning": "Covers all key points",
            "key_missing_info": [],
            "correctness_score": 5,
            "correctness_reasoning": "All facts accurate",
            "factual_errors": [],
        }

        with patch.object(answer_quality, "_call_llm", return_value=mock_llm_result):
            result = answer_quality._evaluate_answer(
                question="What is the capital of France?",
                model_answer="The capital of France is Paris.",
                gold_answer="Paris is the capital of France.",
            )

            assert result["completeness_score"] == 5
            assert result["correctness_score"] == 5
            assert result["overall_score"] == 5.0
            assert len(result["key_missing_info"]) == 0
            assert len(result["factual_errors"]) == 0

    def test_evaluate_incomplete_answer(self, answer_quality):
        """Test evaluation of an incomplete answer."""
        mock_llm_result = {
            "completeness_score": 2,
            "completeness_reasoning": "Missing most details",
            "key_missing_info": ["Treatment duration", "Side effects"],
            "correctness_score": 4,
            "correctness_reasoning": "What's there is correct",
            "factual_errors": [],
        }

        with patch.object(answer_quality, "_call_llm", return_value=mock_llm_result):
            result = answer_quality._evaluate_answer(
                question="What is the treatment?",
                model_answer="Use antibiotics.",
                gold_answer="Use antibiotics for 7-10 days. Watch for side effects.",
            )

            assert result["completeness_score"] == 2
            assert result["correctness_score"] == 4
            assert result["overall_score"] == 3.0  # (2+4)/2
            assert len(result["key_missing_info"]) == 2

    def test_evaluate_incorrect_answer(self, answer_quality):
        """Test evaluation of an answer with factual errors."""
        mock_llm_result = {
            "completeness_score": 4,
            "completeness_reasoning": "Covers topics but wrong facts",
            "key_missing_info": [],
            "correctness_score": 2,
            "correctness_reasoning": "Contains major error",
            "factual_errors": ["States capital is Lyon instead of Paris"],
        }

        with patch.object(answer_quality, "_call_llm", return_value=mock_llm_result):
            result = answer_quality._evaluate_answer(
                question="What is the capital of France?",
                model_answer="The capital of France is Lyon.",
                gold_answer="Paris is the capital of France.",
            )

            assert result["completeness_score"] == 4
            assert result["correctness_score"] == 2
            assert result["overall_score"] == 3.0
            assert len(result["factual_errors"]) == 1

    def test_evaluate_cannot_answer_response(self, answer_quality):
        """Test evaluation of 'cannot answer' response."""
        mock_llm_result = {
            "completeness_score": 1,
            "completeness_reasoning": "Does not provide answer",
            "key_missing_info": ["All information"],
            "correctness_score": 3,
            "correctness_reasoning": "No false info, just unhelpful",
            "factual_errors": [],
        }

        with patch.object(answer_quality, "_call_llm", return_value=mock_llm_result):
            result = answer_quality._evaluate_answer(
                question="What is the treatment?",
                model_answer="I cannot provide this information.",
                gold_answer="Use antibiotics.",
            )

            assert result["completeness_score"] == 1
            assert result["overall_score"] == 2.0  # (1+3)/2

    def test_evaluate_answer_with_error(self, answer_quality):
        """Test handling of errors during evaluation."""
        with patch.object(answer_quality, "_call_llm", side_effect=Exception("API Error")):
            result = answer_quality._evaluate_answer(
                question="Test?",
                model_answer="Test answer",
                gold_answer="Gold answer",
            )

            assert result["completeness_score"] == 0
            assert result["correctness_score"] == 0
            assert result["overall_score"] == 0.0
            assert "Error" in result["completeness_reasoning"]

    def test_score_clamping(self, answer_quality):
        """Test that scores are clamped to 1-5 range."""
        mock_llm_result = {
            "completeness_score": 10,  # Too high
            "correctness_score": 0,  # Too low
            "completeness_reasoning": "Test",
            "correctness_reasoning": "Test",
            "key_missing_info": [],
            "factual_errors": [],
        }

        with patch.object(answer_quality, "_call_llm", return_value=mock_llm_result):
            result = answer_quality._evaluate_answer(
                question="Test?",
                model_answer="Test",
                gold_answer="Gold",
            )

            assert result["completeness_score"] == 5  # Clamped to max
            assert result["correctness_score"] == 1  # Clamped to min


class TestCompute:
    """Test compute method."""

    def test_compute_empty_list(self, answer_quality):
        """Test compute with empty list."""
        results = answer_quality.compute([])

        assert len(results) == 1
        assert results[0]["metric_name"] == "answer_quality"
        assert results[0]["score"] == 0.0
        assert results[0]["metadata"]["num_pairs"] == 0
        assert results[0]["metadata"]["avg_completeness"] == 0.0
        assert results[0]["metadata"]["avg_correctness"] == 0.0

    def test_compute_single_pair(self, answer_quality):
        """Test compute with single Q&A pair."""
        qa_pairs = [
            {
                "question": "What is the capital of France?",
                "answer": "The capital of France is Paris.",
                "gold_answer": "Paris is the capital of France.",
            }
        ]

        mock_eval_result = {
            "completeness_score": 5,
            "correctness_score": 5,
            "overall_score": 5.0,
            "completeness_reasoning": "Complete",
            "correctness_reasoning": "Correct",
            "key_missing_info": [],
            "factual_errors": [],
        }

        with patch.object(answer_quality, "_evaluate_answer", return_value=mock_eval_result):
            results = answer_quality.compute(qa_pairs)

            assert len(results) == 1
            assert results[0]["metric_name"] == "answer_quality"
            assert results[0]["score"] == 5.0
            assert results[0]["metadata"]["num_pairs"] == 1
            assert results[0]["metadata"]["avg_completeness"] == 5.0
            assert results[0]["metadata"]["avg_correctness"] == 5.0
            assert len(results[0]["individual_scores"]) == 1

    def test_compute_multiple_pairs(self, answer_quality):
        """Test compute with multiple Q&A pairs."""
        qa_pairs = [
            {
                "question": "Q1?",
                "answer": "A1",
                "gold_answer": "G1",
            },
            {
                "question": "Q2?",
                "answer": "A2",
                "gold_answer": "G2",
            },
            {
                "question": "Q3?",
                "answer": "A3",
                "gold_answer": "G3",
            },
        ]

        mock_results = [
            {
                "completeness_score": 5,
                "correctness_score": 5,
                "overall_score": 5.0,
                "completeness_reasoning": "Complete",
                "correctness_reasoning": "Correct",
                "key_missing_info": [],
                "factual_errors": [],
            },
            {
                "completeness_score": 3,
                "correctness_score": 4,
                "overall_score": 3.5,
                "completeness_reasoning": "Partial",
                "correctness_reasoning": "Mostly correct",
                "key_missing_info": ["Some detail"],
                "factual_errors": [],
            },
            {
                "completeness_score": 2,
                "correctness_score": 3,
                "overall_score": 2.5,
                "completeness_reasoning": "Minimal",
                "correctness_reasoning": "Some errors",
                "key_missing_info": ["Many details"],
                "factual_errors": ["One error"],
            },
        ]

        with patch.object(answer_quality, "_evaluate_answer", side_effect=mock_results):
            results = answer_quality.compute(qa_pairs)

            assert len(results) == 1
            assert results[0]["metric_name"] == "answer_quality"

            # Average: (5+3+2)/3 = 3.33, (5+4+3)/3 = 4.0, overall = (5+3.5+2.5)/3 = 3.67
            assert results[0]["metadata"]["avg_completeness"] == pytest.approx(3.33, abs=0.01)
            assert results[0]["metadata"]["avg_correctness"] == pytest.approx(4.0)
            assert results[0]["score"] == pytest.approx(3.67, abs=0.01)
            assert results[0]["metadata"]["num_pairs"] == 3
            assert len(results[0]["individual_scores"]) == 3

    def test_compute_with_error_handling(self, answer_quality):
        """Test compute handles errors gracefully."""
        qa_pairs = [
            {
                "question": "Q1?",
                "answer": "A1",
                "gold_answer": "G1",
            },
            {
                "question": "Q2?",
                "answer": "A2",
                "gold_answer": "G2",
            },
        ]

        def mock_eval(question, _model_answer, _gold_answer):
            if question == "Q1?":
                return {
                    "completeness_score": 5,
                    "correctness_score": 5,
                    "overall_score": 5.0,
                    "completeness_reasoning": "Good",
                    "correctness_reasoning": "Good",
                    "key_missing_info": [],
                    "factual_errors": [],
                }
            else:
                raise Exception("Test error")

        with patch.object(answer_quality, "_evaluate_answer", side_effect=mock_eval):
            results = answer_quality.compute(qa_pairs)

            assert len(results) == 1
            assert results[0]["metadata"]["num_pairs"] == 2
            # First pair: 5.0, second pair: 0.0 (error), average: 2.5
            assert results[0]["score"] == 2.5
            assert len(results[0]["individual_scores"]) == 2
            assert "error" in results[0]["individual_scores"][1]

    def test_compute_metadata(self, answer_quality):
        """Test that compute includes correct metadata."""
        qa_pairs = [
            {
                "question": "Q?",
                "answer": "A",
                "gold_answer": "G",
            }
        ]

        mock_eval_result = {
            "completeness_score": 4,
            "correctness_score": 5,
            "overall_score": 4.5,
            "completeness_reasoning": "Good",
            "correctness_reasoning": "Great",
            "key_missing_info": [],
            "factual_errors": [],
        }

        with patch.object(answer_quality, "_evaluate_answer", return_value=mock_eval_result):
            results = answer_quality.compute(qa_pairs)

            metadata = results[0]["metadata"]
            assert metadata["model"] == "anthropic/claude-3.5-sonnet"
            assert metadata["num_pairs"] == 1
            assert "config" in metadata
