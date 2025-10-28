"""Tests for readability metric."""

from retrieval_evals.evals.ungrounded.metrics.readability import Readability
from retrieval_evals.types import QAPair


class TestReadabilityInit:
    """Tests for Readability initialization."""

    def test_init(self) -> None:
        """Test initialization."""
        metric = Readability()
        assert metric.name == "readability"

    def test_name_property(self) -> None:
        """Test that name property returns correct value."""
        metric = Readability()
        assert metric.name == "readability"


class TestSyllableCount:
    """Tests for syllable counting."""

    def test_count_syllables_simple(self) -> None:
        """Test syllable counting for simple words."""
        metric = Readability()

        assert metric._count_syllables("cat") == 1
        assert metric._count_syllables("dog") == 1
        assert metric._count_syllables("hello") == 2
        assert metric._count_syllables("computer") == 3
        assert metric._count_syllables("elementary") == 5

    def test_count_syllables_silent_e(self) -> None:
        """Test syllable counting handles silent e."""
        metric = Readability()

        assert metric._count_syllables("make") == 1
        assert metric._count_syllables("take") == 1
        assert metric._count_syllables("complete") == 2

    def test_count_syllables_short_words(self) -> None:
        """Test that short words return at least 1 syllable."""
        metric = Readability()

        assert metric._count_syllables("a") == 1
        assert metric._count_syllables("I") == 1
        assert metric._count_syllables("be") == 1


class TestTextSplitting:
    """Tests for text splitting functions."""

    def test_split_sentences(self) -> None:
        """Test sentence splitting."""
        metric = Readability()

        text = "This is sentence one. This is sentence two! And sentence three?"
        sentences = metric._split_sentences(text)

        assert len(sentences) == 3
        assert "This is sentence one" in sentences[0]

    def test_split_sentences_empty(self) -> None:
        """Test sentence splitting with empty text."""
        metric = Readability()

        sentences = metric._split_sentences("")
        assert len(sentences) == 0

    def test_split_words(self) -> None:
        """Test word splitting."""
        metric = Readability()

        text = "Hello, world! This is a test."
        words = metric._split_words(text)

        assert len(words) == 6
        assert "Hello" in words
        assert "world" in words

    def test_split_words_with_punctuation(self) -> None:
        """Test that word splitting removes punctuation."""
        metric = Readability()

        text = "can't, don't, won't"
        words = metric._split_words(text)

        # Should split contractions and remove punctuation
        assert all(word.isalpha() for word in words)


class TestReadabilityFormulas:
    """Tests for readability formulas."""

    def test_flesch_reading_ease_simple(self) -> None:
        """Test Flesch Reading Ease for simple text."""
        metric = Readability()

        # Short sentences, simple words = high score
        score = metric._compute_flesch_reading_ease(
            total_words=10,
            total_sentences=2,
            total_syllables=12,  # Simple words
        )

        assert 60.0 < score <= 100.0  # Should be fairly easy

    def test_flesch_reading_ease_complex(self) -> None:
        """Test Flesch Reading Ease for complex text."""
        metric = Readability()

        # Long sentences, complex words = low score
        score = metric._compute_flesch_reading_ease(
            total_words=50,
            total_sentences=1,  # One very long sentence
            total_syllables=100,  # Complex words
        )

        assert 0.0 <= score < 40.0  # Should be difficult

    def test_flesch_kincaid_grade(self) -> None:
        """Test Flesch-Kincaid Grade Level."""
        metric = Readability()

        grade = metric._compute_flesch_kincaid_grade(
            total_words=20,
            total_sentences=2,
            total_syllables=25,
        )

        assert 0.0 <= grade <= 20.0  # Reasonable grade level

    def test_gunning_fog(self) -> None:
        """Test Gunning Fog Index."""
        metric = Readability()

        fog = metric._compute_gunning_fog(
            total_words=20,
            total_sentences=2,
            complex_words=5,
        )

        assert 0.0 <= fog <= 20.0

    def test_formulas_with_zero_input(self) -> None:
        """Test that formulas handle zero input gracefully."""
        metric = Readability()

        assert metric._compute_flesch_reading_ease(0, 0, 0) == 0.0
        assert metric._compute_flesch_kincaid_grade(0, 0, 0) == 0.0
        assert metric._compute_gunning_fog(0, 0, 0) == 0.0


class TestTextAnalysis:
    """Tests for text analysis."""

    def test_analyze_simple_text(self) -> None:
        """Test analysis of simple text."""
        metric = Readability()

        text = "The cat sat on the mat. The dog ran in the park."
        analysis = metric._analyze_text(text)

        assert "flesch_reading_ease" in analysis
        assert "flesch_kincaid_grade" in analysis
        assert "gunning_fog" in analysis
        assert "smog_index" in analysis
        assert "coleman_liau" in analysis
        assert "ari" in analysis
        assert analysis["avg_sentence_length"] > 0
        assert analysis["avg_word_length"] > 0

    def test_analyze_empty_text(self) -> None:
        """Test analysis of empty text."""
        metric = Readability()

        analysis = metric._analyze_text("")

        assert analysis["flesch_reading_ease"] == 0.0
        assert analysis["avg_sentence_length"] == 0.0

    def test_analyze_single_sentence(self) -> None:
        """Test analysis of single sentence."""
        metric = Readability()

        text = "This is a simple test sentence."
        analysis = metric._analyze_text(text)

        assert analysis["flesch_reading_ease"] > 0.0
        assert analysis["avg_sentence_length"] > 0.0


class TestCompute:
    """Tests for compute method."""

    def test_compute_empty_pairs(self) -> None:
        """Test compute with empty QA pairs."""
        metric = Readability()
        qa_pairs: list[QAPair] = []

        results = metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "readability"
        assert results[0]["score"] == 0.0
        assert results[0]["individual_scores"] == []

    def test_compute_single_pair(self) -> None:
        """Test compute with single QA pair."""
        metric = Readability()
        qa_pairs: list[QAPair] = [
            {
                "question": "What is a cat?",
                "answer": "A cat is a small pet. Cats are furry. They like to play.",
            }
        ]

        results = metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "readability"
        assert 0.0 <= results[0]["score"] <= 100.0
        assert len(results[0]["individual_scores"]) == 1

        individual = results[0]["individual_scores"][0]
        assert "flesch_reading_ease" in individual
        assert "gunning_fog" in individual

    def test_compute_multiple_pairs(self) -> None:
        """Test compute with multiple QA pairs."""
        metric = Readability()
        qa_pairs: list[QAPair] = [
            {
                "question": "Q1?",
                "answer": "Simple answer one.",
            },
            {
                "question": "Q2?",
                "answer": "Simple answer two.",
            },
        ]

        results = metric.compute(qa_pairs)

        assert len(results[0]["individual_scores"]) == 2
        assert results[0]["metadata"]["num_pairs"] == 2

    def test_compute_complex_vs_simple(self) -> None:
        """Test that complex text scores lower than simple text."""
        metric = Readability()

        simple_pairs: list[QAPair] = [
            {
                "question": "Q?",
                "answer": "The cat ran. The dog jumped. They played.",
            }
        ]

        complex_pairs: list[QAPair] = [
            {
                "question": "Q?",
                "answer": (
                    "The implementation of comprehensive methodological frameworks "
                    "necessitates the utilization of sophisticated algorithmic "
                    "constructs to facilitate multifaceted computational analyses."
                ),
            }
        ]

        simple_result = metric.compute(simple_pairs)[0]["score"]
        complex_result = metric.compute(complex_pairs)[0]["score"]

        # Simple text should have higher Flesch Reading Ease
        assert simple_result > complex_result

    def test_metadata_includes_interpretation(self) -> None:
        """Test that metadata includes score interpretation."""
        metric = Readability()
        qa_pairs: list[QAPair] = [{"question": "Q?", "answer": "Simple answer."}]

        results = metric.compute(qa_pairs)

        assert "score_interpretation" in results[0]["metadata"]
        interpretation = results[0]["metadata"]["score_interpretation"]
        assert "90-100" in interpretation
        assert "0-29" in interpretation
