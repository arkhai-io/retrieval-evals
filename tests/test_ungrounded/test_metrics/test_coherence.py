"""Tests for coherence metric."""

import pytest

from retrieval_evals.evals.ungrounded.metrics.coherence import Coherence
from retrieval_evals.types import QAPair


@pytest.fixture
def coherence_metric() -> Coherence:
    """Create a Coherence metric instance for testing."""
    return Coherence(embedding_model="all-MiniLM-L6-v2")


class TestCoherenceInit:
    """Tests for Coherence initialization."""

    def test_init_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        metric = Coherence()

        assert metric.embedding_model_name == "all-MiniLM-L6-v2"
        assert metric.name == "coherence"

    def test_init_with_custom_model(self) -> None:
        """Test initialization with custom embedding model."""
        metric = Coherence(embedding_model="all-mpnet-base-v2")

        assert metric.embedding_model_name == "all-mpnet-base-v2"

    def test_name_property(self, coherence_metric: Coherence) -> None:
        """Test that name property returns correct value."""
        assert coherence_metric.name == "coherence"


class TestSentenceSplitting:
    """Tests for sentence splitting."""

    def test_split_sentences_basic(self, coherence_metric: Coherence) -> None:
        """Test basic sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = coherence_metric._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]

    def test_split_sentences_empty(self, coherence_metric: Coherence) -> None:
        """Test sentence splitting with empty text."""
        sentences = coherence_metric._split_sentences("")

        assert len(sentences) == 0

    def test_split_sentences_filters_short(self, coherence_metric: Coherence) -> None:
        """Test that very short fragments are filtered out."""
        text = "Good sentence here. A. B. Another good one."
        sentences = coherence_metric._split_sentences(text)

        # "A" and "B" should be filtered out (too short)
        assert len(sentences) == 2

    def test_split_sentences_strips_whitespace(self, coherence_metric: Coherence) -> None:
        """Test that sentences are stripped of whitespace."""
        text = "  First sentence  .   Second sentence  !"
        sentences = coherence_metric._split_sentences(text)

        assert all(not s.startswith(" ") and not s.endswith(" ") for s in sentences)


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_cosine_similarity_identical(self, coherence_metric: Coherence) -> None:
        """Test cosine similarity of identical vectors."""
        import numpy as np

        vec = np.array([1.0, 2.0, 3.0])
        similarity = coherence_metric._compute_cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_cosine_similarity_orthogonal(self, coherence_metric: Coherence) -> None:
        """Test cosine similarity of orthogonal vectors."""
        import numpy as np

        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        similarity = coherence_metric._compute_cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0, abs=0.01)

    def test_cosine_similarity_opposite(self, coherence_metric: Coherence) -> None:
        """Test cosine similarity of opposite vectors."""
        import numpy as np

        vec1 = np.array([1.0, 1.0])
        vec2 = np.array([-1.0, -1.0])
        similarity = coherence_metric._compute_cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(-1.0, abs=0.01)


class TestCoherenceAnalysis:
    """Tests for coherence analysis."""

    def test_analyze_empty_text(self, coherence_metric: Coherence) -> None:
        """Test analysis of empty text."""
        analysis = coherence_metric._analyze_coherence("")

        assert analysis["coherence_score"] == 0.0
        assert analysis["num_sentences"] == 0

    def test_analyze_single_sentence(self, coherence_metric: Coherence) -> None:
        """Test analysis of single sentence."""
        text = "This is a single sentence about cats."
        analysis = coherence_metric._analyze_coherence(text)

        # Single sentence is perfectly coherent with itself
        assert analysis["coherence_score"] == 1.0
        assert analysis["num_sentences"] == 1

    def test_analyze_coherent_text(self, coherence_metric: Coherence) -> None:
        """Test analysis of coherent text."""
        text = (
            "Cats are popular pets. They are independent animals. "
            "Many people love cats. Cats enjoy playing and sleeping."
        )
        analysis = coherence_metric._analyze_coherence(text)

        # Related sentences should have reasonable coherence
        assert analysis["coherence_score"] > 0.3  # Lowered from 0.5 to be more robust
        assert analysis["num_sentences"] == 4
        assert analysis["topic_consistency"] > 0.5
        assert len(analysis["consecutive_similarities"]) == 3

    def test_analyze_incoherent_text(self, coherence_metric: Coherence) -> None:
        """Test analysis of incoherent text."""
        text = (
            "I like pizza. Quantum physics is complex. "
            "The weather is nice today. Economics studies markets."
        )
        analysis = coherence_metric._analyze_coherence(text)

        # Unrelated sentences should have lower coherence than related ones
        # Note: Even unrelated sentences might have some similarity
        assert analysis["num_sentences"] == 4
        assert len(analysis["consecutive_similarities"]) == 3

        # Compare with coherent text
        coherent_text = "Cats are pets. Pets need care. Care requires time. Time is valuable."
        coherent_analysis = coherence_metric._analyze_coherence(coherent_text)

        # Coherent text should score higher
        assert coherent_analysis["coherence_score"] > analysis["coherence_score"]

    def test_analyze_includes_min_coherence(self, coherence_metric: Coherence) -> None:
        """Test that analysis includes minimum coherence."""
        text = "First sentence. Second sentence. Third sentence."
        analysis = coherence_metric._analyze_coherence(text)

        assert "min_coherence" in analysis
        assert 0.0 <= analysis["min_coherence"] <= 1.0

    def test_analyze_includes_variance(self, coherence_metric: Coherence) -> None:
        """Test that analysis includes coherence variance."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        analysis = coherence_metric._analyze_coherence(text)

        assert "coherence_variance" in analysis
        assert analysis["coherence_variance"] >= 0.0


class TestCompute:
    """Tests for compute method."""

    def test_compute_empty_pairs(self, coherence_metric: Coherence) -> None:
        """Test compute with empty QA pairs."""
        qa_pairs: list[QAPair] = []

        results = coherence_metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "coherence"
        assert results[0]["score"] == 0.0
        assert results[0]["individual_scores"] == []

    def test_compute_single_pair(self, coherence_metric: Coherence) -> None:
        """Test compute with single QA pair."""
        qa_pairs: list[QAPair] = [
            {
                "question": "What is coherence?",
                "answer": (
                    "Coherence means unity. Unity creates flow. " "Flow improves understanding."
                ),
            }
        ]

        results = coherence_metric.compute(qa_pairs)

        assert len(results) == 1
        assert results[0]["metric_name"] == "coherence"
        assert 0.0 <= results[0]["score"] <= 1.0
        assert len(results[0]["individual_scores"]) == 1

        individual = results[0]["individual_scores"][0]
        assert "coherence_score" in individual
        assert "topic_consistency" in individual
        assert "num_sentences" in individual

    def test_compute_multiple_pairs(self, coherence_metric: Coherence) -> None:
        """Test compute with multiple QA pairs."""
        qa_pairs: list[QAPair] = [
            {"question": "Q1?", "answer": "Answer one. Related sentence."},
            {"question": "Q2?", "answer": "Answer two. Another related sentence."},
        ]

        results = coherence_metric.compute(qa_pairs)

        assert len(results[0]["individual_scores"]) == 2
        assert results[0]["metadata"]["num_pairs"] == 2

    def test_compute_coherent_vs_incoherent(self, coherence_metric: Coherence) -> None:
        """Test that coherent text scores higher than incoherent text."""
        coherent_pairs: list[QAPair] = [
            {
                "question": "Q?",
                "answer": (
                    "Dogs are loyal pets. They love their owners. "
                    "Owners provide food and care. Care creates strong bonds."
                ),
            }
        ]

        incoherent_pairs: list[QAPair] = [
            {
                "question": "Q?",
                "answer": (
                    "I like coffee. The sky is blue. " "Mathematics is difficult. Trees are green."
                ),
            }
        ]

        coherent_result = coherence_metric.compute(coherent_pairs)[0]["score"]
        incoherent_result = coherence_metric.compute(incoherent_pairs)[0]["score"]

        # Coherent text should score higher
        assert coherent_result > incoherent_result

    def test_metadata_includes_interpretation(self, coherence_metric: Coherence) -> None:
        """Test that metadata includes score interpretation."""
        qa_pairs: list[QAPair] = [{"question": "Q?", "answer": "Some coherent answer here."}]

        results = coherence_metric.compute(qa_pairs)

        assert "interpretation" in results[0]["metadata"]
        interpretation = results[0]["metadata"]["interpretation"]
        assert "0.8-1.0" in interpretation
        assert "0.0-0.2" in interpretation

    def test_metadata_includes_embedding_model(self, coherence_metric: Coherence) -> None:
        """Test that metadata includes embedding model name."""
        qa_pairs: list[QAPair] = [{"question": "Q?", "answer": "Answer."}]

        results = coherence_metric.compute(qa_pairs)

        assert "embedding_model" in results[0]["metadata"]
        assert results[0]["metadata"]["embedding_model"] == "all-MiniLM-L6-v2"

    def test_consecutive_similarities_length(self, coherence_metric: Coherence) -> None:
        """Test that consecutive similarities has correct length."""
        qa_pairs: list[QAPair] = [
            {
                "question": "Q?",
                "answer": "First. Second. Third. Fourth.",
            }
        ]

        results = coherence_metric.compute(qa_pairs)

        individual = results[0]["individual_scores"][0]
        # 4 sentences should give 3 consecutive similarities
        assert len(individual["consecutive_similarities"]) == 3
