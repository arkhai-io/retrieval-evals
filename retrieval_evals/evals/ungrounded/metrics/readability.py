"""Readability analysis metric using multiple readability formulas.

Evaluates text readability using established formulas:
- Flesch Reading Ease: 0-100 scale (higher = easier)
- Flesch-Kincaid Grade Level: US grade level
- Gunning Fog Index: Years of education needed
- SMOG Index: Simple Measure of Gobbledygook
- Coleman-Liau Index: Based on characters vs. words
- Automated Readability Index (ARI)

Also provides basic text statistics:
- Average sentence length
- Average word length
- Syllables per word

Lower grade levels and higher reading ease indicate more accessible text.
"""

import re
from statistics import mean

from retrieval_evals.evals.ungrounded.metrics.base import Metric
from retrieval_evals.types import EvalResult, QAPair


class Readability(Metric):
    """Readability metric using multiple readability formulas.

    Computes various readability scores to assess how easy text is to read
    and what education level is required to understand it.
    """

    def __init__(self) -> None:
        """Initialize readability metric."""
        super().__init__()

    @property
    def name(self) -> str:
        """Return metric name."""
        return "readability"

    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word using simple heuristic.

        Args:
            word: Word to count syllables in

        Returns:
            Estimated syllable count
        """
        word = word.lower()
        # Remove non-alphabetic characters
        word = re.sub(r"[^a-z]", "", word)

        if len(word) <= 3:
            return 1

        # Count vowel groups
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent e
        if word.endswith("e"):
            syllable_count -= 1

        # Ensure at least one syllable
        return max(1, syllable_count)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting on . ! ?
        sentences = re.split(r"[.!?]+", text)
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _split_words(self, text: str) -> list[str]:
        """Split text into words.

        Args:
            text: Text to split

        Returns:
            List of words
        """
        # Split on whitespace and remove punctuation
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        return words

    def _compute_flesch_reading_ease(
        self, total_words: int, total_sentences: int, total_syllables: int
    ) -> float:
        """Compute Flesch Reading Ease score.

        Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        Score: 0-100 (higher = easier to read)

        Args:
            total_words: Total word count
            total_sentences: Total sentence count
            total_syllables: Total syllable count

        Returns:
            Flesch Reading Ease score
        """
        if total_words == 0 or total_sentences == 0:
            return 0.0

        avg_words_per_sentence = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words

        score = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        return max(0.0, min(100.0, score))  # Clamp to 0-100

    def _compute_flesch_kincaid_grade(
        self, total_words: int, total_sentences: int, total_syllables: int
    ) -> float:
        """Compute Flesch-Kincaid Grade Level.

        Formula: 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59

        Args:
            total_words: Total word count
            total_sentences: Total sentence count
            total_syllables: Total syllable count

        Returns:
            Grade level (US school grade)
        """
        if total_words == 0 or total_sentences == 0:
            return 0.0

        avg_words_per_sentence = total_words / total_sentences
        avg_syllables_per_word = total_syllables / total_words

        grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
        return max(0.0, grade)

    def _compute_gunning_fog(
        self, total_words: int, total_sentences: int, complex_words: int
    ) -> float:
        """Compute Gunning Fog Index.

        Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
        Complex words = 3+ syllables

        Args:
            total_words: Total word count
            total_sentences: Total sentence count
            complex_words: Count of words with 3+ syllables

        Returns:
            Gunning Fog Index (years of education)
        """
        if total_words == 0 or total_sentences == 0:
            return 0.0

        avg_words_per_sentence = total_words / total_sentences
        percent_complex = 100 * (complex_words / total_words)

        fog = 0.4 * (avg_words_per_sentence + percent_complex)
        return max(0.0, fog)

    def _compute_smog_index(self, total_sentences: int, complex_words: int) -> float:
        """Compute SMOG Index.

        Formula: 1.0430 * sqrt(complex_words * (30/sentences)) + 3.1291

        Args:
            total_sentences: Total sentence count
            complex_words: Count of words with 3+ syllables

        Returns:
            SMOG Index (grade level)
        """
        if total_sentences == 0:
            return 0.0

        import math

        smog = 1.0430 * math.sqrt(complex_words * (30 / total_sentences)) + 3.1291
        return max(0.0, smog)

    def _compute_coleman_liau(
        self, total_words: int, total_sentences: int, total_chars: int
    ) -> float:
        """Compute Coleman-Liau Index.

        Formula: 0.0588 * L - 0.296 * S - 15.8
        L = average letters per 100 words
        S = average sentences per 100 words

        Args:
            total_words: Total word count
            total_sentences: Total sentence count
            total_chars: Total character count

        Returns:
            Coleman-Liau Index (grade level)
        """
        if total_words == 0:
            return 0.0

        L = (total_chars / total_words) * 100
        S = (total_sentences / total_words) * 100

        cli = 0.0588 * L - 0.296 * S - 15.8
        return max(0.0, cli)

    def _compute_ari(self, total_words: int, total_sentences: int, total_chars: int) -> float:
        """Compute Automated Readability Index.

        Formula: 4.71 * (chars/words) + 0.5 * (words/sentences) - 21.43

        Args:
            total_words: Total word count
            total_sentences: Total sentence count
            total_chars: Total character count

        Returns:
            ARI (grade level)
        """
        if total_words == 0 or total_sentences == 0:
            return 0.0

        avg_chars_per_word = total_chars / total_words
        avg_words_per_sentence = total_words / total_sentences

        ari = 4.71 * avg_chars_per_word + 0.5 * avg_words_per_sentence - 21.43
        return max(0.0, ari)

    def _analyze_text(self, text: str) -> dict[str, float]:
        """Analyze text and compute all readability metrics.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with readability scores
        """
        if not text.strip():
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "gunning_fog": 0.0,
                "smog_index": 0.0,
                "coleman_liau": 0.0,
                "ari": 0.0,
                "avg_sentence_length": 0.0,
                "avg_word_length": 0.0,
                "avg_syllables_per_word": 0.0,
            }

        sentences = self._split_sentences(text)
        words = self._split_words(text)

        if not sentences or not words:
            return {
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "gunning_fog": 0.0,
                "smog_index": 0.0,
                "coleman_liau": 0.0,
                "ari": 0.0,
                "avg_sentence_length": 0.0,
                "avg_word_length": 0.0,
                "avg_syllables_per_word": 0.0,
            }

        # Compute basic statistics
        total_sentences = len(sentences)
        total_words = len(words)
        total_chars = sum(len(word) for word in words)
        total_syllables = sum(self._count_syllables(word) for word in words)
        complex_words = sum(1 for word in words if self._count_syllables(word) >= 3)

        return {
            "flesch_reading_ease": self._compute_flesch_reading_ease(
                total_words, total_sentences, total_syllables
            ),
            "flesch_kincaid_grade": self._compute_flesch_kincaid_grade(
                total_words, total_sentences, total_syllables
            ),
            "gunning_fog": self._compute_gunning_fog(total_words, total_sentences, complex_words),
            "smog_index": self._compute_smog_index(total_sentences, complex_words),
            "coleman_liau": self._compute_coleman_liau(total_words, total_sentences, total_chars),
            "ari": self._compute_ari(total_words, total_sentences, total_chars),
            "avg_sentence_length": total_words / total_sentences,
            "avg_word_length": total_chars / total_words,
            "avg_syllables_per_word": total_syllables / total_words,
        }

    def compute(self, qa_pairs: list[QAPair]) -> list[EvalResult]:
        """Compute readability scores for Q&A pairs.

        Args:
            qa_pairs: List of Q&A pairs

        Returns:
            List with single evaluation result
        """
        if not qa_pairs:
            return [
                {
                    "metric_name": self.name,
                    "score": 0.0,
                    "individual_scores": [],
                    "metadata": {"num_pairs": 0, "config": self.config},
                }
            ]

        individual_scores = []
        flesch_scores = []

        for pair in qa_pairs:
            analysis = self._analyze_text(pair["answer"])
            individual_scores.append(analysis)
            flesch_scores.append(analysis["flesch_reading_ease"])

        # Use average Flesch Reading Ease as overall score
        avg_flesch = mean(flesch_scores) if flesch_scores else 0.0

        return [
            {
                "metric_name": self.name,
                "score": avg_flesch,
                "individual_scores": individual_scores,
                "metadata": {
                    "num_pairs": len(qa_pairs),
                    "config": self.config,
                    "score_interpretation": {
                        "90-100": "Very Easy (5th grade)",
                        "80-89": "Easy (6th grade)",
                        "70-79": "Fairly Easy (7th grade)",
                        "60-69": "Standard (8th-9th grade)",
                        "50-59": "Fairly Difficult (10th-12th grade)",
                        "30-49": "Difficult (College)",
                        "0-29": "Very Difficult (College graduate)",
                    },
                },
            }
        ]
