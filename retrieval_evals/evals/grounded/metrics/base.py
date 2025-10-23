"""Base class for grounded metrics."""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from retrieval_evals.types import EvalResult, QAPairWithGold


class Metric(ABC):
    """Base class for all grounded metrics."""

    def __init__(self, **kwargs: Any):
        """Initialize metric with config.

        Args:
            **kwargs: Configuration parameters for the metric
        """
        self.config = kwargs

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the metric name."""
        pass

    @abstractmethod
    def compute(self, qa_pairs: list[QAPairWithGold]) -> list[EvalResult]:
        """Compute metric scores for Q&A pairs.

        Args:
            qa_pairs: List of Q&A pairs with gold standard answers

        Returns:
            List of evaluation results
        """
        pass

    def load_data(self, json_path: Path | str) -> list[QAPairWithGold]:
        """Load Q&A pairs with gold standard answers from JSON file.

        Args:
            json_path: Path to JSON file containing Q&A pairs with gold answers

        Returns:
            List of Q&A pairs with gold standard answers

        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            ValueError: If file is not a JSON file or contains invalid JSON
        """
        # Convert to Path object for consistent handling
        path = Path(json_path)

        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        # Check if it's a file (not a directory)
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        # Check file extension
        if path.suffix.lower() != ".json":
            raise ValueError(f"File must be a JSON file, got: {path.suffix}")

        # Load and validate JSON syntax
        try:
            with open(path) as f:
                data: list[QAPairWithGold] = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {path}: {e}") from e

        return data

    def evaluate(self, data: Path | str | list[QAPairWithGold]) -> list[EvalResult]:
        """Run evaluation on data.

        Args:
            data: Either a path to JSON file OR list of Q&A pairs

        Returns:
            List of evaluation results
        """
        # Load data if it's a file path
        qa_pairs = self.load_data(data) if isinstance(data, Path | str) else data

        # Compute the metric
        return self.compute(qa_pairs)
