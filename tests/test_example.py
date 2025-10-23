"""Example test file to verify pytest setup."""

import retrieval_evals


def test_version() -> None:
    """Test that version is defined."""
    assert retrieval_evals.__version__ == "0.1.0"


def test_example() -> None:
    """Example test case."""
    assert 1 + 1 == 2
