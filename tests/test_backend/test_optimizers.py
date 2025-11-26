"""Tests for optimizer utilities."""

from __future__ import annotations

import pytest

from backend.optimizers import (
    split_examples,
    get_num_threads,
    GEPA_NUM_THREADS_LOCAL,
    GEPA_NUM_THREADS_REMOTE_MAX,
    GEPA_VALSET_MIN,
    GEPA_VALSET_MAX,
)


class TestSplitExamples:
    """Tests for split_examples function."""

    def test_minimum_examples_raises(self):
        """Need at least GEPA_VALSET_MIN + 1 examples."""
        with pytest.raises(ValueError, match="Need at least"):
            split_examples([1])

    def test_two_examples_splits_correctly(self):
        """Two examples: 1 train, 1 val."""
        train, val = split_examples([1, 2])
        assert train == [1]
        assert val == [2]

    def test_three_examples_splits_correctly(self):
        """Three examples: 2 train, 1 val."""
        train, val = split_examples([1, 2, 3])
        assert train == [1, 2]
        assert val == [3]

    def test_ten_examples_gives_one_val(self):
        """10 examples: 9 train, 1 val (10% baseline = 1)."""
        examples = list(range(10))
        train, val = split_examples(examples)
        assert len(train) == 9
        assert len(val) == 1
        assert val == [9]

    def test_thirty_examples_gives_three_val(self):
        """30 examples: 27 train, 3 val (10% = 3, capped at GEPA_VALSET_MAX)."""
        examples = list(range(30))
        train, val = split_examples(examples)
        assert len(train) == 27
        assert len(val) == 3
        assert val == [27, 28, 29]

    def test_hundred_examples_caps_at_max(self):
        """100 examples: 97 train, 3 val (10% = 10, but capped at GEPA_VALSET_MAX)."""
        examples = list(range(100))
        train, val = split_examples(examples)
        assert len(train) == 100 - GEPA_VALSET_MAX
        assert len(val) == GEPA_VALSET_MAX

    def test_preserves_order(self):
        """Train gets first N, val gets last M."""
        examples = ["a", "b", "c", "d", "e"]
        train, val = split_examples(examples)
        assert train == ["a", "b", "c", "d"]
        assert val == ["e"]

    def test_valset_bounds(self):
        """Valset size is always between GEPA_VALSET_MIN and GEPA_VALSET_MAX."""
        for n in range(2, 200):
            train, val = split_examples(list(range(n)))
            assert GEPA_VALSET_MIN <= len(val) <= GEPA_VALSET_MAX
            assert len(train) + len(val) == n


class TestGetNumThreads:
    """Tests for get_num_threads function."""

    def test_local_always_returns_one(self):
        """Local mode always returns GEPA_NUM_THREADS_LOCAL (no parallelism)."""
        assert get_num_threads(1, remote=False) == GEPA_NUM_THREADS_LOCAL
        assert get_num_threads(10, remote=False) == GEPA_NUM_THREADS_LOCAL
        assert get_num_threads(100, remote=False) == GEPA_NUM_THREADS_LOCAL

    def test_remote_scales_with_examples(self):
        """Remote mode scales threads with example count."""
        assert get_num_threads(5, remote=True) == 5
        assert get_num_threads(10, remote=True) == 10
        assert get_num_threads(15, remote=True) == 15

    def test_remote_caps_at_max(self):
        """Remote mode caps at GEPA_NUM_THREADS_REMOTE_MAX."""
        assert get_num_threads(50, remote=True) == GEPA_NUM_THREADS_REMOTE_MAX
        assert get_num_threads(100, remote=True) == GEPA_NUM_THREADS_REMOTE_MAX

    def test_remote_default_is_false(self):
        """Default remote=False returns local thread count."""
        assert get_num_threads(100) == GEPA_NUM_THREADS_LOCAL
