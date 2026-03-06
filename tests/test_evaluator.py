"""Tests for the evaluator module."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import exact_match, numeric_match


def test_exact_match():
    assert exact_match("hello", "hello") is True
    assert exact_match("Hello", "hello") is True
    assert exact_match("  hello  ", "hello") is True
    assert exact_match("hello", "world") is False


def test_numeric_match():
    assert numeric_match("42", "42") is True
    assert numeric_match("The answer is 42", "42") is True
    assert numeric_match("#### 42", "#### 42") is True
    assert numeric_match("43", "42") is False


def test_numeric_match_decimals():
    assert numeric_match("3.14", "3.14") is True
    assert numeric_match("3.14159", "3.14") is False


def test_numeric_match_with_commas():
    assert numeric_match("1,000", "1000") is True
    assert numeric_match("#### 1,234", "1234") is True


if __name__ == "__main__":
    test_exact_match()
    test_numeric_match()
    test_numeric_match_decimals()
    test_numeric_match_with_commas()
    print("All evaluator tests passed!")
