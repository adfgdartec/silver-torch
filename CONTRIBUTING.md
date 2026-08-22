# Contributing to silver-torch

Thank you for helping make inspectable model training easier to trust.

1. Create a focused branch and include tests for behavior changes.
2. Install development dependencies with `python -m pip install -e '.[dev,pytorch]'`.
3. Run `ruff check .` and `pytest -q` before opening a pull request.
4. Explain the user problem, API impact, and verification in the pull request.

Keep the core API small, deterministic, and explicit. New runtime dependencies
need a clear benefit and should remain optional when practical.

