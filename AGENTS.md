# ASK ME QUESTIONS

Unless I request a simple change, any discussion of architecture, design, feature building, debugging, etc. should include clarifying questions asked ONE AT A TIME so we may align our understanding of the task & high level goals. Prompt me back.

# Testing Textual Applications

**CRITICAL:** Never run the Textual app directly (e.g., `python charlie.py`) in a non-interactive shell or agent environment. It will hang or output raw terminal control codes.

**Always use headless testing:**
- Run tests via `pytest` using Textual's `run_test()` API
- Use `app.query_one()` to assert widget state, not stdout
- Use `pytest-textual-snapshot` for visual testing (generates SVG files you can read)

**See `tests/test_frontend/test_charlie.py` for comprehensive testing patterns and examples.**
