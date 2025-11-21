# ASK ME QUESTIONS

Unless I request a simple change, any discussion of architecture, design, feature building, debugging, etc. should include clarifying questions asked ONE AT A TIME so we may align our understanding of the task & high level goals. Prompt me back.

## NO GIT OPS

You are only permitted to run read-only git operations to understand the project state. You ARE NOT permitted to run git add, commit, push, etc.

## Testing Textual Applications

**CRITICAL:** Never run the Textual app directly (e.g., `python charlie.py`) in a non-interactive shell or agent environment. It will hang or output raw terminal control codes.

**Always use headless testing:**
- Run tests via `pytest` using Textual's `run_test()` API
- Use `app.query_one()` to assert widget state, not stdout
- Use `pytest-textual-snapshot` for visual testing (generates SVG files you can read)

**See `tests/test_frontend/test_charlie.py` for comprehensive testing patterns and examples.**

## Testing LLM Inference

**Mark all tests that make real LLM inference calls with `@pytest.mark.inference`:**
- Default behavior: `pytest` excludes inference tests (fast test runs)
- Run inference tests: `pytest -m inference`
- Run all tests: `pytest -m ""`
- Use the `require_llm` fixture to skip if no LLM configured
- Reuse the session-configured LLM (`dspy.settings.lm`) for all inference tests. Only tests that explicitly verify load/unload behavior should instantiate or unload models; everything else should use the shared model seeded by `configure_dspy_for_backend`/`reuse_session_lm`.

**Pattern:**
```python
@pytest.mark.inference
@pytest.mark.asyncio
async def test_with_llm_call(isolated_graph, require_llm):
    # Test automatically skips if dspy.settings.lm is None
    # Test code that makes LLM calls via DSPy
```

## Testing Best Practices

*ALWAYS* use the respective conftest.py file for a given test suite for fixtures.
