# ASK ME QUESTIONS

Unless I request a simple change, any discussion of architecture, design, feature building, debugging, etc. should include clarifying questions asked ONE AT A TIME so we may align our understanding of the task & high level goals. Prompt me back!

## NO GIT PUSH, EVER

Git add & commit are fine for feature branches but NEVER for `main`!

## Comment Rules

- Add code comments sparingly.
- Focus on *why* something is done, especially for complex logic, rather than *what* is done.
- Only add high-value comments if necessary for clarity or if requested by the user.
- Do not edit comments that are separate from the code you are changing. *NEVER* talk to the user or describe your changes through comments.
- Ensure all comments are evergreen and don't reference line numbers or bugs that were previously fixed, espcially for tests.

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

## State Machine Diagram Regeneration

**CRITICAL: When modifying any state machine (adding/removing states, events, or transitions), regenerate its diagram immediately after implementation.**

State diagrams are generated from the state machine definition and must stay in sync to avoid diagram drift. If a plan or task mentions a state diagram file (e.g., `frontend/diagrams/sidebar_state_machine.png`), regenerate it whenever you modify the corresponding state machine.

**How to regenerate:**
- Look for a `generate_diagram()` function in the state machine module
- Call it explicitly: `python -c "from path.to.module import generate_diagram; generate_diagram()"`
- Verify the updated diagram is committed
- Never skip this step - diagram drift makes architecture documentation unreliable

### Misc. Things to Remember

- Use the respective conftest.py file for a given test suite for fixtures.
- Add imports to the top of the module unless necessary for a given feature.
- Opt for using hardcoded constants unless I explicitly ask to check for an env var.
- Always use the project's uv-managed virtual environment (e.g., `uv add ...`) instead of the system Python for installing dependencies and running tests.
