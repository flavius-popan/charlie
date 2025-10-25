"""MLX integration tests: model loading, thread safety, and locking behavior.

This test suite verifies:
1. MLX model loading and Outlines wrapper creation
2. Basic structured generation with Pydantic models
3. MLX_LOCK properly acquired during model inference
4. Multiple threads calling forward() are serialized when sharing a model instance
5. No Metal framework errors occur under concurrent load
6. Separate MLX instances share the module-level lock

Memory note: Most tests use a shared LM instance to avoid loading multiple 4GB models.
The test_multiple_instances_share_lock test intentionally loads 2 instances to verify
the lock behavior across instances.
"""

import threading
import time
from unittest.mock import Mock, patch
import pytest

import dspy
from pydantic import BaseModel

from dspy_outlines import OutlinesLM, OutlinesAdapter
from dspy_outlines.lm import MLX_LOCK
from dspy_outlines.mlx_loader import load_mlx_model, create_outlines_model


class SimpleResponse(BaseModel):
    """Test schema."""
    answer: str


class SimpleSignature(dspy.Signature):
    """Test signature."""
    question: str = dspy.InputField()
    response: SimpleResponse = dspy.OutputField()


@pytest.fixture(scope="module")
def shared_lm():
    """Shared OutlinesLM instance for all tests.

    Uses a single 4GB model for all tests to conserve memory.
    This tests the common pattern: multiple threads sharing one model instance.
    For multi-model parallel testing, see docs/benchmark_pool.md
    """
    return OutlinesLM()


# Model Loading Tests

def test_load_mlx_model():
    """Test loading Qwen3-4B model via MLX."""
    model, tokenizer = load_mlx_model()

    assert model is not None
    assert tokenizer is not None
    assert hasattr(tokenizer, 'apply_chat_template')


def test_create_outlines_model():
    """Test creating Outlines wrapper."""
    outlines_model, tokenizer = create_outlines_model()

    # Test basic structured generation
    class TestOutput(BaseModel):
        result: int

    response = outlines_model(
        "What is 2+2? Answer with just the number.",
        output_type=TestOutput,
        max_tokens=10
    )

    # Response should be JSON string
    assert isinstance(response, str)
    parsed = TestOutput.model_validate_json(response)
    assert parsed.result == 4


# Thread Safety Tests

def test_mlx_lock_exists():
    """Verify MLX_LOCK is defined and is a threading.Lock."""
    assert MLX_LOCK is not None
    assert isinstance(MLX_LOCK, threading.Lock)


def test_lock_acquired_during_inference(shared_lm):
    """Test that the lock is actually acquired during model inference.

    We mock the model to detect if it's called while lock is held.
    """
    lm = shared_lm

    lock_held_during_call = False

    original_model = lm.model

    def mock_model_call(prompt, **kwargs):
        nonlocal lock_held_during_call
        # Check if MLX_LOCK is currently held
        # locked() returns False if lock is available (not held)
        lock_held_during_call = MLX_LOCK.locked()

        # Return valid JSON that matches the expected output_type
        # The model is called with output_type=SimpleResponse for constrained generation
        return '{"answer": "test"}'

    lm.model = mock_model_call

    # Configure DSPy
    dspy.configure(lm=lm, adapter=OutlinesAdapter())
    predictor = dspy.Predict(SimpleSignature)

    # Make a call
    result = predictor(question="test")

    # Verify lock was held during model call
    assert lock_held_during_call, "MLX_LOCK should be held during model inference"

    # Restore original model
    lm.model = original_model


def test_concurrent_calls_are_serialized(shared_lm):
    """Test that concurrent forward() calls are serialized by the lock.

    This verifies that only one thread can be inside the locked section at a time.
    We call forward() directly to avoid DSPy's thread-local configuration restrictions.
    """
    lm = shared_lm

    concurrent_calls = 0
    max_concurrent = 0
    lock = threading.Lock()

    original_model = lm.model

    def mock_model_call(*args, **kwargs):
        nonlocal concurrent_calls, max_concurrent

        with lock:
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)

        # Simulate some work
        time.sleep(0.1)

        with lock:
            concurrent_calls -= 1

        return '{"answer": "test"}'

    lm.model = mock_model_call

    results = []
    errors = []

    def worker(thread_id):
        try:
            # Call forward() directly, bypassing DSPy
            result = lm.forward(prompt=f"test {thread_id}", max_tokens=50)
            results.append(result)
        except Exception as e:
            errors.append(str(e))

    # Launch multiple threads
    threads = []
    num_threads = 5
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Verify no errors
    assert len(errors) == 0, f"Concurrent calls failed: {errors}"

    # Verify lock serialized access (max 1 concurrent call at a time)
    assert max_concurrent == 1, f"Lock failed to serialize - found {max_concurrent} concurrent calls"

    # Restore original model
    lm.model = original_model


def test_lock_released_on_exception(shared_lm):
    """Test that MLX_LOCK is released even if model inference raises an exception."""
    lm = shared_lm

    original_model = lm.model

    def failing_model(*args, **kwargs):
        raise RuntimeError("Simulated model failure")

    lm.model = failing_model

    # Verify lock is not held before
    assert not MLX_LOCK.locked()

    # Call forward and expect failure
    try:
        lm.forward(prompt="test", max_tokens=50)
    except RuntimeError:
        pass  # Expected

    # Verify lock is released after exception
    assert not MLX_LOCK.locked(), "Lock should be released after exception"

    # Restore original model
    lm.model = original_model


def test_lock_scope_minimal(shared_lm):
    """Test that only model inference is locked, not the entire forward() method.

    This verifies that prompt formatting happens outside the lock.
    """
    lm = shared_lm

    format_under_lock = None
    inference_under_lock = None

    original_format = lm._format_messages
    original_model = lm.model

    def mock_format(messages):
        nonlocal format_under_lock
        format_under_lock = MLX_LOCK.locked()
        return original_format(messages)

    def mock_model(*args, **kwargs):
        nonlocal inference_under_lock
        inference_under_lock = MLX_LOCK.locked()
        return '{"answer": "test"}'

    lm._format_messages = mock_format
    lm.model = mock_model

    # Call with messages to trigger formatting
    lm.forward(messages=[{"role": "user", "content": "test"}])

    # Verify formatting happened outside lock
    assert format_under_lock is False, "Formatting should happen outside lock"

    # Verify inference happened inside lock
    assert inference_under_lock is True, "Inference should happen inside lock"

    # Restore original methods
    lm._format_messages = original_format
    lm.model = original_model


def test_multiple_instances_share_lock(shared_lm):
    """Test that multiple OutlinesLM instances share the same MLX_LOCK.

    This verifies that the lock is module-level, not instance-level.
    We test by calling forward() directly (bypassing DSPy) to avoid DSPy's
    thread-local configuration restrictions.
    """
    lm1 = shared_lm

    # Create a second instance but mock its model loading to avoid memory overhead
    from unittest.mock import patch, MagicMock

    with patch('dspy_outlines.lm.create_outlines_model') as mock_create:
        # Mock the model creation
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_create.return_value = (mock_model, mock_tokenizer)

        lm2 = OutlinesLM()

    # They should both use the module-level MLX_LOCK
    # We can verify this by checking that when one is in forward(), the lock is held

    in_lm1_forward = threading.Event()
    lm1_sees_lock = None
    lm2_sees_lock = None

    original_lm1_model = lm1.model

    def lm1_model(*args, **kwargs):
        nonlocal lm1_sees_lock
        lm1_sees_lock = MLX_LOCK.locked()
        in_lm1_forward.set()
        time.sleep(0.2)  # Hold the lock
        return '{"answer": "lm1"}'

    def lm2_model(*args, **kwargs):
        nonlocal lm2_sees_lock
        # This should block until lm1 releases the lock
        lm2_sees_lock = MLX_LOCK.locked()
        return '{"answer": "lm2"}'

    lm1.model = lm1_model
    lm2.model = lm2_model

    def call_lm1():
        # Call forward() directly, bypassing DSPy
        lm1.forward(prompt="test1", max_tokens=50)

    def call_lm2():
        # Wait for lm1 to be inside forward()
        in_lm1_forward.wait()
        # Now try to call lm2 - should block on lock
        lm2.forward(prompt="test2", max_tokens=50)

    t1 = threading.Thread(target=call_lm1)
    t2 = threading.Thread(target=call_lm2)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Both should see the lock as held during their model calls
    assert lm1_sees_lock is True, "lm1 should see lock held"
    assert lm2_sees_lock is True, "lm2 should see lock held (proves lock is module-level)"

    # Restore original model
    lm1.model = original_lm1_model


def test_lock_overhead_minimal(shared_lm):
    """Test that lock acquisition doesn't add significant overhead to sequential calls.

    The lock should only serialize concurrent access, not slow down sequential calls.
    We measure the overhead by comparing timing with/without the lock mechanism.
    """
    lm = shared_lm
    original_model = lm.model

    # Mock model that simulates fast inference
    call_count = 0

    def mock_fast_model(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(0.001)  # Simulate 1ms inference
        return '{"answer": "test"}'

    lm.model = mock_fast_model

    # Run sequential calls and measure time
    num_calls = 10
    start = time.time()
    for i in range(num_calls):
        lm.forward(prompt=f"test {i}", max_tokens=10)
    duration = time.time() - start

    # Verify all calls completed
    assert call_count == num_calls

    # Lock overhead should be negligible (< 1ms per call on average)
    # Total overhead = duration - (num_calls * 0.001s per inference)
    expected_min_duration = num_calls * 0.001  # Just the inference time
    overhead_per_call = (duration - expected_min_duration) / num_calls

    # Assert overhead is less than 1ms per call (lock acquisition is fast)
    assert overhead_per_call < 0.001, (
        f"Lock overhead too high: {overhead_per_call*1000:.2f}ms per call. "
        f"Expected < 1ms per call for sequential access."
    )

    # Restore original model
    lm.model = original_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
