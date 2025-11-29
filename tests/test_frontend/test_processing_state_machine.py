"""Unit tests for ProcessingStateMachine."""

import pytest
from frontend.state.processing_state_machine import ProcessingStateMachine, ProcessingOutput


class TestProcessingOutputDataclass:
    """Test ProcessingOutput dataclass."""

    def test_idle_factory_creates_correct_defaults(self):
        """idle() factory should create output with all idle defaults."""
        output = ProcessingOutput.idle()
        assert output.pane_visible is False
        assert output.status_text == ""
        assert output.queue_text == ""
        assert output.show_dot is False
        assert output.poll_interval == 2.0
        assert output.is_inferring is False
        assert output.active_episode_uuid is None
        assert output.inference_enabled is True


class TestProcessingStateMachineInitialization:
    """Test state machine initialization."""

    def test_initial_state_is_idle(self):
        """Machine should start in idle state."""
        machine = ProcessingStateMachine()
        assert machine.current_state == machine.idle

    def test_output_when_idle(self):
        """Idle state should have correct output flags."""
        machine = ProcessingStateMachine()
        output = machine.output
        assert output.pane_visible is False
        assert output.status_text == ""
        assert output.show_dot is False
        assert output.poll_interval == 2.0
        assert output.is_inferring is False


class TestStateTransitions:
    """Test state transitions via status events."""

    def test_idle_to_loading(self):
        """status_loading from idle should go to loading."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")
        assert machine.current_state == machine.loading

    def test_idle_to_inferring(self):
        """status_inferring from idle should go to inferring."""
        machine = ProcessingStateMachine()
        machine.send("status_inferring")
        assert machine.current_state == machine.inferring

    def test_idle_to_unloading(self):
        """status_unloading from idle should go to unloading."""
        machine = ProcessingStateMachine()
        machine.send("status_unloading")
        assert machine.current_state == machine.unloading

    def test_loading_to_inferring(self):
        """status_inferring from loading should go to inferring."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")
        machine.send("status_inferring")
        assert machine.current_state == machine.inferring

    def test_loading_to_idle(self):
        """status_idle from loading should go to idle."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")
        machine.send("status_idle")
        assert machine.current_state == machine.idle

    def test_inferring_to_unloading(self):
        """status_unloading from inferring should go to unloading."""
        machine = ProcessingStateMachine()
        machine.send("status_inferring")
        machine.send("status_unloading")
        assert machine.current_state == machine.unloading

    def test_inferring_to_idle(self):
        """status_idle from inferring should go to idle."""
        machine = ProcessingStateMachine()
        machine.send("status_inferring")
        machine.send("status_idle")
        assert machine.current_state == machine.idle

    def test_unloading_to_idle(self):
        """status_idle from unloading should go to idle."""
        machine = ProcessingStateMachine()
        machine.send("status_unloading")
        machine.send("status_idle")
        assert machine.current_state == machine.idle


class TestSelfTransitions:
    """Test self-transitions for idempotent polling."""

    def test_idle_self_transition(self):
        """status_idle from idle should stay in idle (no TransitionNotAllowed)."""
        machine = ProcessingStateMachine()
        assert machine.current_state == machine.idle
        machine.send("status_idle")
        assert machine.current_state == machine.idle

    def test_loading_self_transition(self):
        """status_loading from loading should stay in loading."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")
        machine.send("status_loading")
        assert machine.current_state == machine.loading

    def test_inferring_self_transition(self):
        """status_inferring from inferring should stay in inferring."""
        machine = ProcessingStateMachine()
        machine.send("status_inferring")
        machine.send("status_inferring")
        assert machine.current_state == machine.inferring

    def test_unloading_self_transition(self):
        """status_unloading from unloading should stay in unloading."""
        machine = ProcessingStateMachine()
        machine.send("status_unloading")
        machine.send("status_unloading")
        assert machine.current_state == machine.unloading

    def test_repeated_polling_no_error(self):
        """Simulating repeated polling should not raise errors."""
        machine = ProcessingStateMachine()
        # Simulate repeated idle polls
        for _ in range(5):
            machine.send("status_idle")
        assert machine.current_state == machine.idle

        # Transition and repeat
        machine.send("status_loading")
        for _ in range(5):
            machine.send("status_loading")
        assert machine.current_state == machine.loading


class TestOutputProperties:
    """Test output property computation for each state."""

    def test_idle_output(self):
        """Idle state output should hide pane and poll slowly."""
        machine = ProcessingStateMachine()
        output = machine.output
        assert output.pane_visible is False
        assert output.status_text == ""
        assert output.queue_text == ""
        assert output.show_dot is False
        assert output.poll_interval == 2.0
        assert output.is_inferring is False

    def test_loading_output(self):
        """Loading state output should show pane with loading text."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")
        output = machine.output
        assert output.pane_visible is True
        assert output.status_text == "Loading model..."
        assert output.show_dot is True
        assert output.poll_interval == 0.3
        assert output.is_inferring is False

    def test_inferring_output(self):
        """Inferring state output should show extracting text."""
        machine = ProcessingStateMachine()
        machine.send("status_inferring")
        output = machine.output
        assert output.pane_visible is True
        assert output.status_text == "Extracting:"
        assert output.show_dot is True
        assert output.poll_interval == 0.3
        assert output.is_inferring is True

    def test_unloading_output(self):
        """Unloading state output should show unloading text."""
        machine = ProcessingStateMachine()
        machine.send("status_unloading")
        output = machine.output
        assert output.pane_visible is True
        assert output.status_text == "Unloading model..."
        assert output.show_dot is True
        assert output.poll_interval == 0.3
        assert output.is_inferring is False


class TestInferringWithInferenceDisabled:
    """Test 'Finishing extracting:' variant when inference disabled."""

    def test_inferring_with_inference_disabled_shows_finishing(self):
        """When inferring but inference_enabled=False, show 'Finishing extracting:'."""
        machine = ProcessingStateMachine()
        # Use apply_status to set inference_enabled to False while inferring
        status = {
            "model_state": "inferring",
            "active_uuid": "test-uuid",
            "pending_count": 0,
            "inference_enabled": False,
        }
        output = machine.apply_status(status)
        assert output.status_text == "Finishing extracting:"
        assert output.is_inferring is True
        assert output.inference_enabled is False

    def test_inferring_with_inference_enabled_shows_extracting(self):
        """When inferring with inference_enabled=True, show 'Extracting:'."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "inferring",
            "active_uuid": "test-uuid",
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.status_text == "Extracting:"
        assert output.is_inferring is True
        assert output.inference_enabled is True


class TestQueueText:
    """Test queue_text output computation."""

    def test_queue_text_with_pending(self):
        """Queue text should show count when pending_count > 0."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "inferring",
            "active_uuid": "test-uuid",
            "pending_count": 5,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.queue_text == "Queue: 5 remaining"

    def test_queue_text_empty_when_zero(self):
        """Queue text should be empty when pending_count is 0."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "inferring",
            "active_uuid": "test-uuid",
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.queue_text == ""


class TestApplyStatus:
    """Test apply_status method."""

    def test_apply_status_idle(self):
        """apply_status with idle model_state should go to idle."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")  # Start non-idle
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert machine.current_state == machine.idle
        assert output.pane_visible is False

    def test_apply_status_loading(self):
        """apply_status with loading model_state should go to loading."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "loading",
            "active_uuid": None,
            "pending_count": 3,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert machine.current_state == machine.loading
        assert output.pane_visible is True
        assert output.status_text == "Loading model..."

    def test_apply_status_inferring(self):
        """apply_status with inferring model_state should go to inferring."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "inferring",
            "active_uuid": "episode-123",
            "pending_count": 2,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert machine.current_state == machine.inferring
        assert output.is_inferring is True
        assert output.active_episode_uuid == "episode-123"

    def test_apply_status_unloading(self):
        """apply_status with unloading model_state should go to unloading."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "unloading",
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert machine.current_state == machine.unloading
        assert output.status_text == "Unloading model..."

    def test_apply_status_returns_output(self):
        """apply_status should return ProcessingOutput."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert isinstance(output, ProcessingOutput)


class TestApplyStatusDefensiveDefaults:
    """Test apply_status handles malformed/missing data."""

    def test_missing_model_state_defaults_to_idle(self):
        """Missing model_state should default to idle."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")  # Start non-idle
        status = {
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert machine.current_state == machine.idle

    def test_unknown_model_state_defaults_to_idle(self):
        """Unknown model_state value should default to idle."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")  # Start non-idle
        status = {
            "model_state": "unknown_state",
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert machine.current_state == machine.idle

    def test_missing_pending_count_defaults_to_zero(self):
        """Missing pending_count should default to 0."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "inferring",
            "active_uuid": "test-uuid",
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.queue_text == ""  # pending_count=0 means empty queue_text

    def test_missing_inference_enabled_defaults_to_true(self):
        """Missing inference_enabled should default to True."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "inferring",
            "active_uuid": "test-uuid",
            "pending_count": 0,
        }
        output = machine.apply_status(status)
        assert output.inference_enabled is True
        assert output.status_text == "Extracting:"  # Not "Finishing extracting:"

    def test_empty_status_dict(self):
        """Empty status dict should default to idle with safe values."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")  # Start non-idle
        output = machine.apply_status({})
        assert machine.current_state == machine.idle
        assert output.pane_visible is False


class TestPollInterval:
    """Test poll_interval computation."""

    def test_poll_interval_slow_when_idle(self):
        """Poll interval should be 2.0s when idle."""
        machine = ProcessingStateMachine()
        assert machine.output.poll_interval == 2.0

    def test_poll_interval_fast_when_loading(self):
        """Poll interval should be 0.3s when loading."""
        machine = ProcessingStateMachine()
        machine.send("status_loading")
        assert machine.output.poll_interval == 0.3

    def test_poll_interval_fast_when_inferring(self):
        """Poll interval should be 0.3s when inferring."""
        machine = ProcessingStateMachine()
        machine.send("status_inferring")
        assert machine.output.poll_interval == 0.3

    def test_poll_interval_fast_when_unloading(self):
        """Poll interval should be 0.3s when unloading."""
        machine = ProcessingStateMachine()
        machine.send("status_unloading")
        assert machine.output.poll_interval == 0.3


class TestCompleteWorkflows:
    """Test complete processing workflows."""

    def test_full_processing_cycle(self):
        """Simulate full cycle: idle -> loading -> inferring -> unloading -> idle."""
        machine = ProcessingStateMachine()

        # Start idle
        assert machine.current_state == machine.idle
        assert machine.output.pane_visible is False

        # Loading model
        status = {"model_state": "loading", "active_uuid": None, "pending_count": 3, "inference_enabled": True}
        output = machine.apply_status(status)
        assert machine.current_state == machine.loading
        assert output.pane_visible is True
        assert output.status_text == "Loading model..."

        # Inferring
        status = {"model_state": "inferring", "active_uuid": "ep-1", "pending_count": 2, "inference_enabled": True}
        output = machine.apply_status(status)
        assert machine.current_state == machine.inferring
        assert output.status_text == "Extracting:"
        assert output.active_episode_uuid == "ep-1"
        assert output.queue_text == "Queue: 2 remaining"

        # Continue inferring with different episode
        status = {"model_state": "inferring", "active_uuid": "ep-2", "pending_count": 1, "inference_enabled": True}
        output = machine.apply_status(status)
        assert machine.current_state == machine.inferring
        assert output.active_episode_uuid == "ep-2"
        assert output.queue_text == "Queue: 1 remaining"

        # Unloading model
        status = {"model_state": "unloading", "active_uuid": None, "pending_count": 0, "inference_enabled": True}
        output = machine.apply_status(status)
        assert machine.current_state == machine.unloading
        assert output.status_text == "Unloading model..."

        # Back to idle
        status = {"model_state": "idle", "active_uuid": None, "pending_count": 0, "inference_enabled": True}
        output = machine.apply_status(status)
        assert machine.current_state == machine.idle
        assert output.pane_visible is False

    def test_inference_disabled_during_extraction(self):
        """When inference disabled during extraction, show 'Finishing extracting:'."""
        machine = ProcessingStateMachine()

        # Start inferring with inference enabled
        status = {"model_state": "inferring", "active_uuid": "ep-1", "pending_count": 5, "inference_enabled": True}
        output = machine.apply_status(status)
        assert output.status_text == "Extracting:"

        # User disables inference while extracting
        status = {"model_state": "inferring", "active_uuid": "ep-1", "pending_count": 5, "inference_enabled": False}
        output = machine.apply_status(status)
        assert output.status_text == "Finishing extracting:"
        assert output.is_inferring is True

        # Finishes and unloads
        status = {"model_state": "unloading", "active_uuid": None, "pending_count": 5, "inference_enabled": False}
        output = machine.apply_status(status)
        assert output.status_text == "Unloading model..."

    def test_rapid_state_transitions(self):
        """Rapid state transitions within poll cycles should work correctly."""
        machine = ProcessingStateMachine()

        # Rapid transitions
        machine.apply_status({"model_state": "loading", "active_uuid": None, "pending_count": 1, "inference_enabled": True})
        machine.apply_status({"model_state": "inferring", "active_uuid": "ep-1", "pending_count": 0, "inference_enabled": True})
        machine.apply_status({"model_state": "unloading", "active_uuid": None, "pending_count": 0, "inference_enabled": True})
        output = machine.apply_status({"model_state": "idle", "active_uuid": None, "pending_count": 0, "inference_enabled": True})

        assert machine.current_state == machine.idle
        assert output.pane_visible is False


class TestInternalDataTracking:
    """Test that internal data is correctly tracked."""

    def test_active_episode_uuid_tracked(self):
        """active_episode_uuid should be tracked in output."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "inferring",
            "active_uuid": "test-episode-uuid",
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.active_episode_uuid == "test-episode-uuid"

    def test_active_episode_uuid_none_when_not_inferring(self):
        """active_episode_uuid should still be in output even when not inferring."""
        machine = ProcessingStateMachine()
        # First set an active episode
        machine.apply_status({
            "model_state": "inferring",
            "active_uuid": "test-uuid",
            "pending_count": 0,
            "inference_enabled": True,
        })
        # Then transition to unloading with no active uuid
        output = machine.apply_status({
            "model_state": "unloading",
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": True,
        })
        assert output.active_episode_uuid is None

    def test_inference_enabled_passed_through(self):
        """inference_enabled should be passed through to output."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": False,
        }
        output = machine.apply_status(status)
        assert output.inference_enabled is False


class TestQueueAwareVisibility:
    """Test pane visibility and poll interval when queue has pending work."""

    def test_pane_not_visible_when_idle_with_pending_queue(self):
        """Pane should NOT be visible when idle, even with pending queue (grace period)."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 5,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert machine.current_state == machine.idle
        assert output.pane_visible is False  # Hidden during grace period

    def test_poll_interval_medium_when_idle_with_pending_queue(self):
        """Poll interval should be medium (0.5s) when idle but queue has pending work."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 3,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.poll_interval == 0.5  # Medium speed - faster than idle but not as fast as active

    def test_pane_hidden_when_idle_with_empty_queue(self):
        """Pane should be hidden when idle and queue is empty."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 0,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.pane_visible is False

    def test_show_dot_false_when_idle_with_pending_queue(self):
        """Show dot should be False when idle (even with pending queue)."""
        machine = ProcessingStateMachine()
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 5,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        # Dot only shows during active processing, not while waiting
        assert output.show_dot is False

    def test_transition_between_jobs_hides_pane_briefly(self):
        """Pane hides when idle between jobs, then shows when next job loads."""
        machine = ProcessingStateMachine()

        # First job inferring
        status = {
            "model_state": "inferring",
            "active_uuid": "ep-1",
            "pending_count": 2,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.pane_visible is True

        # First job completes, goes to idle (pane hides during grace period)
        status = {
            "model_state": "idle",
            "active_uuid": None,
            "pending_count": 1,  # Still has pending work
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        # Pane hides during idle/grace period
        assert output.pane_visible is False
        assert output.poll_interval == 0.5  # Medium polling to pick up next job

        # Next job starts loading
        status = {
            "model_state": "loading",
            "active_uuid": None,
            "pending_count": 1,
            "inference_enabled": True,
        }
        output = machine.apply_status(status)
        assert output.pane_visible is True
