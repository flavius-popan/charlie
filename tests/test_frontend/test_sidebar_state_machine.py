"""Unit tests for SidebarStateMachine."""

import pytest
from frontend.state.sidebar_state_machine import SidebarStateMachine, SidebarOutput


class TestSidebarStateMachineInitialization:
    """Test state machine initialization."""

    def test_initial_state_is_hidden(self):
        """Machine should start in hidden state."""
        machine = SidebarStateMachine()
        assert machine.current_state == machine.hidden

    def test_output_when_hidden(self):
        """Hidden state should have correct output flags."""
        machine = SidebarStateMachine()
        output = machine.output
        assert output.visible is False
        assert output.should_poll is False
        assert output.active_processing is False
        assert output.loading is False

    def test_initialization_with_visible_flag(self):
        """Should seed to appropriate state when visible=True."""
        machine = SidebarStateMachine(
            visible=True,
            inference_enabled=True,
            initial_status="pending_nodes",
        )
        # Should be in processing_nodes (seeds in after_show)
        assert machine.current_state == machine.processing_nodes


class TestShowEvent:
    """Test show event transitions."""

    def test_show_from_hidden_to_processing_nodes(self):
        """Show should transition to processing_nodes when status pending_nodes."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        assert machine.current_state == machine.processing_nodes

    def test_show_from_hidden_to_awaiting_edges(self):
        """Show should transition to awaiting_edges when status pending_edges."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_edges", inference_enabled=True)
        assert machine.current_state == machine.awaiting_edges

    def test_show_from_hidden_to_disabled_when_inference_disabled(self):
        """Show should go to disabled if inference is disabled."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=False)
        assert machine.current_state == machine.disabled

    def test_show_with_entities_and_done_status(self):
        """Show should go to ready_entities when done and entities present."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities

    def test_show_without_entities_and_done_status(self):
        """Show should go to empty_idle when done and no entities."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=False)
        assert machine.current_state == machine.empty_idle

    def test_show_with_no_status(self):
        """Show with no status and no entities should go to empty_idle."""
        machine = SidebarStateMachine()
        machine.send("show", status=None, inference_enabled=True, entities_present=False)
        assert machine.current_state == machine.empty_idle


class TestHideEvent:
    """Test hide event transitions."""

    def test_hide_from_processing_nodes(self):
        """Hide should go to hidden from processing_nodes."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        assert machine.current_state == machine.processing_nodes
        machine.send("hide")
        assert machine.current_state == machine.hidden

    def test_hide_from_disabled(self):
        """Hide should go to hidden from disabled."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=False)
        assert machine.current_state == machine.disabled
        machine.send("hide")
        assert machine.current_state == machine.hidden

    def test_hide_from_ready_entities(self):
        """Hide should go to hidden from ready_entities."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities
        machine.send("hide")
        assert machine.current_state == machine.hidden


class TestInferenceToggle:
    """Test inference enabled/disabled transitions."""

    def test_inference_disabled_from_processing_nodes(self):
        """Disabling inference from processing should go to disabled."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        assert machine.current_state == machine.processing_nodes
        machine.send("inference_disabled")
        assert machine.current_state == machine.disabled
        assert machine.output.message is not None
        assert "Inference disabled" in machine.output.message

    def test_inference_disabled_from_ready_entities(self):
        """Disabling inference from ready_entities should go to disabled."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities
        machine.send("inference_disabled")
        assert machine.current_state == machine.disabled

    def test_inference_enabled_from_disabled_back_to_processing(self):
        """Enabling inference should restore appropriate state."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=False)
        assert machine.current_state == machine.disabled
        machine.send("inference_enabled", status="pending_nodes")
        assert machine.current_state == machine.processing_nodes

    def test_inference_enabled_with_entities(self):
        """Enabling inference with entities should go to ready_entities."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=False)
        assert machine.current_state == machine.disabled
        machine.send("inference_enabled", status="done", entities_present=True)
        assert machine.current_state == machine.ready_entities


class TestStatusUpdates:
    """Test status_pending_nodes and status_pending_edges_or_done events."""

    def test_status_pending_nodes_from_hidden(self):
        """status_pending_nodes from hidden should go to processing_nodes."""
        machine = SidebarStateMachine()
        machine.send("status_pending_nodes")
        assert machine.current_state == machine.processing_nodes

    def test_status_pending_edges_from_processing_nodes(self):
        """status_pending_edges from processing should go to awaiting_edges."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        machine.send("status_pending_edges_or_done", status="pending_edges")
        assert machine.current_state == machine.awaiting_edges

    def test_status_done_with_entities_from_awaiting(self):
        """Status done with entities should go to ready_entities."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_edges", inference_enabled=True)
        assert machine.current_state == machine.awaiting_edges
        machine.send("status_pending_edges_or_done", status="done", entities_present=True)
        assert machine.current_state == machine.ready_entities

    def test_status_done_without_entities_from_awaiting(self):
        """Status done without entities should go to empty_idle."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_edges", inference_enabled=True)
        assert machine.current_state == machine.awaiting_edges
        machine.send("status_pending_edges_or_done", status="done", entities_present=False)
        assert machine.current_state == machine.empty_idle


class TestCacheEvents:
    """Test cache_entities_found and cache_empty events."""

    def test_cache_entities_found_from_processing_nodes(self):
        """Cache hit during processing should go to ready_entities."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        machine.send("cache_entities_found")
        assert machine.current_state == machine.ready_entities

    def test_cache_entities_found_from_awaiting_edges(self):
        """Cache hit during awaiting should go to ready_entities."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_edges", inference_enabled=True)
        machine.send("cache_entities_found")
        assert machine.current_state == machine.ready_entities

    def test_cache_empty_from_processing_nodes(self):
        """Cache empty during processing should go to empty_idle."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        machine.send("cache_empty")
        assert machine.current_state == machine.empty_idle

    def test_cache_empty_from_ready_entities(self):
        """Cache empty from ready should go to empty_idle."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        machine.send("cache_empty")
        assert machine.current_state == machine.empty_idle


class TestUserDeletion:
    """Test user_deleted_entity event."""

    def test_delete_entity_with_more_left(self):
        """Deleting entity with more remaining should stay in ready_entities."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities
        # Still have entities after delete
        machine.send("user_deleted_entity", entities_present=True)
        assert machine.current_state == machine.ready_entities

    def test_delete_last_entity(self):
        """Deleting last entity should go to empty_idle."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities
        # No entities left after delete
        machine.send("user_deleted_entity", entities_present=False)
        assert machine.current_state == machine.empty_idle

    def test_delete_last_entity_clears_status(self):
        """Verify status is None in empty_idle state after last entity deletion.

        Status must be None to prevent EntitySidebar from showing incorrect UI messages
        based on stale status values.
        """
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities
        assert machine.output.status == "done"
        assert machine.output.active_processing is False

        machine.send("user_deleted_entity", entities_present=False)
        assert machine.current_state == machine.empty_idle
        assert machine.output.status is None
        assert machine.output.active_processing is False
        assert machine.output.entities_present is False


class TestEpisodeClosedEvent:
    """Test episode_closed event."""

    def test_episode_closed_from_processing(self):
        """Episode closed should go to hidden from any state."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        machine.send("episode_closed")
        assert machine.current_state == machine.hidden

    def test_episode_closed_from_disabled(self):
        """Episode closed from disabled should go to hidden."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=False)
        machine.send("episode_closed")
        assert machine.current_state == machine.hidden


class TestOutputFlags:
    """Test output property generation."""

    def test_hidden_state_output(self):
        """Hidden state output should have all flags false."""
        machine = SidebarStateMachine()
        output = machine.output
        assert output.visible is False
        assert output.should_poll is False
        assert output.active_processing is False
        assert output.loading is False
        assert output.message is None

    def test_processing_nodes_state_output(self):
        """Processing nodes output should indicate active processing."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        output = machine.output
        assert output.visible is True
        assert output.should_poll is True
        assert output.active_processing is True
        assert output.loading is True
        assert output.status == "pending_nodes"

    def test_ready_entities_state_output(self):
        """Ready entities output should have processing false."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        output = machine.output
        assert output.visible is True
        assert output.should_poll is False
        assert output.active_processing is False
        assert output.loading is False
        assert output.entities_present is True

    def test_empty_idle_state_output(self):
        """Empty idle output should show 'No connections' message."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=False)
        output = machine.output
        assert output.visible is True
        assert output.should_poll is False
        assert output.loading is False
        assert output.entities_present is False

    def test_disabled_state_output_message_pending(self):
        """Disabled state with pending status should show appropriate message."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=False)
        output = machine.output
        assert output.visible is True
        assert output.message is not None
        assert "paused" in output.message

    def test_disabled_state_output_message_done(self):
        """Disabled state with done status should show enable message."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=False)
        output = machine.output
        assert output.visible is True
        assert output.message is not None
        assert "enable inference" in output.message


class TestComplexScenarios:
    """Test realistic multi-step scenarios."""

    def test_full_extraction_workflow_from_edit(self):
        """Simulate opening from edit screen with pending extraction."""
        # Start from edit screen (visible, pending_nodes)
        machine = SidebarStateMachine(
            visible=True,
            inference_enabled=True,
            initial_status="pending_nodes",
        )
        assert machine.current_state == machine.processing_nodes
        assert machine.output.should_poll is True

        # Polling finds pending_edges
        machine.send("status_pending_edges_or_done", status="pending_edges")
        assert machine.current_state == machine.awaiting_edges

        # Cache fetch finds entities
        machine.send("cache_entities_found")
        assert machine.current_state == machine.ready_entities
        assert machine.output.loading is False

        # User hides sidebar
        machine.send("hide")
        assert machine.current_state == machine.hidden

    def test_inference_disable_and_reenable(self):
        """Test disabling and re-enabling inference during processing."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        assert machine.current_state == machine.processing_nodes

        # Disable inference
        machine.send("inference_disabled")
        assert machine.current_state == machine.disabled
        assert machine.output.should_poll is False

        # Re-enable while still pending
        machine.send("inference_enabled", status="pending_edges")
        assert machine.current_state == machine.awaiting_edges

    def test_deletion_workflow(self):
        """Test entity deletion flow."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities

        # Delete one entity but more remain
        machine.send("user_deleted_entity", entities_present=True)
        assert machine.current_state == machine.ready_entities

        # Delete last entity
        machine.send("user_deleted_entity", entities_present=False)
        assert machine.current_state == machine.empty_idle

    def test_back_from_processing_clears_polling(self):
        """Test that hiding during processing stops polling."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        assert machine.output.should_poll is True

        machine.send("episode_closed")
        assert machine.current_state == machine.hidden
        assert machine.output.should_poll is False


class TestApplyEventMethod:
    """Test the apply_event convenience method."""

    def test_apply_event_returns_output(self):
        """apply_event should return SidebarOutput."""
        machine = SidebarStateMachine()
        output = machine.apply_event("show", status="pending_nodes", inference_enabled=True)
        assert isinstance(output, SidebarOutput)
        assert output.visible is True

    def test_apply_event_handles_unknown_event(self):
        """apply_event should handle errors gracefully."""
        machine = SidebarStateMachine()
        # Try to call an invalid event from hidden (should fail silently)
        output = machine.apply_event("status_pending_edges_or_done")
        # Should return current state output (still hidden)
        assert output.visible is False

    def test_apply_event_chain(self):
        """apply_event can be chained to simulate sequence."""
        machine = SidebarStateMachine()
        output1 = machine.apply_event("show", status="pending_nodes", inference_enabled=True)
        assert output1.visible is True
        assert machine.current_state == machine.processing_nodes

        output2 = machine.apply_event("cache_entities_found")
        assert output2.visible is True
        assert machine.current_state == machine.ready_entities


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    def test_invalid_transition_ignored_gracefully(self):
        """Invalid transitions should be ignored, not crash."""
        machine = SidebarStateMachine()
        # user_deleted_entity is only valid from ready_entities and empty_idle
        # Trying from hidden should fail gracefully
        try:
            machine.apply_event("user_deleted_entity")
        except Exception:
            pytest.fail("Invalid transition should not raise exception")
        # Should stay in hidden
        assert machine.current_state == machine.hidden

    def test_data_persistence_across_transitions(self):
        """Status and entities_present should persist across state changes."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True, entities_present=False)
        assert machine.output.status == "pending_nodes"
        assert machine.output.entities_present is False

        # Transition to awaiting_edges, data should persist
        machine.send("status_pending_edges_or_done", status="pending_edges")
        assert machine.output.status == "pending_edges"
        assert machine.output.entities_present is False

        # Transition to ready_entities with new entities_present
        machine.send("cache_entities_found", entities_present=True)
        assert machine.output.status == "pending_edges"
        assert machine.output.entities_present is True

    def test_apply_event_with_missing_data(self):
        """apply_event should work with missing/partial data."""
        machine = SidebarStateMachine()
        # show event with only status, no inference_enabled
        output = machine.apply_event("show", status="pending_nodes")
        # Should use default inference_enabled=True from __init__
        assert machine.current_state == machine.processing_nodes

    def test_apply_event_with_extra_unknown_data(self):
        """apply_event should ignore unknown parameters."""
        machine = SidebarStateMachine()
        output = machine.apply_event(
            "show",
            status="pending_nodes",
            inference_enabled=True,
            unknown_param="should be ignored",
            another_param=123,
        )
        assert machine.current_state == machine.processing_nodes
        assert output.visible is True

    def test_rapid_status_updates(self):
        """Rapid status updates should queue correctly."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_nodes", inference_enabled=True)
        assert machine.current_state == machine.processing_nodes

        # Rapid fire events
        machine.send("status_pending_edges_or_done", status="pending_edges")
        assert machine.current_state == machine.awaiting_edges

        machine.send("cache_entities_found")
        assert machine.current_state == machine.ready_entities

        # Another status change shouldn't break things
        machine.send("status_pending_nodes")
        assert machine.current_state == machine.processing_nodes

        machine.send("cache_empty")
        assert machine.current_state == machine.empty_idle

    def test_inference_toggle_with_status_preservation(self):
        """Disabling/enabling inference should preserve status across toggles."""
        machine = SidebarStateMachine()
        machine.send("show", status="pending_edges", inference_enabled=True)
        assert machine.current_state == machine.awaiting_edges

        # Disable inference
        machine.send("inference_disabled")
        assert machine.current_state == machine.disabled
        assert machine.output.status == "pending_edges"  # Status preserved

        # Re-enable with same status
        machine.send("inference_enabled", status="pending_edges")
        assert machine.current_state == machine.awaiting_edges
        assert machine.output.status == "pending_edges"

    def test_entity_state_transitions_without_entities(self):
        """Transitions with empty entities list should go to empty_idle."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=False)
        assert machine.current_state == machine.empty_idle

        # Even if status changes but no entities
        machine.send("status_pending_nodes", entities_present=False)
        assert machine.current_state == machine.processing_nodes

        machine.send("cache_empty")
        assert machine.current_state == machine.empty_idle

    def test_state_after_multiple_deletions(self):
        """Multiple deletion events should handle entity depletion correctly."""
        machine = SidebarStateMachine()
        machine.send("show", status="done", inference_enabled=True, entities_present=True)
        assert machine.current_state == machine.ready_entities

        # Delete but entities remain
        machine.send("user_deleted_entity", entities_present=True)
        assert machine.current_state == machine.ready_entities

        # Delete more but still have entities
        machine.send("user_deleted_entity", entities_present=True)
        assert machine.current_state == machine.ready_entities

        # Delete last entity
        machine.send("user_deleted_entity", entities_present=False)
        assert machine.current_state == machine.empty_idle
