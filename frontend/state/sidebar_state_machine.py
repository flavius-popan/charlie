"""Sidebar state machine for managing entity sidebar visibility and processing state.

This module provides a pure logic state machine (no Textual dependencies) that models
sidebar visibility, processing states, and user interactions. The machine outputs
flags and messages that UI components use for rendering.
"""

import logging
from dataclasses import dataclass
from statemachine import StateMachine, State

logger = logging.getLogger("charlie")


@dataclass
class SidebarOutput:
    """Output state from the sidebar state machine."""

    visible: bool
    """Whether sidebar should be displayed."""

    should_poll: bool
    """Whether ViewScreen should start polling for status updates."""

    active_processing: bool
    """Whether processing is actively happening (show spinner)."""

    loading: bool
    """Whether still waiting for initial data load."""

    message: str | None
    """Optional override message (e.g., 'Inference disabled...')."""

    status: str | None
    """Current backend status: 'pending_nodes', 'pending_edges', 'done', or None."""

    entities_present: bool
    """Whether any entities are available to display."""


class SidebarStateMachine(StateMachine):
    """State machine for sidebar visibility and processing lifecycle.

    This pure-logic state machine (no Textual dependencies) models the sidebar's
    visibility and processing state, producing output flags for UI consumption.

    States:
    - hidden: Sidebar not displayed
    - disabled: Inference disabled (sidebar visible but shows disabled message)
    - processing_nodes: Extracting nodes (sidebar visible with spinner)
    - awaiting_edges: Nodes extracted, extracting edges (spinner, may have partial results)
    - ready_entities: Extraction complete with entities (show entity list)
    - empty_idle: Extraction complete, no entities (show 'No connections found')

    Events and Guards:
    Guards are evaluated before before_* hooks execute. This allows guards to route
    events conditionally based on event data (available in the data dict) and current
    machine state. Guards check `data.get("key", self._key)` to prefer event data
    over stale internal state when deciding transitions.

    Maintenance:
    After modifying states, events, or transitions, regenerate the state diagram
    using generate_diagram() to keep frontend/diagrams/sidebar_state_machine.png
    in sync with the code.

    Example:
        >>> machine = SidebarStateMachine(inference_enabled=True, initial_status="pending_nodes")
        >>> # Route show event to processing_nodes based on data and guards
        >>> output = machine.apply_event("show", status="pending_nodes", inference_enabled=True)
        >>> assert machine.current_state == machine.processing_nodes
        >>> assert output.should_poll is True
    """

    # States
    hidden = State(initial=True)
    disabled = State()
    processing_nodes = State()
    awaiting_edges = State()
    ready_entities = State()
    empty_idle = State()

    # Events
    show = (
        hidden.to(disabled, cond="_should_go_disabled")
        | hidden.to(processing_nodes, cond="_should_go_processing")
        | hidden.to(awaiting_edges, cond="_should_go_awaiting")
        | hidden.to(ready_entities, cond="_should_go_ready")
        | hidden.to(empty_idle)
    )

    hide = (
        disabled.to(hidden)
        | processing_nodes.to(hidden)
        | awaiting_edges.to(hidden)
        | ready_entities.to(hidden)
        | empty_idle.to(hidden)
    )

    episode_closed = (
        disabled.to(hidden)
        | processing_nodes.to(hidden)
        | awaiting_edges.to(hidden)
        | ready_entities.to(hidden)
        | empty_idle.to(hidden)
    )

    inference_disabled = (
        hidden.to(disabled)
        | processing_nodes.to(disabled)
        | awaiting_edges.to(disabled)
        | ready_entities.to(disabled)
        | empty_idle.to(disabled)
    )

    inference_enabled = (
        disabled.to(processing_nodes, cond="_should_go_processing_on_enable")
        | disabled.to(awaiting_edges, cond="_should_go_awaiting_on_enable")
        | disabled.to(ready_entities, cond="_should_go_ready_on_enable")
        | disabled.to(empty_idle)
    )

    status_pending_nodes = (
        hidden.to(processing_nodes)
        | awaiting_edges.to(processing_nodes)
        | ready_entities.to(processing_nodes)
        | empty_idle.to(processing_nodes)
        | processing_nodes.to(processing_nodes)
    )

    status_pending_edges_or_done = (
        processing_nodes.to(awaiting_edges, cond="_should_go_awaiting_from_processing")
        | processing_nodes.to(ready_entities, cond="_should_go_ready_from_processing")
        | processing_nodes.to(empty_idle)
        | awaiting_edges.to(ready_entities, cond="_should_go_ready_from_awaiting")
        | awaiting_edges.to(empty_idle)
        | hidden.to(hidden)
        | ready_entities.to(ready_entities)
        | empty_idle.to(empty_idle)
    )

    cache_entities_found = (
        processing_nodes.to(ready_entities)
        | awaiting_edges.to(ready_entities)
        | empty_idle.to(ready_entities)
        | hidden.to(ready_entities)
        | ready_entities.to(ready_entities)
    )

    cache_empty = (
        processing_nodes.to(empty_idle)
        | awaiting_edges.to(empty_idle)
        | ready_entities.to(empty_idle)
        | hidden.to(empty_idle)
        | empty_idle.to(empty_idle)
    )

    user_deleted_entity = (
        ready_entities.to(empty_idle, cond="_no_entities_left")
        | ready_entities.to(ready_entities)
        | empty_idle.to(empty_idle)
    )

    def __init__(
        self,
        initial_status: str | None = None,
        inference_enabled: bool = True,
        entities_present: bool = False,
        visible: bool = False,
    ):
        """Initialize the sidebar state machine.

        Args:
            initial_status: Starting backend status ('pending_nodes', 'pending_edges', 'done', None).
            inference_enabled: Whether inference is enabled.
            entities_present: Whether entities are initially available.
            visible: Whether sidebar should be initially visible. When True, the machine
                immediately seeds to the appropriate state (processing_nodes, awaiting_edges,
                ready_entities, etc.) based on status and inference_enabled. This is used
                for the "from_edit" initialization path.

                For other scenarios (e.g., home screen), create with visible=False and later
                call apply_event("show", ...) to control the visible transition explicitly.
                This prevents race conditions between machine state and UI state.
        """
        super().__init__()
        self._status = initial_status
        self._inference_enabled = inference_enabled
        self._entities_present = entities_present
        self._initial_visible = visible

        # Seed initial state if needed
        if visible and inference_enabled:
            self._seed_initial_state()

    def _seed_initial_state(self) -> None:
        """Transition to appropriate state based on initial conditions."""
        if not self._inference_enabled:
            self.send("inference_disabled")
        elif self._status == "pending_nodes":
            self.send("status_pending_nodes")
        elif self._status == "pending_edges":
            self.send("status_pending_edges_or_done")
        elif self._entities_present:
            self.send("cache_entities_found")
        else:
            self.send("cache_empty")

    @property
    def output(self) -> SidebarOutput:
        """Generate output state based on current machine state."""
        return SidebarOutput(
            visible=self.visible_flag,
            should_poll=self.should_poll_flag,
            active_processing=self.active_processing_flag,
            loading=self.loading_flag,
            message=self.message_flag,
            status=self._status,
            entities_present=self._entities_present,
        )

    @property
    def visible_flag(self) -> bool:
        """Whether sidebar should be displayed."""
        return self.current_state != self.hidden

    @property
    def should_poll_flag(self) -> bool:
        """Whether ViewScreen should run polling worker."""
        return self.current_state in (self.processing_nodes, self.awaiting_edges)

    @property
    def active_processing_flag(self) -> bool:
        """Whether to show active processing indicator."""
        return self.current_state in (self.processing_nodes, self.awaiting_edges)

    @property
    def loading_flag(self) -> bool:
        """Whether still loading/processing."""
        return self.current_state in (
            self.processing_nodes,
            self.awaiting_edges,
        )

    @property
    def message_flag(self) -> str | None:
        """Override message for sidebar display."""
        if self.current_state == self.disabled:
            if self._status in ("pending_nodes", "pending_edges"):
                return "Inference disabled; extraction is paused."
            return "Inference disabled; enable inference to extract connections."
        return None

    @property
    def inference_enabled_flag(self) -> bool:
        """Whether inference is currently enabled."""
        return self._inference_enabled

    # Guard conditions for routing

    def _should_go_disabled(self, **data) -> bool:
        """Guard: should go to disabled state."""
        inference_enabled = data.get("inference_enabled", self._inference_enabled)
        return not inference_enabled

    def _should_go_processing(self, **data) -> bool:
        """Guard: should go to processing_nodes."""
        inference_enabled = data.get("inference_enabled", self._inference_enabled)
        status = data.get("status", self._status)
        return inference_enabled and status == "pending_nodes"

    def _should_go_awaiting(self, **data) -> bool:
        """Guard: should go to awaiting_edges."""
        inference_enabled = data.get("inference_enabled", self._inference_enabled)
        status = data.get("status", self._status)
        return inference_enabled and status == "pending_edges"

    def _should_go_ready(self, **data) -> bool:
        """Guard: should go to ready_entities."""
        inference_enabled = data.get("inference_enabled", self._inference_enabled)
        status = data.get("status", self._status)
        entities_present = data.get("entities_present", self._entities_present)
        return inference_enabled and status in ("done", None) and entities_present

    def _should_go_processing_on_enable(self, **data) -> bool:
        """Guard: should go to processing when enabling with pending_nodes."""
        status = data.get("status", self._status)
        return status == "pending_nodes"

    def _should_go_awaiting_on_enable(self, **data) -> bool:
        """Guard: should go to awaiting when enabling with pending_edges."""
        status = data.get("status", self._status)
        return status == "pending_edges"

    def _should_go_ready_on_enable(self, **data) -> bool:
        """Guard: should go to ready when enabling with entities."""
        status = data.get("status", self._status)
        entities_present = data.get("entities_present", self._entities_present)
        return status in ("done", None) and entities_present

    def _should_go_awaiting_from_processing(self, **data) -> bool:
        """Guard: should go from processing to awaiting."""
        status = data.get("status", self._status)
        return status == "pending_edges"

    def _should_go_ready_from_processing(self, **data) -> bool:
        """Guard: should go from processing to ready."""
        status = data.get("status", self._status)
        entities_present = data.get("entities_present", self._entities_present)
        return status in ("done", None) and entities_present

    def _should_go_ready_from_awaiting(self, **data) -> bool:
        """Guard: should go from awaiting to ready."""
        status = data.get("status", self._status)
        entities_present = data.get("entities_present", self._entities_present)
        return status in ("done", None) and entities_present

    def _no_entities_left(self, **data) -> bool:
        """Guard: no entities left after deletion."""
        entities_present = data.get("entities_present", self._entities_present)
        return not entities_present

    # Event handlers for data updates

    def before_show(
        self,
        status: str | None = None,
        inference_enabled: bool | None = None,
        entities_present: bool | None = None,
        **data,
    ) -> None:
        """Capture data when sidebar is shown."""
        if status is not None:
            self._status = status
        if inference_enabled is not None:
            self._inference_enabled = inference_enabled
        if entities_present is not None:
            self._entities_present = entities_present

    def before_inference_disabled(self) -> None:
        """Mark inference as disabled."""
        self._inference_enabled = False

    def before_inference_enabled(
        self,
        status: str | None = None,
        entities_present: bool | None = None,
        **data,
    ) -> None:
        """Mark inference as enabled."""
        self._inference_enabled = True
        if status is not None:
            self._status = status
        if entities_present is not None:
            self._entities_present = entities_present

    def before_status_pending_nodes(
        self,
        entities_present: bool | None = None,
        **data,
    ) -> None:
        """Update status when pending_nodes reported."""
        self._status = "pending_nodes"
        if entities_present is not None:
            self._entities_present = entities_present

    def before_status_pending_edges_or_done(
        self,
        status: str | None = None,
        entities_present: bool | None = None,
        **data,
    ) -> None:
        """Update status when pending_edges or done reported."""
        if status is not None:
            self._status = status
        if entities_present is not None:
            self._entities_present = entities_present

    def before_cache_entities_found(
        self,
        status: str | None = None,
        **data,
    ) -> None:
        """Update entities flag when cache hit."""
        self._entities_present = True
        if status is not None:
            self._status = status

    def before_cache_empty(
        self,
        status: str | None = None,
        **data,
    ) -> None:
        """Update entities flag when cache empty."""
        self._entities_present = False
        if status is not None:
            self._status = status

    def before_user_deleted_entity(
        self,
        entities_present: bool | None = None,
        status: str | None = None,
        **data,
    ) -> None:
        """Handle user deletion event."""
        if entities_present is not None:
            self._entities_present = entities_present
        if status is not None:
            self._status = status
        elif entities_present is False:
            # When last entity is deleted (transitioning to empty_idle),
            # clear status to prevent stale "pending_edges"/"pending_nodes" from
            # causing EntitySidebar to show "Awaiting processing..." instead of "No connections found"
            self._status = None

    def apply_event(self, event_name: str, **data) -> SidebarOutput:
        """Apply event and return updated output.

        Args:
            event_name: Name of the event to send.
            **data: Optional data for the event (status, inference_enabled, entities_present).

        Returns:
            Updated output state after event processing.
        """
        try:
            self.send(event_name, **data)
        except Exception as e:
            logger.error(f"Error applying event {event_name}: {e}", exc_info=True)
        return self.output


def generate_diagram(output_path: str = "frontend/diagrams/sidebar_state_machine.png") -> bool:
    """Generate state machine diagram as PNG file.

    Requires python-statemachine[diagrams] extra to be installed.

    IMPORTANT: When modifying SidebarStateMachine (states, events, transitions),
    this function MUST be re-run to regenerate the diagram and prevent diagram drift.
    Run: python -c "from frontend.state.sidebar_state_machine import generate_diagram; generate_diagram()"

    Args:
        output_path: Path where PNG diagram will be written.

    Returns:
        True if diagram generated successfully, False if dependencies missing or error occurred.
    """
    try:
        machine = SidebarStateMachine()
        machine._graph().write_png(output_path)
        logger.info(f"Generated sidebar state machine diagram: {output_path}")
        return True
    except ImportError:
        logger.warning(
            "python-statemachine[diagrams] not installed. "
            "Install with: uv add 'python-statemachine[diagrams]'"
        )
        return False
    except Exception as e:
        logger.error(f"Failed to generate state machine diagram: {e}", exc_info=True)
        return False
