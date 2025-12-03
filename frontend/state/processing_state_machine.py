"""Processing state machine for managing processing pane display.

This module provides a pure logic state machine (no Textual dependencies) that models
the processing pane visibility and status display. The machine outputs flags and text
that UI components use for rendering.
"""

import logging
from dataclasses import dataclass
from statemachine import StateMachine, State

logger = logging.getLogger("charlie")


@dataclass
class ProcessingOutput:
    """Output state from the processing state machine."""

    pane_visible: bool
    """Whether processing pane should be displayed."""

    status_text: str
    """Status text: 'Loading model...', 'Extracting:', 'Finishing extracting:', etc."""

    queue_text: str
    """Queue text: 'Queue: N remaining' or empty string."""

    show_dot: bool
    """Whether to show animated ProcessingDot."""

    poll_interval: float
    """How fast to poll: 0.3s when active, 2.0s when idle."""

    is_inferring: bool
    """Whether currently in inferring state."""

    active_episode_uuid: str | None
    """For connections pane / entry dots / name resolution."""

    inference_enabled: bool
    """For status_text and downstream consumers."""

    @classmethod
    def idle(cls) -> "ProcessingOutput":
        """Create output for idle state with default values."""
        return cls(
            pane_visible=False,
            status_text="",
            queue_text="",
            show_dot=False,
            poll_interval=2.0,
            is_inferring=False,
            active_episode_uuid=None,
            inference_enabled=True,
        )


class ProcessingStateMachine(StateMachine):
    """State machine for processing pane display.

    This pure-logic state machine (no Textual dependencies) models the processing
    pane's visibility and status, producing output flags for UI consumption.

    States:
    - idle: Model not loaded, no active processing. Pane hidden.
    - loading: Model is loading into memory. Pane visible, shows "Loading model..."
    - inferring: Model is actively extracting. Pane visible, shows "Extracting:" or "Finishing extracting:"
    - unloading: Model is unloading from memory. Pane visible, shows "Unloading model..."

    The machine is a passive follower of the worker's state - it doesn't initiate
    transitions, it just reflects what get_processing_status() reports.
    """

    # States
    idle = State(initial=True)
    loading = State()
    inferring = State()
    unloading = State()

    # Events with self-transitions for idempotent polling
    status_idle = idle.to(idle) | loading.to(idle) | inferring.to(idle) | unloading.to(idle)
    status_loading = idle.to(loading) | loading.to(loading) | inferring.to(loading) | unloading.to(loading)
    status_inferring = idle.to(inferring) | loading.to(inferring) | inferring.to(inferring) | unloading.to(inferring)
    status_unloading = idle.to(unloading) | loading.to(unloading) | inferring.to(unloading) | unloading.to(unloading)

    def __init__(self):
        super().__init__()
        self._active_episode_uuid: str | None = None
        self._queue_count: int = 0
        self._inference_enabled: bool = True
        self._model_loading_blocked: bool = False
        self._retry_count: int = 0
        self._max_retries: int = 3

    @property
    def output(self) -> ProcessingOutput:
        """Compute output based on current state and internal data."""
        state_name = self.current_state.id

        # Status text with variants for finishing/retrying
        if state_name == "idle":
            status_text = ""
        elif state_name == "loading":
            status_text = "Loading model..."
        elif state_name == "inferring":
            if self._retry_count > 0:
                status_text = f"Retry {self._retry_count} of {self._max_retries}:"
            elif not self._inference_enabled:
                status_text = "Finishing extracting:"
            else:
                status_text = "Extracting:"
        elif state_name == "unloading":
            status_text = "Unloading model..."
        else:
            status_text = ""

        is_active = state_name != "idle"
        has_pending_work = self._queue_count > 0

        # Show pane when actively processing, or when idle with pending work
        # that will actually be processed (inference enabled, not blocked)
        can_process_pending = (
            has_pending_work
            and self._inference_enabled
            and not self._model_loading_blocked
        )
        return ProcessingOutput(
            pane_visible=is_active or can_process_pending,
            status_text=status_text,
            queue_text=f"Queue: {self._queue_count} remaining" if self._queue_count > 0 else "",
            show_dot=(state_name in ("loading", "inferring", "unloading")),
            poll_interval=0.3 if is_active else (0.5 if has_pending_work else 2.0),
            is_inferring=(state_name == "inferring"),
            active_episode_uuid=self._active_episode_uuid,
            inference_enabled=self._inference_enabled,
        )

    def apply_status(self, status: dict) -> ProcessingOutput:
        """Apply polling status and return output.

        Args:
            status: Dict from get_processing_status() with keys:
                - model_state: "idle", "loading", "inferring", "unloading"
                - active_uuid: str | None
                - pending_count: int
                - inference_enabled: bool

        Returns:
            ProcessingOutput with all computed display values
        """
        # Update internal data with defensive defaults (handle malformed input)
        self._active_episode_uuid = status.get("active_uuid")
        self._queue_count = status.get("pending_count", 0)
        self._inference_enabled = status.get("inference_enabled", True)
        self._model_loading_blocked = status.get("model_loading_blocked", False)
        self._retry_count = status.get("retry_count", 0)
        self._max_retries = status.get("max_retries", 3)

        # Route to appropriate state based on model_state (default to idle)
        model_state = status.get("model_state", "idle")
        if model_state == "idle":
            self.send("status_idle")
        elif model_state == "loading":
            self.send("status_loading")
        elif model_state == "inferring":
            self.send("status_inferring")
        elif model_state == "unloading":
            self.send("status_unloading")
        else:
            self.send("status_idle")  # Unknown state defaults to idle

        return self.output
