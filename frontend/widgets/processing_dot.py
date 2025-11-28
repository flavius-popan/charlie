"""Single animated dot indicator for processing state."""

from __future__ import annotations

from time import time
from typing import TYPE_CHECKING

from rich.style import Style
from rich.text import Text
from textual.color import Gradient
from textual.widget import Widget

if TYPE_CHECKING:
    from textual.app import RenderResult


class ProcessingDot(Widget):
    """Single animated dot that cycles through theme accent colors.

    Modeled on Textual's LoadingIndicator but with a single dot.
    Uses dynamic theme colors via self.colors (from CSS color: $accent).
    """

    DEFAULT_CSS = """
    ProcessingDot {
        width: 2;
        height: 1;
        content-align: left middle;
        color: $accent;
    }
    """

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(name=name, id=id, classes=classes)
        self._start_time: float = 0.0

    def on_mount(self) -> None:
        self._start_time = time()
        self.auto_refresh = 1 / 16  # ~16fps

    def render(self) -> RenderResult:
        # Static fallback when animations disabled
        if self.app.animation_level == "none":
            return Text("\u25cf ")

        elapsed = time() - self._start_time
        speed = 0.8
        dot = "\u25cf"

        # Get colors from theme via CSS (same pattern as LoadingIndicator)
        _, _, background, color = self.colors

        gradient = Gradient(
            (0.0, background.blend(color, 0.1)),
            (0.7, color),
            (1.0, color.lighten(0.1)),
        )

        blend = (elapsed * speed) % 1
        gradient_color = gradient.get_color((1 - blend) ** 2)

        return Text(f"{dot} ", Style.from_color(gradient_color.rich_color))
