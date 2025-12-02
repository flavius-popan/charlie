"""Tests for DeleteEntityModal widget."""

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, RadioButton, RadioSet

from frontend.widgets import DeleteEntityModal, DeleteEntityResult


class DeleteModalTestApp(App):
    """Test app for DeleteEntityModal."""

    def __init__(self, entity_name: str, default_scope: str, checkbox_default: bool = True):
        super().__init__()
        self.entity_name = entity_name
        self.default_scope = default_scope
        self.checkbox_default = checkbox_default

    def compose(self) -> ComposeResult:
        return []

    def action_show_modal(self) -> None:
        """Action to show the delete modal."""
        modal = DeleteEntityModal(
            entity_name=self.entity_name,
            default_scope=self.default_scope,
            checkbox_default=self.checkbox_default,
        )
        self.push_screen(modal)


@pytest.mark.asyncio
async def test_delete_modal_defaults_scope_entry():
    """When default_scope='entry', first radio button should be pressed."""
    app = DeleteModalTestApp("Alice", default_scope="entry")

    async with app.run_test() as pilot:
        app.action_show_modal()
        await pilot.pause()

        modal = app.screen
        radio_set = modal.query_one("#delete-scope", RadioSet)
        assert radio_set.pressed_index == 0


@pytest.mark.asyncio
async def test_delete_modal_defaults_scope_all():
    """When default_scope='all', second radio button should be selected."""
    app = DeleteModalTestApp("Bob", default_scope="all")

    async with app.run_test() as pilot:
        app.action_show_modal()
        await pilot.pause()

        modal = app.screen
        radio_set = modal.query_one("#delete-scope", RadioSet)
        assert radio_set.pressed_index == 1


@pytest.mark.asyncio
async def test_delete_modal_checkbox_default_true():
    """Checkbox should be checked by default when checkbox_default=True."""
    app = DeleteModalTestApp("Charlie", default_scope="entry", checkbox_default=True)

    async with app.run_test() as pilot:
        app.action_show_modal()
        await pilot.pause()

        modal = app.screen
        checkbox = modal.query_one("#delete-block", Checkbox)
        assert checkbox.value is True


@pytest.mark.asyncio
async def test_delete_modal_checkbox_default_false():
    """Checkbox should be unchecked when checkbox_default=False."""
    app = DeleteModalTestApp("Diana", default_scope="entry", checkbox_default=False)

    async with app.run_test() as pilot:
        app.action_show_modal()
        await pilot.pause()

        modal = app.screen
        checkbox = modal.query_one("#delete-block", Checkbox)
        assert checkbox.value is False


@pytest.mark.asyncio
async def test_delete_modal_confirm_returns_correct_result():
    """Clicking Remove should dismiss modal with confirmed=True."""
    app = DeleteModalTestApp("Eve", default_scope="all", checkbox_default=True)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Eve", default_scope="all", checkbox_default=True)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        modal_screen = app.screen
        confirm_button = modal_screen.query_one("#confirm", Button)
        confirm_button.press()

        await asyncio.sleep(0.1)
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert isinstance(result, DeleteEntityResult)
        assert result.confirmed is True
        assert result.scope == "all"
        assert result.block_future is True


@pytest.mark.asyncio
async def test_delete_modal_confirm_with_entry_scope():
    """Clicking Remove with entry scope should return correct scope value."""
    app = DeleteModalTestApp("Frank", default_scope="entry", checkbox_default=False)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Frank", default_scope="entry", checkbox_default=False)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        modal_screen = app.screen
        confirm_button = modal_screen.query_one("#confirm", Button)
        confirm_button.press()

        await asyncio.sleep(0.1)
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert result.confirmed is True
        assert result.scope == "entry"
        assert result.block_future is False


@pytest.mark.asyncio
async def test_delete_modal_cancel_returns_not_confirmed():
    """Clicking Cancel should return DeleteEntityResult with confirmed=False."""
    app = DeleteModalTestApp("Grace", default_scope="entry", checkbox_default=True)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Grace", default_scope="entry", checkbox_default=True)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        modal_screen = app.screen
        cancel_button = modal_screen.query_one("#cancel", Button)
        cancel_button.press()

        await asyncio.sleep(0.1)
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert isinstance(result, DeleteEntityResult)
        assert result.confirmed is False


@pytest.mark.asyncio
async def test_delete_modal_escape_returns_not_confirmed():
    """Pressing Escape should return DeleteEntityResult with confirmed=False."""
    app = DeleteModalTestApp("Henry", default_scope="all", checkbox_default=True)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Henry", default_scope="all", checkbox_default=True)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert isinstance(result, DeleteEntityResult)
        assert result.confirmed is False


@pytest.mark.asyncio
async def test_delete_modal_displays_entity_name():
    """Title should display the entity name in quotes."""
    entity_name = "Isabella"
    app = DeleteModalTestApp(entity_name, default_scope="entry")

    async with app.run_test() as pilot:
        app.action_show_modal()
        await pilot.pause()

        modal = app.screen
        title_label = modal.query_one("#delete-title")
        title_text = title_label.render().plain

        assert f'"{entity_name}"' in title_text
        assert "Remove" in title_text


@pytest.mark.asyncio
async def test_delete_modal_cancel_preserves_modal_state():
    """Canceling should return result reflecting current modal state."""
    app = DeleteModalTestApp("Jack", default_scope="all", checkbox_default=False)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Jack", default_scope="all", checkbox_default=False)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        modal_screen = app.screen
        cancel_button = modal_screen.query_one("#cancel", Button)
        cancel_button.press()

        await asyncio.sleep(0.1)
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert result.confirmed is False
        assert result.scope == "all"
        assert result.block_future is False


@pytest.mark.asyncio
async def test_delete_modal_escape_preserves_modal_state():
    """Pressing Escape should return result reflecting current modal state."""
    app = DeleteModalTestApp("Kate", default_scope="entry", checkbox_default=True)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Kate", default_scope="entry", checkbox_default=True)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert result.confirmed is False
        assert result.scope == "entry"
        assert result.block_future is True


@pytest.mark.asyncio
async def test_delete_modal_change_scope_before_confirm():
    """Changing radio selection before confirming should be reflected in result."""
    app = DeleteModalTestApp("Liam", default_scope="entry", checkbox_default=True)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Liam", default_scope="entry", checkbox_default=True)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        modal_screen = app.screen
        radio_all = modal_screen.query_one("#radio-all", RadioButton)
        radio_all.value = True
        await pilot.pause()

        confirm_button = modal_screen.query_one("#confirm", Button)
        confirm_button.press()

        await asyncio.sleep(0.1)
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert result.confirmed is True
        assert result.scope == "all"


@pytest.mark.asyncio
async def test_delete_modal_change_checkbox_before_confirm():
    """Changing checkbox before confirming should be reflected in result."""
    app = DeleteModalTestApp("Maya", default_scope="entry", checkbox_default=False)
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Maya", default_scope="entry", checkbox_default=False)
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        modal_screen = app.screen
        checkbox = modal_screen.query_one("#delete-block", Checkbox)
        checkbox.value = True
        await pilot.pause()

        confirm_button = modal_screen.query_one("#confirm", Button)
        confirm_button.press()

        await asyncio.sleep(0.1)
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert result.confirmed is True
        assert result.block_future is True


@pytest.mark.asyncio
async def test_delete_modal_show_scope_false_hides_radio():
    """When show_scope=False, scope radio should not be present."""
    app = DeleteModalTestApp("Noah", default_scope="all")

    async with app.run_test() as pilot:
        modal = DeleteEntityModal(
            "Noah", default_scope="all", checkbox_default=True, show_scope=False
        )
        app.push_screen(modal)
        await pilot.pause()

        modal_screen = app.screen
        # Scope radio should not exist
        radio_sets = modal_screen.query("#delete-scope")
        assert len(list(radio_sets)) == 0, "Scope radio should be hidden"

        # Checkbox should still exist
        checkbox = modal_screen.query_one("#delete-block", Checkbox)
        assert checkbox.value is True


@pytest.mark.asyncio
async def test_delete_modal_show_scope_false_returns_default_scope():
    """When show_scope=False, result should use default_scope value."""
    app = DeleteModalTestApp("Olivia", default_scope="all")
    result_holder = {}

    async def on_result(result: DeleteEntityResult | None) -> None:
        result_holder["result"] = result

    async with app.run_test() as pilot:
        modal = DeleteEntityModal(
            "Olivia", default_scope="all", checkbox_default=True, show_scope=False
        )
        app.push_screen(modal, callback=on_result)
        await pilot.pause()

        modal_screen = app.screen
        confirm_button = modal_screen.query_one("#confirm", Button)
        confirm_button.press()

        await asyncio.sleep(0.1)
        await pilot.pause()

        result = result_holder.get("result")
        assert result is not None
        assert result.confirmed is True
        assert result.scope == "all", "Should use default_scope when radio is hidden"
        assert result.block_future is True


@pytest.mark.asyncio
async def test_delete_modal_show_scope_true_by_default():
    """show_scope should default to True, showing the radio."""
    app = DeleteModalTestApp("Peter", default_scope="entry")

    async with app.run_test() as pilot:
        # Use default show_scope (True)
        modal = DeleteEntityModal("Peter", default_scope="entry", checkbox_default=True)
        app.push_screen(modal)
        await pilot.pause()

        modal_screen = app.screen
        # Scope radio should exist
        radio_set = modal_screen.query_one("#delete-scope", RadioSet)
        assert radio_set is not None, "Scope radio should be visible by default"


@pytest.mark.asyncio
async def test_delete_modal_left_right_navigation():
    """Left/right arrow keys should navigate between Cancel and Delete buttons."""
    app = DeleteModalTestApp("Quinn", default_scope="entry")

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Quinn", default_scope="entry", checkbox_default=True)
        app.push_screen(modal)
        await pilot.pause()

        modal_screen = app.screen
        cancel_btn = modal_screen.query_one("#cancel", Button)
        confirm_btn = modal_screen.query_one("#confirm", Button)

        # Focus the Cancel button first
        cancel_btn.focus()
        await pilot.pause()
        assert cancel_btn.has_focus, "Cancel button should have focus"

        # Press right to move to Confirm button
        await pilot.press("right")
        await pilot.pause()
        assert confirm_btn.has_focus, (
            "Right arrow should move focus to Confirm button. "
            "Bug: Binding uses 'focus_next' instead of 'app.focus_next'"
        )


@pytest.mark.asyncio
async def test_delete_modal_vim_navigation():
    """Vim keys (h/l) should navigate between buttons."""
    app = DeleteModalTestApp("Quinn", default_scope="entry")

    async with app.run_test() as pilot:
        modal = DeleteEntityModal("Quinn", default_scope="entry", checkbox_default=True)
        app.push_screen(modal)
        await pilot.pause()

        modal_screen = app.screen
        cancel_btn = modal_screen.query_one("#cancel", Button)
        confirm_btn = modal_screen.query_one("#confirm", Button)

        cancel_btn.focus()
        await pilot.pause()
        assert cancel_btn.has_focus

        await pilot.press("l")
        await pilot.pause()
        assert confirm_btn.has_focus, "'l' should move focus to next button"

        await pilot.press("h")
        await pilot.pause()
        assert cancel_btn.has_focus, "'h' should move focus to previous button"
