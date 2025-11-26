"""Shared utilities for journal entry importers."""

from argparse import ArgumentParser
from datetime import datetime
from uuid import uuid5, NAMESPACE_URL

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from backend import add_journal_entry
from backend.database import episode_exists
from backend.settings import DEFAULT_JOURNAL


def generate_file_uuid(source: str, filepath: str) -> str:
    """Deterministic UUID from source + filepath."""
    return str(uuid5(NAMESPACE_URL, f"{source}:{filepath}"))


def setup_argparse(description: str) -> ArgumentParser:
    """Create argument parser with common importer args."""
    parser = ArgumentParser(description=description)
    parser.add_argument("input", help="Input file path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without persisting")
    parser.add_argument("--journal", default=DEFAULT_JOURNAL, help=f"Target journal (default: {DEFAULT_JOURNAL})")
    return parser


async def import_entries(
    entries: list[tuple[str, datetime, str]],
    *,
    source: str,
    journal: str,
    dry_run: bool,
    console: Console,
) -> tuple[int, int]:
    """Import entries with progress bar.

    Args:
        entries: List of (content, datetime, uuid) tuples
        source: Source description (e.g., "Day One", "Blogger")
        journal: Target journal name
        dry_run: If True, skip actual persistence
        console: Rich console for output

    Returns:
        (imported_count, skipped_count)
    """
    imported = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Importing entries...", total=len(entries))

        for content, entry_time, uuid in entries:
            # Check existence for accurate skip counting
            if await episode_exists(uuid, journal):
                skipped += 1
            else:
                if not dry_run:
                    await add_journal_entry(
                        content,
                        reference_time=entry_time,
                        journal=journal,
                        source_description=source,
                        uuid=uuid,
                    )
                imported += 1

            progress.advance(task)

    return imported, skipped
