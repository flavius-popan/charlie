#!/usr/bin/env python
"""Day One JSON export importer.

Usage:
    python importers/dayone.py export.zip [--dry-run] [--journal NAME]
    python importers/dayone.py export.json [--dry-run] [--journal NAME]
"""

import sys
from pathlib import Path

# Enable direct script execution: python importers/dayone.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import json
import re
import zipfile
from datetime import datetime, timezone

from rich.console import Console

from importers.utils import setup_argparse, import_entries


def convert_dayone_uuid(hex_uuid: str) -> str:
    """Convert Day One 32-char hex UUID to standard format.

    Day One: 6F30404AB159433D8C9AF57052E4F3B6
    Standard: 6F30404A-B159-433D-8C9A-F57052E4F3B6
    """
    h = hex_uuid.upper()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


# Markdown cleaners: (pattern, replacement)
# Applied in order during import
MARKDOWN_CLEANERS = [
    # Day One image references use custom URL scheme - strip before unescape
    (r"!\[\]\(dayone-moment://[^)]+\)", ""),
    # Day One escapes markdown special chars with backslashes: \. \! \# \* \, etc.
    (r"\\([.!#*_\[\](){}+\-|`~<>,])", r"\1"),
]


def clean_dayone_markdown(text: str) -> str:
    """Apply all markdown cleaners to normalize Day One content."""
    for pattern, replacement in MARKDOWN_CLEANERS:
        text = re.sub(pattern, replacement, text)
    return text


def parse_dayone_date(iso_string: str) -> datetime:
    """Parse Day One ISO 8601 date string to UTC datetime."""
    # Day One uses format: 2025-11-26T18:50:20Z
    dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    return dt.astimezone(timezone.utc)


def load_dayone_json(path: Path) -> tuple[list[dict], str]:
    """Load Day One export, handling both .zip and .json.

    Returns:
        (entries list, journal_name from filename)
    """
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            json_files = [n for n in zf.namelist() if n.endswith(".json")]
            if not json_files:
                raise ValueError(f"No JSON file found in {path}")
            json_name = json_files[0]
            journal_name = Path(json_name).stem
            with zf.open(json_name) as f:
                data = json.load(f)
    else:
        journal_name = path.stem
        with open(path) as f:
            data = json.load(f)

    return data.get("entries", []), journal_name


def parse_entries(raw_entries: list[dict], source_journal: str) -> list[tuple[str, datetime, str]]:
    """Convert Day One entries to import format.

    Returns:
        List of (content, datetime, uuid) tuples
    """
    entries = []
    for entry in raw_entries:
        uuid = convert_dayone_uuid(entry["uuid"])
        content = clean_dayone_markdown(entry.get("text", ""))
        entry_time = parse_dayone_date(entry["creationDate"])

        if not content.strip():
            continue

        entries.append((content, entry_time, uuid))

    return entries


async def main():
    parser = setup_argparse("Import journal entries from Day One JSON export")
    args = parser.parse_args()

    console = Console()
    input_path = Path(args.input)

    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        sys.exit(1)

    raw_entries, source_journal = load_dayone_json(input_path)
    entries = parse_entries(raw_entries, source_journal)

    if not entries:
        console.print("[yellow]No entries found to import[/yellow]")
        sys.exit(0)

    console.print(f"Found {len(entries)} entries from Day One: {source_journal}")

    if args.dry_run:
        console.print("[yellow]Dry run - no changes will be made[/yellow]")

    source_desc = f"Day One: {source_journal}"
    imported, skipped = await import_entries(
        entries,
        source=source_desc,
        journal=args.journal,
        dry_run=args.dry_run,
        console=console,
    )

    action = "Would import" if args.dry_run else "Imported"
    console.print(f"\n[green]{action}: {imported}, Skipped: {skipped}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
