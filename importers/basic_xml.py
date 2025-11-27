#!/usr/bin/env python
"""Basic XML importer for date/post structured journals.

Usage:
    python importers/basic_xml.py corpus.xml [--timezone America/New_York] [--dry-run] [--journal NAME]

Expected XML format (e.g., blog authorship corpus):
    <Root>
        <date>26,November,2025</date>
        <post>Entry content here...</post>
        <date>25,November,2025</date>
        <post>Another entry...</post>
    </Root>
"""

import sys
from pathlib import Path

# Enable direct script execution: python importers/basic_xml.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from rich.console import Console

from importers.utils import setup_argparse, import_entries, generate_file_uuid


def parse_date(date_str: str, tz: ZoneInfo) -> datetime:
    """Parse date format: DD,Month,YYYY -> UTC datetime."""
    dt = datetime.strptime(date_str.strip(), "%d,%B,%Y")
    local_dt = dt.replace(tzinfo=tz)
    return local_dt.astimezone(timezone.utc)


def parse_xml(path: Path, tz: ZoneInfo) -> list[tuple[str, datetime, str]]:
    """Parse XML file with <date>/<post> structure."""
    tree = ET.parse(path)
    root = tree.getroot()

    entries = []
    current_date = None
    post_index = 0
    filepath_str = str(path.resolve())

    for elem in root:
        if elem.tag == "date":
            current_date = parse_date(elem.text, tz)
        elif elem.tag == "post":
            if current_date is None:
                continue

            content = (elem.text or "").strip()
            if not content:
                continue

            uuid = generate_file_uuid("basic_xml", f"{filepath_str}:{post_index}")
            entries.append((content, current_date, uuid))
            post_index += 1

    return entries


def get_local_timezone() -> tuple[datetime.tzinfo, str]:
    """Get local timezone, returns (tzinfo, display_name)."""
    local_dt = datetime.now().astimezone()
    local_tz = local_dt.tzinfo
    # Try to get IANA name, fall back to offset string
    tz_name = getattr(local_tz, "key", None) or local_dt.strftime("%Z")
    return local_tz, tz_name


async def main():
    parser = setup_argparse("Import journal entries from XML with date/post structure")
    parser.add_argument("--timezone", help="Timezone for entries (default: system timezone)")
    args = parser.parse_args()

    console = Console()
    input_path = Path(args.input)

    if not input_path.exists():
        console.print(f"[red]File not found: {input_path}[/red]")
        sys.exit(1)

    if args.timezone:
        try:
            tz = ZoneInfo(args.timezone)
        except Exception as e:
            console.print(f"[red]Invalid timezone: {args.timezone} ({e})[/red]")
            sys.exit(1)
    else:
        tz, tz_name = get_local_timezone()
        console.print(f"[dim]Using system timezone: {tz_name} (use --timezone to override)[/dim]")

    entries = parse_xml(input_path, tz)

    if not entries:
        console.print("[yellow]No entries found to import[/yellow]")
        sys.exit(0)

    console.print(f"Found {len(entries)} entries")

    if args.dry_run:
        console.print("[yellow]Dry run - no changes will be made[/yellow]")

    imported, skipped = await import_entries(
        entries,
        source="XML Import",
        journal=args.journal,
        dry_run=args.dry_run,
        console=console,
    )

    action = "Would import" if args.dry_run else "Imported"
    console.print(f"\n[green]{action}: {imported}, Skipped: {skipped}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
