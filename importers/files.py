#!/usr/bin/env python
"""File-based importer for text files (.txt, .md, .rtf, etc.)

Usage:
    python importers/files.py ~/journals/ [--dry-run] [--recursive]
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

# Enable direct script execution: python importers/files.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
from argparse import ArgumentParser

from rich.console import Console
from striprtf.striprtf import rtf_to_text

from importers.utils import generate_file_uuid, import_entries
from backend.settings import DEFAULT_JOURNAL

DEFAULT_EXTENSIONS = {"txt", "md", "rtf", "markdown", "text"}


def _is_hidden(path: Path, base: Path) -> bool:
    """Check if path or any parent (relative to base) is hidden."""
    if path.name.startswith("."):
        return True
    # Check parent directories between path and base
    try:
        relative = path.relative_to(base)
        for part in relative.parts[:-1]:  # Exclude filename, already checked
            if part.startswith("."):
                return True
    except ValueError:
        pass
    return False


def collect_files(
    directory: Path,
    extensions: set[str],
    recursive: bool,
) -> list[Path]:
    """Collect matching files from directory.

    Args:
        directory: Directory to scan
        extensions: Set of extensions to include (without dots)
        recursive: Whether to recurse into subdirectories

    Returns:
        List of matching file paths, sorted alphabetically
    """
    files = []
    pattern = "**/*" if recursive else "*"

    for path in directory.glob(pattern):
        # Skip non-files, symlinks, and hidden files/directories
        if not path.is_file():
            continue
        if path.is_symlink():
            continue
        if _is_hidden(path, directory):
            continue

        ext = path.suffix.lstrip(".").lower()
        if ext in extensions:
            files.append(path)

    return sorted(files)


def extract_content(path: Path) -> tuple[str | None, str | None]:
    """Extract text content from a file.

    Args:
        path: Path to the file

    Returns:
        Tuple of (content, error). On success, content is the text and error is None.
        On failure, content is None and error describes the problem.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        return None, f"UTF-8 decode error: {e}"

    # Strip RTF formatting if needed
    if path.suffix.lower() == ".rtf":
        try:
            return rtf_to_text(raw), None
        except Exception as e:
            return None, f"RTF parse error: {e}"

    return raw, None


def parse_filename_date(path: Path, fmt: str, tz: ZoneInfo) -> datetime | None:
    """Parse date from filename using strptime format.

    Args:
        path: Path to the file
        fmt: strptime format string (e.g., "%Y-%m-%d" or "%Y-%m-%d-%H%M")
        tz: Timezone to apply

    Returns:
        Timezone-aware datetime if parsing succeeds, None otherwise
    """
    stem = path.stem
    try:
        naive_dt = datetime.strptime(stem, fmt)
        return naive_dt.replace(tzinfo=tz)
    except ValueError:
        return None


def get_file_date(path: Path, source: str, tz: ZoneInfo) -> datetime:
    """Get file date from metadata.

    Args:
        path: Path to the file
        source: "created" or "modified"
        tz: Timezone to apply

    Returns:
        Timezone-aware datetime
    """
    stat = path.stat()

    if source == "modified":
        timestamp = stat.st_mtime
    else:  # "created"
        # Use birthtime on macOS, fall back to mtime on Linux
        if hasattr(stat, "st_birthtime"):
            timestamp = stat.st_birthtime
        else:
            timestamp = stat.st_mtime

    # Create UTC datetime then convert to target timezone
    dt_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt_utc.astimezone(tz)


def parse_entries(
    directory: Path,
    extensions: set[str],
    recursive: bool,
    date_source: str,
    tz: ZoneInfo,
    filename_date_format: str | None = None,
) -> tuple[list[tuple[str, datetime, str]], list[str], list[str]]:
    """Parse files from directory into import entries.

    Args:
        directory: Directory to scan
        extensions: File extensions to include
        recursive: Whether to recurse into subdirectories
        date_source: "created" or "modified" (fallback when filename parsing fails)
        tz: Timezone to apply to file dates
        filename_date_format: Optional strptime format to parse dates from filenames

    Returns:
        Tuple of (entries, errors, warnings) where:
        - entries is list of (content, datetime, uuid) tuples
        - errors is list of error messages for files that couldn't be parsed
        - warnings is list of warning messages (e.g., filename date parse failures)
    """
    files = collect_files(directory, extensions, recursive)
    entries = []
    errors = []
    warnings = []

    for path in files:
        content, error = extract_content(path)

        if error:
            errors.append(f"{path.name}: {error}")
            continue

        # Skip empty/whitespace-only content
        if not content or not content.strip():
            continue

        # Try filename date first if format provided, fall back to file metadata
        entry_time = None
        if filename_date_format:
            entry_time = parse_filename_date(path, filename_date_format, tz)
            if entry_time is None:
                warnings.append(f"{path.name}: filename doesn't match format, using file metadata")

        if entry_time is None:
            entry_time = get_file_date(path, date_source, tz)

        uuid = generate_file_uuid("files", str(path))
        entries.append((content, entry_time, uuid))

    return entries, errors, warnings


def get_local_timezone() -> tuple[ZoneInfo, str]:
    """Get local timezone, returns (tzinfo, display_name)."""
    local_dt = datetime.now().astimezone()
    local_tz = local_dt.tzinfo
    tz_name = getattr(local_tz, "key", None) or local_dt.strftime("%Z")
    return local_tz, tz_name


def setup_argparse() -> ArgumentParser:
    """Create argument parser for files importer."""
    parser = ArgumentParser(
        description="Import journal entries from a directory of text files"
    )
    parser.add_argument("input", help="Input directory path")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be imported without persisting",
    )
    parser.add_argument(
        "--journal",
        default=DEFAULT_JOURNAL,
        help=f"Target journal (default: {DEFAULT_JOURNAL})",
    )
    parser.add_argument(
        "--timezone",
        help="Timezone for file timestamps (default: system timezone)",
    )
    parser.add_argument(
        "--date-source",
        choices=["created", "modified"],
        default="created",
        help="File metadata to use for date (default: created)",
    )
    parser.add_argument(
        "--filename-date",
        metavar="FORMAT",
        help="Parse date from filename using strptime format (e.g., '%%Y-%%m-%%d' or '%%Y-%%m-%%d-%%H%%M')",
    )
    parser.add_argument(
        "--extensions",
        help=f"Comma-separated extensions to include (default: {','.join(sorted(DEFAULT_EXTENSIONS))})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )
    return parser


async def main():
    parser = setup_argparse()
    args = parser.parse_args()

    console = Console()
    input_path = Path(args.input)

    # Validate input directory
    if not input_path.exists():
        console.print(f"[red]Directory not found: {input_path}[/red]")
        sys.exit(1)
    if not input_path.is_dir():
        console.print(f"[red]Not a directory: {input_path}[/red]")
        sys.exit(1)

    # Parse timezone
    if args.timezone:
        try:
            tz = ZoneInfo(args.timezone)
            tz_name = args.timezone
        except Exception as e:
            console.print(f"[red]Invalid timezone: {args.timezone} ({e})[/red]")
            sys.exit(1)
    else:
        tz, tz_name = get_local_timezone()

    console.print(f"[dim]Using timezone: {tz_name}[/dim]")

    if args.filename_date:
        console.print(f"[dim]Parsing dates from filenames: {args.filename_date}[/dim]")

    # Parse extensions
    if args.extensions:
        extensions = {ext.strip().lower() for ext in args.extensions.split(",")}
    else:
        extensions = DEFAULT_EXTENSIONS

    console.print(f"Scanning {input_path.name}...")

    # Parse entries
    entries, errors, warnings = parse_entries(
        input_path,
        extensions=extensions,
        recursive=args.recursive,
        date_source=args.date_source,
        tz=tz,
        filename_date_format=args.filename_date,
    )

    # Report errors and warnings
    for error in errors:
        console.print(f"[yellow]Warning: {error}[/yellow]")
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")

    if not entries:
        console.print("[yellow]No entries found to import[/yellow]")
        sys.exit(0)

    console.print(f"Found {len(entries)} entries in {input_path.name}")

    if args.dry_run:
        console.print("[yellow]Dry run - no changes will be made[/yellow]")

    source_desc = f"Files: {input_path.name}"
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
