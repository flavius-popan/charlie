"""Tests for files importer."""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from importers.files import (
    collect_files,
    extract_content,
    get_file_date,
    parse_entries,
    parse_filename_date,
    DEFAULT_EXTENSIONS,
)


class TestCollectFiles:
    """Tests for file discovery."""

    def test_collects_txt_files(self, tmp_path):
        """Should collect .txt files from directory."""
        (tmp_path / "journal.txt").write_text("Entry 1")
        (tmp_path / "notes.txt").write_text("Entry 2")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        files = collect_files(tmp_path, extensions={"txt"}, recursive=False)

        assert len(files) == 2
        assert all(f.suffix == ".txt" for f in files)

    def test_case_insensitive_extension_matching(self, tmp_path):
        """Should match extensions regardless of case."""
        (tmp_path / "upper.TXT").write_text("Upper")
        (tmp_path / "mixed.Txt").write_text("Mixed")
        (tmp_path / "lower.txt").write_text("Lower")

        files = collect_files(tmp_path, extensions={"txt"}, recursive=False)

        assert len(files) == 3

    def test_multiple_extensions(self, tmp_path):
        """Should collect files matching any of the given extensions."""
        (tmp_path / "doc.txt").write_text("Text")
        (tmp_path / "doc.md").write_text("Markdown")
        (tmp_path / "doc.rtf").write_text("RTF")
        (tmp_path / "doc.pdf").write_bytes(b"%PDF")

        files = collect_files(tmp_path, extensions={"txt", "md", "rtf"}, recursive=False)

        assert len(files) == 3
        names = {f.name for f in files}
        assert names == {"doc.txt", "doc.md", "doc.rtf"}

    def test_skips_hidden_files(self, tmp_path):
        """Should skip files starting with a dot."""
        (tmp_path / "visible.txt").write_text("Visible")
        (tmp_path / ".hidden.txt").write_text("Hidden")

        files = collect_files(tmp_path, extensions={"txt"}, recursive=False)

        assert len(files) == 1
        assert files[0].name == "visible.txt"

    def test_skips_hidden_directories(self, tmp_path):
        """Should skip hidden directories when recursing."""
        (tmp_path / "visible").mkdir()
        (tmp_path / "visible" / "doc.txt").write_text("Visible")
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "doc.txt").write_text("Hidden")

        files = collect_files(tmp_path, extensions={"txt"}, recursive=True)

        assert len(files) == 1
        assert "visible" in str(files[0])

    def test_skips_symlinks(self, tmp_path):
        """Should skip symbolic links."""
        real_file = tmp_path / "real.txt"
        real_file.write_text("Real file")
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(real_file)

        files = collect_files(tmp_path, extensions={"txt"}, recursive=False)

        assert len(files) == 1
        assert files[0].name == "real.txt"

    def test_flat_mode_ignores_subdirectories(self, tmp_path):
        """Should only collect files from top-level directory when not recursive."""
        (tmp_path / "top.txt").write_text("Top level")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested")

        files = collect_files(tmp_path, extensions={"txt"}, recursive=False)

        assert len(files) == 1
        assert files[0].name == "top.txt"

    def test_recursive_mode_includes_subdirectories(self, tmp_path):
        """Should collect files from subdirectories when recursive."""
        (tmp_path / "top.txt").write_text("Top level")
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested")
        deep = subdir / "deep"
        deep.mkdir()
        (deep / "deeper.txt").write_text("Deep")

        files = collect_files(tmp_path, extensions={"txt"}, recursive=True)

        assert len(files) == 3

    def test_returns_sorted_by_name(self, tmp_path):
        """Should return files sorted alphabetically."""
        (tmp_path / "charlie.txt").write_text("C")
        (tmp_path / "alpha.txt").write_text("A")
        (tmp_path / "bravo.txt").write_text("B")

        files = collect_files(tmp_path, extensions={"txt"}, recursive=False)

        names = [f.name for f in files]
        assert names == ["alpha.txt", "bravo.txt", "charlie.txt"]

    def test_empty_directory_returns_empty_list(self, tmp_path):
        """Should return empty list for empty directory."""
        files = collect_files(tmp_path, extensions={"txt"}, recursive=False)

        assert files == []


class TestExtractContent:
    """Tests for content extraction from files."""

    def test_extracts_txt_content(self, tmp_path):
        """Should read .txt files as UTF-8."""
        path = tmp_path / "journal.txt"
        path.write_text("Hello, world!")

        content, error = extract_content(path)

        assert content == "Hello, world!"
        assert error is None

    def test_extracts_md_content(self, tmp_path):
        """Should read .md files as UTF-8."""
        path = tmp_path / "notes.md"
        path.write_text("# Header\n\nSome markdown.")

        content, error = extract_content(path)

        assert content == "# Header\n\nSome markdown."
        assert error is None

    def test_extracts_rtf_content(self, tmp_path):
        """Should strip RTF formatting and return plain text."""
        path = tmp_path / "document.rtf"
        # Minimal RTF with "Hello World"
        rtf_content = r"{\rtf1\ansi Hello World}"
        path.write_text(rtf_content)

        content, error = extract_content(path)

        assert "Hello World" in content
        assert r"\rtf" not in content
        assert error is None

    def test_handles_utf8_with_special_chars(self, tmp_path):
        """Should handle UTF-8 special characters."""
        path = tmp_path / "unicode.txt"
        path.write_text("Caf\u00e9 \u2014 \u00a9 2025")

        content, error = extract_content(path)

        assert content == "Caf\u00e9 \u2014 \u00a9 2025"
        assert error is None

    def test_handles_decode_failure(self, tmp_path):
        """Should return error for non-UTF-8 files."""
        path = tmp_path / "binary.txt"
        # Write invalid UTF-8 bytes
        path.write_bytes(b"\x80\x81\x82\x83")

        content, error = extract_content(path)

        assert content is None
        assert error is not None
        assert "decode" in error.lower() or "utf" in error.lower()

    def test_handles_rtf_parse_failure(self, tmp_path):
        """Should return error for malformed RTF."""
        path = tmp_path / "bad.rtf"
        path.write_text("This is not valid RTF content")

        content, error = extract_content(path)

        # striprtf may just return the text or error - either is acceptable
        # The key is it shouldn't crash
        assert content is not None or error is not None


class TestGetFileDate:
    """Tests for file date extraction."""

    def test_returns_datetime(self, tmp_path):
        """Should return a datetime object."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        result = get_file_date(path, source="created", tz=ZoneInfo("UTC"))

        assert isinstance(result, datetime)

    def test_datetime_is_timezone_aware(self, tmp_path):
        """Should return timezone-aware datetime."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        result = get_file_date(path, source="created", tz=ZoneInfo("UTC"))

        assert result.tzinfo is not None

    def test_applies_specified_timezone(self, tmp_path):
        """Should apply the specified timezone."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        result = get_file_date(path, source="created", tz=ZoneInfo("America/New_York"))

        assert result.tzinfo == ZoneInfo("America/New_York")

    def test_modified_source_uses_mtime(self, tmp_path):
        """Should use modification time when source is 'modified'."""
        path = tmp_path / "test.txt"
        path.write_text("content")
        # Set mtime to a specific time
        specific_time = time.time() - 3600  # 1 hour ago
        os.utime(path, (specific_time, specific_time))

        result = get_file_date(path, source="modified", tz=ZoneInfo("UTC"))

        # Should be close to the mtime we set
        expected = datetime.fromtimestamp(specific_time, tz=timezone.utc)
        assert abs((result - expected).total_seconds()) < 2

    def test_created_source_uses_birthtime_or_mtime(self, tmp_path):
        """Should use birthtime on macOS, mtime as fallback."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        result = get_file_date(path, source="created", tz=ZoneInfo("UTC"))

        # Just verify it returns a reasonable time (within last minute)
        now = datetime.now(tz=timezone.utc)
        assert (now - result).total_seconds() < 60

    def test_different_timezones_same_instant(self, tmp_path):
        """Same file should represent same instant in different timezones."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        utc_result = get_file_date(path, source="created", tz=ZoneInfo("UTC"))
        ny_result = get_file_date(path, source="created", tz=ZoneInfo("America/New_York"))

        # Should represent the same instant
        assert utc_result == ny_result


class TestParseEntries:
    """Tests for converting files to import entries."""

    def test_returns_content_datetime_uuid_tuples(self, tmp_path):
        """Should return list of (content, datetime, uuid) tuples."""
        (tmp_path / "test.txt").write_text("Hello world")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
        )

        assert len(entries) == 1
        content, dt, uuid = entries[0]
        assert content == "Hello world"
        assert isinstance(dt, datetime)
        assert isinstance(uuid, str)
        assert len(uuid) == 36  # Standard UUID format

    def test_skips_empty_files(self, tmp_path):
        """Should skip files with empty content."""
        (tmp_path / "empty.txt").write_text("")
        (tmp_path / "whitespace.txt").write_text("   \n\t  ")
        (tmp_path / "valid.txt").write_text("content")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
        )

        assert len(entries) == 1
        assert entries[0][0] == "content"

    def test_reports_errors_for_failed_files(self, tmp_path):
        """Should return errors for files that couldn't be parsed."""
        (tmp_path / "valid.txt").write_text("valid")
        (tmp_path / "binary.txt").write_bytes(b"\x80\x81\x82")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
        )

        assert len(entries) == 1
        assert len(errors) == 1
        assert "binary.txt" in errors[0]

    def test_uuid_is_deterministic(self, tmp_path):
        """Same file should produce same UUID across runs."""
        (tmp_path / "test.txt").write_text("content")

        entries1, _, _ = parse_entries(
            tmp_path, extensions={"txt"}, recursive=False,
            date_source="created", tz=ZoneInfo("UTC"),
        )
        entries2, _, _ = parse_entries(
            tmp_path, extensions={"txt"}, recursive=False,
            date_source="created", tz=ZoneInfo("UTC"),
        )

        assert entries1[0][2] == entries2[0][2]

    def test_different_files_get_different_uuids(self, tmp_path):
        """Different files should get different UUIDs."""
        (tmp_path / "one.txt").write_text("one")
        (tmp_path / "two.txt").write_text("two")

        entries, _, _ = parse_entries(
            tmp_path, extensions={"txt"}, recursive=False,
            date_source="created", tz=ZoneInfo("UTC"),
        )

        assert entries[0][2] != entries[1][2]


class TestDefaultExtensions:
    """Tests for default extension configuration."""

    def test_includes_common_text_formats(self):
        """Should include txt, md, rtf, markdown, text."""
        assert "txt" in DEFAULT_EXTENSIONS
        assert "md" in DEFAULT_EXTENSIONS
        assert "rtf" in DEFAULT_EXTENSIONS
        assert "markdown" in DEFAULT_EXTENSIONS
        assert "text" in DEFAULT_EXTENSIONS


class TestParseFilenameDate:
    """Tests for parsing dates from filenames."""

    def test_parses_date_only_format(self, tmp_path):
        """Should parse YYYY-MM-DD format from filename."""
        path = tmp_path / "2024-10-13.txt"
        path.write_text("content")

        result = parse_filename_date(path, "%Y-%m-%d", ZoneInfo("UTC"))

        assert result is not None
        assert result.year == 2024
        assert result.month == 10
        assert result.day == 13
        assert result.hour == 0
        assert result.minute == 0

    def test_parses_datetime_format(self, tmp_path):
        """Should parse YYYY-MM-DD-HHMM format from filename."""
        path = tmp_path / "2024-10-13-2354.txt"
        path.write_text("content")

        result = parse_filename_date(path, "%Y-%m-%d-%H%M", ZoneInfo("UTC"))

        assert result is not None
        assert result.year == 2024
        assert result.month == 10
        assert result.day == 13
        assert result.hour == 23
        assert result.minute == 54

    def test_returns_none_for_non_matching_filename(self, tmp_path):
        """Should return None when filename doesn't match format."""
        path = tmp_path / "random-notes.txt"
        path.write_text("content")

        result = parse_filename_date(path, "%Y-%m-%d", ZoneInfo("UTC"))

        assert result is None

    def test_returns_none_for_partial_match(self, tmp_path):
        """Should return None when filename only partially matches."""
        path = tmp_path / "2024-10.txt"
        path.write_text("content")

        result = parse_filename_date(path, "%Y-%m-%d", ZoneInfo("UTC"))

        assert result is None

    def test_applies_timezone(self, tmp_path):
        """Should apply specified timezone to parsed date."""
        path = tmp_path / "2024-10-13.txt"
        path.write_text("content")

        result = parse_filename_date(path, "%Y-%m-%d", ZoneInfo("America/New_York"))

        assert result is not None
        assert result.tzinfo == ZoneInfo("America/New_York")

    def test_uses_stem_ignores_extension(self, tmp_path):
        """Should parse from stem only, ignoring extension."""
        path = tmp_path / "2024-10-13.md"
        path.write_text("content")

        result = parse_filename_date(path, "%Y-%m-%d", ZoneInfo("UTC"))

        assert result is not None
        assert result.year == 2024


class TestParseEntriesWithFilenameDate:
    """Tests for parse_entries with filename_date_format parameter."""

    def test_uses_filename_date_when_format_matches(self, tmp_path):
        """Should use parsed filename date instead of file metadata."""
        path = tmp_path / "2020-01-15.txt"
        path.write_text("Old entry")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
            filename_date_format="%Y-%m-%d",
        )

        assert len(entries) == 1
        assert entries[0][1].year == 2020
        assert entries[0][1].month == 1
        assert entries[0][1].day == 15
        assert len(warnings) == 0

    def test_falls_back_to_metadata_when_format_doesnt_match(self, tmp_path):
        """Should fall back to file metadata when filename doesn't match format."""
        path = tmp_path / "random-notes.txt"
        path.write_text("Some notes")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
            filename_date_format="%Y-%m-%d",
        )

        assert len(entries) == 1
        # Date should be recent (from file metadata), not from filename
        now = datetime.now(tz=timezone.utc)
        assert (now - entries[0][1]).total_seconds() < 60
        assert len(warnings) == 1
        assert "random-notes.txt" in warnings[0]

    def test_warns_for_non_matching_filenames(self, tmp_path):
        """Should emit warning for filenames that don't match format."""
        (tmp_path / "notes.txt").write_text("content")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
            filename_date_format="%Y-%m-%d",
        )

        assert len(warnings) == 1
        assert "doesn't match format" in warnings[0]

    def test_mixed_matching_and_non_matching_files(self, tmp_path):
        """Should handle mix of matching and non-matching filenames."""
        (tmp_path / "2024-10-13.txt").write_text("Dated entry")
        (tmp_path / "random.txt").write_text("Undated entry")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
            filename_date_format="%Y-%m-%d",
        )

        assert len(entries) == 2
        assert len(warnings) == 1
        assert "random.txt" in warnings[0]

        # Find the dated entry and verify its date
        dated_entry = next(e for e in entries if "Dated" in e[0])
        assert dated_entry[1].year == 2024
        assert dated_entry[1].month == 10
        assert dated_entry[1].day == 13

    def test_datetime_format_with_time(self, tmp_path):
        """Should parse datetime format including hours and minutes."""
        (tmp_path / "2024-11-21-0316.txt").write_text("Late night entry")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
            filename_date_format="%Y-%m-%d-%H%M",
        )

        assert len(entries) == 1
        assert entries[0][1].year == 2024
        assert entries[0][1].month == 11
        assert entries[0][1].day == 21
        assert entries[0][1].hour == 3
        assert entries[0][1].minute == 16

    def test_no_warnings_when_format_not_provided(self, tmp_path):
        """Should not emit warnings when filename_date_format is None."""
        (tmp_path / "notes.txt").write_text("content")

        entries, errors, warnings = parse_entries(
            tmp_path,
            extensions={"txt"},
            recursive=False,
            date_source="created",
            tz=ZoneInfo("UTC"),
            filename_date_format=None,
        )

        assert len(entries) == 1
        assert len(warnings) == 0
