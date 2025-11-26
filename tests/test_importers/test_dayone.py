"""Tests for Day One importer."""

from datetime import datetime, timezone

import pytest

from importers.dayone import (
    convert_dayone_uuid,
    clean_dayone_markdown,
    parse_dayone_date,
    load_dayone_json,
    parse_entries,
)


class TestConvertDayoneUuid:
    """Tests for UUID format conversion."""

    def test_converts_32_hex_to_standard_format(self):
        result = convert_dayone_uuid("6F30404AB159433D8C9AF57052E4F3B6")
        assert result == "6F30404A-B159-433D-8C9A-F57052E4F3B6"

    def test_handles_lowercase_input(self):
        result = convert_dayone_uuid("6f30404ab159433d8c9af57052e4f3b6")
        assert result == "6F30404A-B159-433D-8C9A-F57052E4F3B6"

    def test_preserves_all_characters(self):
        input_uuid = "AAAABBBBCCCCDDDDEEEEFFFFGGGGAAAA"
        result = convert_dayone_uuid(input_uuid)
        # Remove dashes and compare
        assert result.replace("-", "") == input_uuid.upper()


class TestCleanDayoneMarkdown:
    """Tests for markdown escape cleaning."""

    def test_removes_escaped_periods(self):
        result = clean_dayone_markdown(r"Hello\. World\.")
        assert result == "Hello. World."

    def test_removes_escaped_hash(self):
        result = clean_dayone_markdown(r"\# Not a header")
        assert result == "# Not a header"

    def test_removes_escaped_asterisks(self):
        result = clean_dayone_markdown(r"\*not bold\*")
        assert result == "*not bold*"

    def test_removes_escaped_underscores(self):
        result = clean_dayone_markdown(r"\_not italic\_")
        assert result == "_not italic_"

    def test_removes_escaped_brackets(self):
        result = clean_dayone_markdown(r"\[not a link\]")
        assert result == "[not a link]"

    def test_removes_escaped_parens(self):
        result = clean_dayone_markdown(r"\(not a link url\)")
        assert result == "(not a link url)"

    def test_preserves_normal_backslashes(self):
        result = clean_dayone_markdown(r"path\\to\\file")
        assert result == r"path\\to\\file"

    def test_handles_mixed_content(self):
        result = clean_dayone_markdown(r"# Header\. List: 1\. 2\. 3\.")
        assert result == "# Header. List: 1. 2. 3."

    def test_strips_dayone_image_references(self):
        result = clean_dayone_markdown("Before ![](dayone-moment://DE9D21695B6B480BBC86D42E861AC857) After")
        assert result == "Before  After"

    def test_strips_multiple_image_references(self):
        result = clean_dayone_markdown(
            "![](dayone-moment://AAA) text ![](dayone-moment://BBB)"
        )
        assert result == " text "

    def test_preserves_normal_markdown_images(self):
        result = clean_dayone_markdown("![alt](https://example.com/image.png)")
        assert result == "![alt](https://example.com/image.png)"


class TestParseDayoneDate:
    """Tests for ISO date parsing."""

    def test_parses_utc_date(self):
        result = parse_dayone_date("2025-11-26T18:50:20Z")
        assert result == datetime(2025, 11, 26, 18, 50, 20, tzinfo=timezone.utc)

    def test_result_is_utc(self):
        result = parse_dayone_date("2025-01-01T00:00:00Z")
        assert result.tzinfo == timezone.utc

    def test_handles_different_times(self):
        result = parse_dayone_date("2024-06-15T09:30:45Z")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 9
        assert result.minute == 30
        assert result.second == 45


class TestLoadDayoneJson:
    """Tests for loading Day One exports."""

    def test_loads_json_file(self, dayone_json_file, dayone_entry):
        entries, journal_name = load_dayone_json(dayone_json_file)
        assert len(entries) == 1
        assert entries[0]["uuid"] == dayone_entry["uuid"]
        assert journal_name == "TestJournal"

    def test_loads_zip_file(self, dayone_zip_file, dayone_entry):
        entries, journal_name = load_dayone_json(dayone_zip_file)
        assert len(entries) == 1
        assert entries[0]["uuid"] == dayone_entry["uuid"]
        assert journal_name == "MyJournal"

    def test_extracts_journal_name_from_filename(self, tmp_path):
        import json
        data = {"metadata": {}, "entries": []}
        path = tmp_path / "PersonalDiary.json"
        path.write_text(json.dumps(data))

        _, journal_name = load_dayone_json(path)
        assert journal_name == "PersonalDiary"


class TestParseEntries:
    """Tests for entry parsing."""

    def test_converts_uuid_format(self, dayone_entry):
        entries = parse_entries([dayone_entry], "TestJournal")
        content, dt, uuid = entries[0]
        assert uuid == "6F30404A-B159-433D-8C9A-F57052E4F3B6"

    def test_cleans_markdown(self, dayone_entry):
        entries = parse_entries([dayone_entry], "TestJournal")
        content, dt, uuid = entries[0]
        assert r"\." not in content
        assert "park." in content

    def test_parses_datetime(self, dayone_entry):
        entries = parse_entries([dayone_entry], "TestJournal")
        content, dt, uuid = entries[0]
        assert dt.tzinfo == timezone.utc
        assert dt.year == 2025

    def test_skips_empty_content(self, dayone_multi_entry_json):
        from importers.dayone import load_dayone_json
        raw_entries, _ = load_dayone_json(dayone_multi_entry_json)
        entries = parse_entries(raw_entries, "Test")
        # Should have 2 entries (third is whitespace-only)
        assert len(entries) == 2

    def test_preserves_entry_order(self, dayone_multi_entry_json):
        from importers.dayone import load_dayone_json
        raw_entries, _ = load_dayone_json(dayone_multi_entry_json)
        entries = parse_entries(raw_entries, "Test")
        assert entries[0][0] == "First entry"
        assert entries[1][0] == "Second entry"
