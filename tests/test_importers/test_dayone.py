"""Tests for Day One importer."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

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
    """Tests for markdown escape cleaning.

    Day One escapes special characters with backslashes. These must be
    removed to produce clean markdown that renders correctly and doesn't
    confuse the entity extractor.
    """

    # Individual character tests
    def test_removes_escaped_periods(self):
        assert clean_dayone_markdown(r"Hello\. World\.") == "Hello. World."

    def test_removes_escaped_exclamation(self):
        assert clean_dayone_markdown(r"Good luck Charlie\!") == "Good luck Charlie!"

    def test_removes_escaped_hash(self):
        assert clean_dayone_markdown(r"\# Not a header") == "# Not a header"

    def test_removes_escaped_asterisks(self):
        assert clean_dayone_markdown(r"\*not bold\*") == "*not bold*"

    def test_removes_escaped_underscores(self):
        assert clean_dayone_markdown(r"\_not italic\_") == "_not italic_"

    def test_removes_escaped_brackets(self):
        assert clean_dayone_markdown(r"\[not a link\]") == "[not a link]"

    def test_removes_escaped_parens(self):
        assert clean_dayone_markdown(r"\(not a link url\)") == "(not a link url)"

    def test_removes_escaped_curly_braces(self):
        assert clean_dayone_markdown(r"\{code\}") == "{code}"

    def test_removes_escaped_plus(self):
        assert clean_dayone_markdown(r"1\+ 2") == "1+ 2"

    def test_removes_escaped_minus(self):
        assert clean_dayone_markdown(r"\- item") == "- item"

    def test_removes_escaped_pipe(self):
        assert clean_dayone_markdown(r"a \| b") == "a | b"

    def test_removes_escaped_backtick(self):
        assert clean_dayone_markdown(r"\`code\`") == "`code`"

    def test_removes_escaped_tilde(self):
        assert clean_dayone_markdown(r"\~\~strikethrough\~\~") == "~~strikethrough~~"

    def test_removes_escaped_angle_brackets(self):
        assert clean_dayone_markdown(r"\<html\>") == "<html>"

    def test_removes_escaped_commas(self):
        assert clean_dayone_markdown(r"Topics: AI\, coding\, life") == "Topics: AI, coding, life"

    # Preservation tests
    def test_preserves_normal_backslashes(self):
        assert clean_dayone_markdown(r"path\\to\\file") == r"path\\to\\file"

    def test_preserves_backslash_before_letters(self):
        assert clean_dayone_markdown(r"line\nbreak") == r"line\nbreak"

    def test_preserves_normal_markdown_images(self):
        assert clean_dayone_markdown("![alt](https://example.com/image.png)") == "![alt](https://example.com/image.png)"

    # Image reference tests
    def test_strips_dayone_image_references(self):
        result = clean_dayone_markdown("Before ![](dayone-moment://DE9D21695B6B480BBC86D42E861AC857) After")
        assert result == "Before  After"

    def test_strips_multiple_image_references(self):
        result = clean_dayone_markdown("![](dayone-moment://AAA) text ![](dayone-moment://BBB)")
        assert result == " text "

    # Comprehensive before/after tests
    def test_realistic_entry_with_escapes(self):
        before = r"Good luck Charlie\! This is a test\. Here's a list: 1\. First 2\. Second"
        after = "Good luck Charlie! This is a test. Here's a list: 1. First 2. Second"
        assert clean_dayone_markdown(before) == after

    def test_entry_with_image_and_escapes(self):
        before = r"Check this out\! ![](dayone-moment://ABC123) Pretty cool\."
        after = "Check this out!  Pretty cool."
        assert clean_dayone_markdown(before) == after

    def test_no_trailing_backslashes(self):
        """Ensure no stray backslashes remain after cleaning."""
        cases = [
            r"Hello\. World\!",
            r"\# Header\.",
            r"List\: 1\. 2\. 3\.",
            r"Name\: Charlie\!",
        ]
        for text in cases:
            result = clean_dayone_markdown(text)
            # No backslash should precede punctuation in output
            assert r"\." not in result, f"Found \\. in: {result}"
            assert r"\!" not in result, f"Found \\! in: {result}"
            assert r"\:" not in result or ":" not in text.replace(r"\:", ""), f"Found \\: in: {result}"

    def test_output_is_valid_for_entity_extraction(self):
        """Output should be clean text suitable for LLM entity extraction."""
        before = r"Met with Charlie\! Great conversation\. Topics: AI\, coding\."
        result = clean_dayone_markdown(before)
        # Should contain the actual names/words without escape artifacts
        assert "Charlie!" in result
        assert "conversation." in result
        # Should not have backslash-escaped sequences
        assert "\\" not in result or result.count("\\") == before.count("\\\\")


class TestParseDayoneDate:
    """Tests for ISO date parsing."""

    def test_parses_utc_date(self):
        result = parse_dayone_date("2025-11-26T18:50:20Z")
        assert result == datetime(2025, 11, 26, 18, 50, 20, tzinfo=timezone.utc)

    def test_result_is_utc_when_no_timezone(self):
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

    def test_converts_to_local_timezone_when_provided(self):
        # 2025-09-03T01:36:57Z (UTC) = 2025-09-02T21:36:57 (America/New_York, -4 DST)
        result = parse_dayone_date("2025-09-03T01:36:57Z", "America/New_York")
        assert result.tzinfo == ZoneInfo("America/New_York")
        assert result.day == 2  # Sep 2 in local time, not Sep 3
        assert result.hour == 21  # 9:36 PM local

    def test_timezone_conversion_preserves_instant(self):
        # Same instant, different representations
        utc_result = parse_dayone_date("2025-09-03T01:36:57Z")
        local_result = parse_dayone_date("2025-09-03T01:36:57Z", "America/New_York")
        # Both should represent the same point in time
        assert utc_result == local_result

    def test_handles_various_timezones(self):
        # Test with different timezone
        result = parse_dayone_date("2025-01-01T08:00:00Z", "America/Los_Angeles")
        assert result.tzinfo == ZoneInfo("America/Los_Angeles")
        assert result.day == 1  # Still Jan 1 (midnight UTC = midnight-8 = Dec 31 4pm? No, 8am UTC = midnight PST)
        assert result.hour == 0  # Midnight PST

    def test_invalid_timezone_falls_back_to_utc(self):
        result = parse_dayone_date("2025-01-01T10:00:00Z", "Invalid/Zone")
        assert result.tzinfo == timezone.utc

    def test_empty_string_timezone_falls_back_to_utc(self):
        result = parse_dayone_date("2025-01-01T10:00:00Z", "")
        assert result.tzinfo == timezone.utc

    def test_timezone_abbreviation_falls_back_to_utc(self):
        # ZoneInfo doesn't accept abbreviations like "PST"
        result = parse_dayone_date("2025-01-01T10:00:00Z", "PST")
        assert result.tzinfo == timezone.utc


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

    def test_parses_datetime_with_timezone(self, dayone_entry):
        # Fixture has timeZone: "America/New_York"
        entries = parse_entries([dayone_entry], "TestJournal")
        content, dt, uuid = entries[0]
        assert dt.tzinfo == ZoneInfo("America/New_York")
        assert dt.year == 2025

    def test_falls_back_to_utc_without_timezone(self):
        entry_no_tz = {
            "uuid": "AAAA0000BBBB1111CCCC2222DDDD3333",
            "text": "No timezone entry",
            "creationDate": "2025-01-01T10:00:00Z",
            # No timeZone field
        }
        entries = parse_entries([entry_no_tz], "TestJournal")
        content, dt, uuid = entries[0]
        assert dt.tzinfo == timezone.utc

    def test_handles_invalid_timezone_gracefully(self):
        entry_bad_tz = {
            "uuid": "AAAA0000BBBB1111CCCC2222DDDD3333",
            "text": "Entry with invalid timezone",
            "creationDate": "2025-01-01T10:00:00Z",
            "timeZone": "InvalidTimezone",
        }
        # Should not crash - should import with UTC fallback
        entries = parse_entries([entry_bad_tz], "TestJournal")
        assert len(entries) == 1
        content, dt, uuid = entries[0]
        assert dt.tzinfo == timezone.utc

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
