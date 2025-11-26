"""Tests for basic XML importer."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from importers.basic_xml import parse_date, parse_xml
from importers.utils import generate_file_uuid


class TestParseDate:
    """Tests for date parsing."""

    def test_parses_date_format(self):
        tz = ZoneInfo("UTC")
        result = parse_date("26,November,2025", tz)
        assert result.day == 26
        assert result.month == 11
        assert result.year == 2025

    def test_converts_to_utc(self):
        tz = ZoneInfo("America/New_York")
        result = parse_date("26,November,2025", tz)
        assert result.tzinfo == timezone.utc

    def test_timezone_offset_applied(self):
        ny_tz = ZoneInfo("America/New_York")
        result = parse_date("26,November,2025", ny_tz)
        # Midnight in NY (EST, UTC-5) = 05:00 UTC
        assert result.hour == 5
        assert result.tzinfo == timezone.utc

    def test_utc_timezone_no_offset(self):
        utc_tz = ZoneInfo("UTC")
        result = parse_date("26,November,2025", utc_tz)
        assert result.hour == 0
        assert result.tzinfo == timezone.utc

    def test_handles_whitespace(self):
        tz = ZoneInfo("UTC")
        result = parse_date("  26,November,2025  ", tz)
        assert result.day == 26

    def test_different_months(self):
        tz = ZoneInfo("UTC")
        months = [
            ("15,January,2025", 1),
            ("15,February,2025", 2),
            ("15,March,2025", 3),
            ("15,December,2025", 12),
        ]
        for date_str, expected_month in months:
            result = parse_date(date_str, tz)
            assert result.month == expected_month


class TestParseXml:
    """Tests for XML parsing."""

    def test_parses_entries(self, basic_xml_file):
        tz = ZoneInfo("UTC")
        entries = parse_xml(basic_xml_file, tz)
        assert len(entries) == 2

    def test_extracts_content(self, basic_xml_file):
        tz = ZoneInfo("UTC")
        entries = parse_xml(basic_xml_file, tz)
        content, _, _ = entries[0]
        assert content == "First blog post content."

    def test_extracts_dates(self, basic_xml_file):
        tz = ZoneInfo("UTC")
        entries = parse_xml(basic_xml_file, tz)
        _, dt, _ = entries[0]
        assert dt.day == 26
        assert dt.month == 11

    def test_generates_deterministic_uuids(self, basic_xml_file):
        tz = ZoneInfo("UTC")
        entries1 = parse_xml(basic_xml_file, tz)
        entries2 = parse_xml(basic_xml_file, tz)
        assert entries1[0][2] == entries2[0][2]
        assert entries1[1][2] == entries2[1][2]

    def test_unique_uuids_per_entry(self, basic_xml_file):
        tz = ZoneInfo("UTC")
        entries = parse_xml(basic_xml_file, tz)
        uuids = [e[2] for e in entries]
        assert len(uuids) == len(set(uuids))

    def test_skips_empty_posts(self, basic_xml_with_empty):
        tz = ZoneInfo("UTC")
        entries = parse_xml(basic_xml_with_empty, tz)
        assert len(entries) == 1
        assert entries[0][0] == "Valid post"

    def test_dates_are_utc(self, basic_xml_file):
        tz = ZoneInfo("America/New_York")
        entries = parse_xml(basic_xml_file, tz)
        for _, dt, _ in entries:
            assert dt.tzinfo == timezone.utc


class TestGenerateFileUuid:
    """Tests for UUID generation utility."""

    def test_deterministic(self):
        uuid1 = generate_file_uuid("source", "path")
        uuid2 = generate_file_uuid("source", "path")
        assert uuid1 == uuid2

    def test_different_sources_different_uuids(self):
        uuid1 = generate_file_uuid("blogger", "file.xml")
        uuid2 = generate_file_uuid("dayone", "file.xml")
        assert uuid1 != uuid2

    def test_different_paths_different_uuids(self):
        uuid1 = generate_file_uuid("blogger", "file1.xml")
        uuid2 = generate_file_uuid("blogger", "file2.xml")
        assert uuid1 != uuid2

    def test_returns_valid_uuid_format(self):
        import uuid
        result = generate_file_uuid("test", "path")
        # Should not raise
        uuid.UUID(result)
