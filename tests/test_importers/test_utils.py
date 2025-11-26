"""Tests for importer utilities."""

import pytest

from importers.utils import setup_argparse, generate_file_uuid


class TestSetupArgparse:
    """Tests for argument parser setup."""

    def test_requires_input_argument(self):
        parser = setup_argparse("Test description")
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_accepts_input_path(self):
        parser = setup_argparse("Test description")
        args = parser.parse_args(["input.json"])
        assert args.input == "input.json"

    def test_dry_run_default_false(self):
        parser = setup_argparse("Test description")
        args = parser.parse_args(["input.json"])
        assert args.dry_run is False

    def test_dry_run_flag(self):
        parser = setup_argparse("Test description")
        args = parser.parse_args(["input.json", "--dry-run"])
        assert args.dry_run is True

    def test_journal_has_default(self):
        parser = setup_argparse("Test description")
        args = parser.parse_args(["input.json"])
        assert args.journal is not None

    def test_journal_override(self):
        parser = setup_argparse("Test description")
        args = parser.parse_args(["input.json", "--journal", "custom"])
        assert args.journal == "custom"


class TestGenerateFileUuid:
    """Tests for deterministic UUID generation."""

    def test_same_input_same_output(self):
        uuid1 = generate_file_uuid("source", "filepath")
        uuid2 = generate_file_uuid("source", "filepath")
        assert uuid1 == uuid2

    def test_different_source_different_uuid(self):
        uuid1 = generate_file_uuid("dayone", "file.json")
        uuid2 = generate_file_uuid("blogger", "file.json")
        assert uuid1 != uuid2

    def test_different_filepath_different_uuid(self):
        uuid1 = generate_file_uuid("source", "file1.json")
        uuid2 = generate_file_uuid("source", "file2.json")
        assert uuid1 != uuid2

    def test_returns_string(self):
        result = generate_file_uuid("source", "filepath")
        assert isinstance(result, str)

    def test_returns_valid_uuid(self):
        from uuid import UUID
        result = generate_file_uuid("source", "filepath")
        UUID(result)  # Should not raise

    def test_handles_special_characters(self):
        result = generate_file_uuid("source", "/path/to/file with spaces.json")
        assert isinstance(result, str)
        assert len(result) == 36  # Standard UUID length with dashes
