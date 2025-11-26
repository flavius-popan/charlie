"""Test fixtures for importer tests."""

import json
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def dayone_entry():
    """Single Day One entry data."""
    return {
        "uuid": "6F30404AB159433D8C9AF57052E4F3B6",
        "text": "# My Journal Entry\n\nToday I went to the park\\. It was nice\\!",
        "creationDate": "2025-11-26T18:50:20Z",
        "timeZone": "America/New_York",
    }


@pytest.fixture
def dayone_json_file(dayone_entry, tmp_path):
    """Create a temporary Day One JSON export file."""
    data = {
        "metadata": {"version": "1.0"},
        "entries": [dayone_entry],
    }
    json_path = tmp_path / "TestJournal.json"
    json_path.write_text(json.dumps(data))
    return json_path


@pytest.fixture
def dayone_zip_file(dayone_entry, tmp_path):
    """Create a temporary Day One ZIP export file."""
    data = {
        "metadata": {"version": "1.0"},
        "entries": [dayone_entry],
    }
    zip_path = tmp_path / "export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("MyJournal.json", json.dumps(data))
    return zip_path


@pytest.fixture
def dayone_multi_entry_json(tmp_path):
    """Day One export with multiple entries."""
    data = {
        "metadata": {"version": "1.0"},
        "entries": [
            {
                "uuid": "AAAA0000BBBB1111CCCC2222DDDD3333",
                "text": "First entry",
                "creationDate": "2025-01-01T10:00:00Z",
            },
            {
                "uuid": "EEEE4444FFFF5555000066667777AAAA",
                "text": "Second entry",
                "creationDate": "2025-01-02T10:00:00Z",
            },
            {
                "uuid": "BBBB8888CCCC9999DDDDAAAAEEEEFFFF",
                "text": "   ",  # Empty/whitespace - should be skipped
                "creationDate": "2025-01-03T10:00:00Z",
            },
        ],
    }
    json_path = tmp_path / "Multi.json"
    json_path.write_text(json.dumps(data))
    return json_path


@pytest.fixture
def blogger_xml_file(tmp_path):
    """Create a temporary Blogger corpus XML file."""
    xml_content = """<Blog>
    <date>26,November,2025</date>
    <post>First blog post content.</post>
    <date>25,November,2025</date>
    <post>Second blog post content.</post>
</Blog>"""
    xml_path = tmp_path / "corpus.xml"
    xml_path.write_text(xml_content)
    return xml_path


@pytest.fixture
def blogger_xml_with_empty(tmp_path):
    """Blogger XML with empty posts that should be skipped."""
    xml_content = """<Blog>
    <date>26,November,2025</date>
    <post>Valid post</post>
    <date>25,November,2025</date>
    <post>   </post>
    <date>24,November,2025</date>
    <post></post>
</Blog>"""
    xml_path = tmp_path / "corpus_empty.xml"
    xml_path.write_text(xml_content)
    return xml_path
