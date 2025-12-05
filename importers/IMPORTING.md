# Importing Journal Entries

Import journal entries from external sources into charlie.

## Supported Formats

| Format | Command |
|--------|---------|
| Text Files | `python importers/files.py ~/journals/` |
| Day One | `python importers/dayone.py export.zip` |
| Basic XML | `python importers/basic_xml.py corpus.xml --timezone UTC` |

## Common Options

All importers support:

- `-h, --help` - Show help message and usage
- `--dry-run` - Show what would be imported without persisting
- `--journal NAME` - Target journal (default: charlie's default journal)

## Files Importer

Imports from a directory of text files (`.txt`, `.md`, `.rtf`, etc.).

```bash
python importers/files.py ~/old-journals/
python importers/files.py ~/notes/ --extensions md --date-source modified
python importers/files.py ~/Documents/journals/ --recursive --timezone America/New_York

# Parse dates from filenames like 2024-10-13.txt or 2024-10-13-2354.txt
python importers/files.py ~/journals/ --filename-date "%Y-%m-%d"
python importers/files.py ~/journals/ --filename-date "%Y-%m-%d-%H%M"
```

Options:
- `--filename-date FORMAT` - Parse date from filename using strptime format (falls back to file metadata if parsing fails)
- `--date-source created|modified` - Use file creation or modification time (default: `created`)
- `--extensions txt,md,rtf` - Comma-separated extensions (default: `txt,md,rtf,markdown,text`)
- `--recursive` - Recurse into subdirectories
- `--timezone ZONE` - Timezone for file timestamps (default: system timezone)

Features:
- Supports `.txt`, `.md`, `.rtf`, `.markdown`, `.text` files by default
- RTF files are automatically stripped of formatting
- Skips hidden files/directories and symlinks
- Can parse dates from filenames (e.g., `2024-10-13.txt`) or use file metadata
- Deterministic UUIDs based on file path (idempotent imports)

**Note:** Moving or renaming files will cause duplicate imports since UUIDs include the file path.

## Day One Importer

Imports from Day One JSON exports (`.json` or `.zip`).

```bash
python importers/dayone.py ~/Downloads/DayOneExport.zip
python importers/dayone.py ~/Downloads/Journal.json --dry-run
```

Features:
- Handles both `.zip` and `.json` exports
- Preserves original Day One UUIDs
- Cleans markdown escape characters
- Skips empty entries

## Basic XML Importer

Imports from XML files with `<date>/<post>` structure (e.g., blog authorship corpus).

```bash
python importers/basic_xml.py corpus.xml
python importers/basic_xml.py corpus.xml --timezone America/New_York --dry-run
```

If `--timezone` is not specified, your system timezone is used (and logged).

Expected XML format:
```xml
<Root>
    <date>26,November,2025</date>
    <post>Entry content here...</post>
</Root>
```

## Idempotency

All imports are idempotent. Re-running an import will skip entries that already exist (matched by UUID). This allows safe re-imports without duplicates.

## Custom Importers

Place custom importers in `importers/custom/` (gitignored). Use `importers/utils.py` for shared utilities:

```python
from importers.utils import setup_argparse, import_entries, generate_file_uuid
```
