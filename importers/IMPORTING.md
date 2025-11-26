# Importing Journal Entries

Import journal entries from external sources into charlie.

## Supported Formats

| Format | Command |
|--------|---------|
| Day One | `python importers/dayone.py export.json` |
| Day One (zip) | `python importers/dayone.py export.zip` |
| Blogger XML | `python importers/xml.py corpus.xml --timezone UTC` |

## Common Options

All importers support:

- `--dry-run` - Show what would be imported without persisting
- `--journal NAME` - Target journal (default: charlie's default journal)

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

## Blogger XML Importer

Imports from Blogger corpus XML format.

```bash
python importers/xml.py corpus.xml --timezone America/New_York
python importers/xml.py corpus.xml --timezone UTC --dry-run
```

The `--timezone` flag is required since the XML format has no timezone info.

Expected XML format:
```xml
<Blog>
    <date>26,November,2025</date>
    <post>Entry content here...</post>
</Blog>
```

## Idempotency

All imports are idempotent. Re-running an import will skip entries that already exist (matched by UUID). This allows safe re-imports without duplicates.

## Custom Importers

Place custom importers in `importers/custom/` (gitignored). Use `importers/utils.py` for shared utilities:

```python
from importers.utils import setup_argparse, import_entries, generate_file_uuid
```
