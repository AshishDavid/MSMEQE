# msmeqe/utils/wiki_desc_utils.py

from __future__ import annotations

from typing import Dict


def load_wikiid_to_desc_tsv(path: str, has_header: bool = False) -> Dict[str, str]:
    """
    Load a TSV file mapping wikiid -> description.

    Expected format (per line):
        wikiid<TAB>description

    Args:
        path: Path to the TSV file.
        has_header: If True, skips the first line.

    Returns:
        dict mapping str(wikiid) -> description string
    """
    mapping: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        first = True
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            if has_header and first:
                first = False
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                # Skip malformed lines
                continue

            wikiid, desc = parts
            wikiid = wikiid.strip()
            desc = desc.strip()

            if wikiid and desc:
                mapping[wikiid] = desc

    return mapping
