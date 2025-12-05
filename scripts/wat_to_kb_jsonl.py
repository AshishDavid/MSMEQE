# msmeqe/scripts/wat_to_kb_jsonl.py

from __future__ import annotations

import argparse
import gzip
import json
import logging
from typing import Dict, Iterable, TextIO

from msmeqe.utils.wiki_desc_utils import load_wikiid_to_desc_tsv
from msmeqe.expansion.wat_adapter import wat_json_to_kb_entities

logger = logging.getLogger(__name__)


def _open_maybe_gzip(path: str, mode: str = "rt") -> TextIO:
    """
    Open a file that may be plain text or .gz (by extension).
    """
    if path.endswith(".gz"):
        return gzip.open(path, mode=mode, encoding="utf-8")  # type: ignore[arg-type]
    return open(path, mode=mode, encoding="utf-8")


def iter_jsonl(path: str) -> Iterable[dict]:
    """
    Yield JSON objects line by line from a (possibly gzipped) JSONL file.
    """
    with _open_maybe_gzip(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON line: %s", line[:200])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert WAT JSONL output into a KB-enriched JSONL with descriptions "
            "from a wikiid->desc TSV file."
        )
    )
    parser.add_argument(
        "--wat-jsonl",
        type=str,
        required=True,
        help="Input WAT JSONL file (optionally .gz).",
    )
    parser.add_argument(
        "--wikiid-desc-tsv",
        type=str,
        required=True,
        help="TSV with wikiid<TAB>description.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        required=True,
        help="Output JSONL path for KB-enriched entities.",
    )
    parser.add_argument(
        "--tsv-has-header",
        action="store_true",
        help="Set if the TSV has a header row to skip.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading wikiid->desc TSV from %s", args.wikiid_desc_tsv)
    wikiid_to_desc: Dict[str, str] = load_wikiid_to_desc_tsv(
        args.wikiid_desc_tsv, has_header=args.tsv_has_header
    )
    logger.info("Loaded %d wikiid descriptions", len(wikiid_to_desc))

    logger.info("Reading WAT JSONL from %s", args.wat_jsonl)
    with _open_maybe_gzip(args.output_jsonl, "wt") as out_f:
        n_docs = 0
        for wat_doc in iter_jsonl(args.wat_jsonl):
            doc_id = wat_doc.get("doc_id")
            if doc_id is None:
                # Just keep moving, some lines might be weird
                continue

            kb_entities = wat_json_to_kb_entities(
                wat_doc=wat_doc,
                wikiid_to_desc=wikiid_to_desc,
            )

            # Convert KBEntity objects back to simple dicts for JSONL
            entities_out = []
            for e in kb_entities:
                entities_out.append(
                    {
                        "uri": e.uri,
                        "label": e.label,
                        "score": e.score,
                        "start": e.start,
                        "end": e.end,
                        "description": e.description,
                    }
                )

            out_obj = {
                "doc_id": doc_id,
                "entities": entities_out,
            }
            out_f.write(json.dumps(out_obj) + "\n")
            n_docs += 1

        logger.info("Wrote KB-enriched JSONL for %d docs to %s", n_docs, args.output_jsonl)


if __name__ == "__main__":
    main()
