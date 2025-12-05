# msmeqe/expansion/wat_adapter.py

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from msmeqe.expansion.kb_expansion import KBEntity

logger = logging.getLogger(__name__)


def wat_json_to_kb_entities(
    wat_doc: Dict[str, Any],
    wikiid_to_desc: Optional[Dict[str, str]] = None,
) -> List[KBEntity]:
    """
    Convert a single WAT JSON doc (already json.loads'ed) into KBEntity objects.

    Expected WAT format (from your WAT script):

        {
          "doc_id": "...",
          "entities": [
            {
              "wikipedia_title": "...",
              "wikipedia_id": 123,
              "start": 10,
              "end": 20,
              "rho": 0.56,
              "prior_prob": 0.12
            },
            ...
          ]
        }

    Args:
        wat_doc: Parsed JSON line from WAT output.
        wikiid_to_desc: Optional dict mapping str(wikipedia_id) -> description.

    Returns:
        List[KBEntity] that kb_expansion.build_kb_candidates() can use.
    """
    entities: List[KBEntity] = []
    wikiid_to_desc = wikiid_to_desc or {}

    raw_entities = wat_doc.get("entities", [])
    if not isinstance(raw_entities, list):
        logger.warning("WAT doc has no 'entities' list: %s", wat_doc.get("doc_id"))
        return entities

    for ann in raw_entities:
        try:
            title = ann.get("wikipedia_title", "")
            if not title:
                continue

            # Wikipedia numeric ID (or string) from WAT
            raw_id = ann.get("wikipedia_id")
            if raw_id is None:
                wikiid_str = ""
            else:
                wikiid_str = str(raw_id)

            # DBpedia-style URI (nice but optional)
            uri = f"http://dbpedia.org/resource/{title.replace(' ', '_')}"

            # Confidence score – you can mix rho and prior_prob if you want
            rho = float(ann.get("rho", 0.0))
            prior = float(ann.get("prior_prob", 0.0)) if "prior_prob" in ann else 0.0
            score = rho  # or, e.g., 0.7 * rho + 0.3 * prior

            start = ann.get("start")
            end = ann.get("end")

            # Look up description from the TSV mapping
            desc = wikiid_to_desc.get(wikiid_str, None)

            entities.append(
                KBEntity(
                    uri=uri,
                    label=title,
                    score=score,
                    start=start,
                    end=end,
                    description=desc,
                )
            )
        except Exception as e:  # robust to weird records
            logger.warning("Failed to parse WAT entity: %r (error: %s)", ann, e)

    return entities
