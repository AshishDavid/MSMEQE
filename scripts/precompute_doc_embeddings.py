# scripts/precompute_doc_embeddings.py

"""
Precompute document embeddings for efficient dense retrieval.

This script:
  1. Loads all documents from the collection
  2. Encodes them using Sentence-BERT
  3. Saves embeddings and doc IDs to disk

Run this ONCE before training the budget model.

Usage:
    python scripts/precompute_doc_embeddings.py \\
        --collection msmarco-passage \\
        --index-path data/msmarco_index \\
        --output-dir data/msmarco_index \\
        --batch-size 512
"""

import logging
import argparse
import sys
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm
import ir_datasets


from src.reranking.semantic_encoder import SemanticEncoder

logger = logging.getLogger(__name__)


def precompute_document_embeddings(
        collection_name: str,
        output_dir: str,
        encoder: SemanticEncoder,
        input_file: str = None,
        batch_size: int = 512,
        max_docs: int = None,
):
    """
    Precompute and save document embeddings using memmap.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterator and Count
    if input_file:
        logger.info(f"Loading from file: {input_file}")
        # Count lines for memmap
        logger.info("Counting documents in file...")
        total_docs = 0
        with open(input_file, 'r') as f:
            for _ in f:
                total_docs += 1
        
        if max_docs:
            total_docs = min(total_docs, max_docs)
            
        def _doc_iter():
            with open(input_file, 'r') as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        
        iterator = _doc_iter()
        
    elif collection_name:
        logger.info(f"Loading collection: {collection_name}")
        dataset = ir_datasets.load(collection_name)
        try:
            total_docs = dataset.docs_count()
            if max_docs:
                total_docs = min(total_docs, max_docs)
        except:
            total_docs = None
            logger.warning("Could not determine total documents count.")
        
        iterator = dataset.docs_iter()
    else:
        raise ValueError("Must provide either --collection or --input-file")

    if total_docs is None:
         # Rough estimate or read into list (not recommended for large data)
         raise ValueError("Cannot determine total documents count for memmap.")

    logger.info(f"Total documents to process: {total_docs}")

    # Initialize memmap
    emb_dim = encoder.get_dim()
    embeddings_path = output_dir / "doc_embeddings.npy"
    
    logger.info(f"Initializing .npy memmap at {embeddings_path} with shape ({total_docs}, {emb_dim})")
    
    doc_embeddings = np.lib.format.open_memmap(
        str(embeddings_path), 
        mode='w+', 
        dtype='float32', 
        shape=(total_docs, emb_dim)
    )

    doc_ids = []
    batch_texts = []
    batch_ids = []
    
    current_idx = 0
    
    # Process docs
    for i, doc in enumerate(tqdm(iterator, total=total_docs, desc="Processing")):
        if max_docs and i >= max_docs:
            break

        doc_text = None
        doc_id = None

        # Handle IR Dataset object
        if hasattr(doc, 'doc_id'):
            doc_id = doc.doc_id
            if hasattr(doc, 'text'):
                doc_text = doc.text
            elif hasattr(doc, 'body'):
                doc_text = doc.body
        
        # Handle Dict (JSONL)
        elif isinstance(doc, dict):
            # Try common fields
            doc_id = doc.get('id') or doc.get('docid') or doc.get('_id')
            doc_text = doc.get('contents') or doc.get('text') or doc.get('body')

        if not doc_text or not doc_id:
            continue

        batch_texts.append(doc_text)
        batch_ids.append(doc_id)

        # Encode batch when full
        if len(batch_texts) >= batch_size:
            embeddings = encoder.encode(batch_texts)
            
            # Write to memmap
            end_idx = current_idx + len(embeddings)
            doc_embeddings[current_idx:end_idx] = embeddings
            doc_embeddings.flush()
            
            current_idx = end_idx
            doc_ids.extend(batch_ids)

            batch_texts = []
            batch_ids = []

    # Encode remaining batch
    if batch_texts:
        embeddings = encoder.encode(batch_texts)
        end_idx = current_idx + len(embeddings)
        doc_embeddings[current_idx:end_idx] = embeddings
        doc_embeddings.flush()
        current_idx = end_idx
        doc_ids.extend(batch_ids)

    logger.info(f"Encoded {len(doc_ids)} documents.")

    # Trim if necessary (create new view if less docs than expected)
    if len(doc_ids) < total_docs:
        logger.warning(f"Processed fewer docs ({len(doc_ids)}) than expected ({total_docs}).")
        # Since open_memmap fixed the size at creation, we can't easily resize in place without copying.
        # But we can update the doc_ids to match. The valid data is just the prefix.
        # Ideally we would resize the file, but for now we just warn.
        pass

    # No need to call np.save() because open_memmap writes directly to the .npy file
    # and we called flush() periodically.
    del doc_embeddings  # Close memmap cleanly

    # Save doc IDs
    ids_path = output_dir / "doc_ids.json"
    logger.info(f"Saving doc IDs to {ids_path}")
    with open(ids_path, 'w') as f:
        json.dump(doc_ids, f)

    logger.info("Precomputation complete!")


def main():
    parser = argparse.ArgumentParser(description="Precompute document embeddings")
    parser.add_argument("--collection", type=str, required=False,
                        help="IR dataset collection name (optional if --input-file used)")
    parser.add_argument("--input-file", type=str, required=False,
                        help="Path to JSONL input file (alternative to --collection)")
    # Note: index-path arg is kept for compatibility but output-dir dictates save location
    parser.add_argument("--index-path", type=str, required=False,
                        help="Path to Lucene index (optional/unused in this script)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--model-name", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Sentence-BERT model name")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Encoding batch size")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Maximum documents to encode (for testing)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

    # Initialize encoder
    logger.info(f"Loading encoder: {args.model_name}")
    encoder = SemanticEncoder(model_name=args.model_name)

    # Precompute embeddings
    precompute_document_embeddings(
        collection_name=args.collection,
        input_file=args.input_file,
        output_dir=args.output_dir,
        encoder=encoder,
        batch_size=args.batch_size,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()