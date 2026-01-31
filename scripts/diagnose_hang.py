import time
import numpy as np
import logging
from src.utils.lucene_utils import initialize_lucene

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lucene():
    logger.info("Testing Lucene Init...")
    start = time.time()
    initialize_lucene(lucene_path="./lucene_jars")
    logger.info(f"Lucene Init took {time.time() - start:.2f}s")

def test_embeddings_load():
    logger.info("Testing Embeddings Load...")
    start = time.time()
    path = "data/msmarco_index/doc_embeddings.npy"
    # Try mmap_mode='r' to see if it's faster/safer
    data = np.load(path, mmap_mode='r')
    logger.info(f"Embeddings Load (mmap) took {time.time() - start:.2f}s. Shape: {data.shape}")

if __name__ == "__main__":
    test_lucene()
    test_embeddings_load()
