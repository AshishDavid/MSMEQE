
import logging
from src.utils.lucene_utils import initialize_lucene, get_lucene_classes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    index_path = "data/robust04_index"
    lucene_path = "lucene_jars"
    
    # Init Lucene
    if not initialize_lucene(lucene_path):
        return
        
    classes = get_lucene_classes()
    DirectoryReader = classes['IndexReader']
    FSDirectory = classes['FSDirectory']
    JavaPaths = classes['JavaPaths']
    Term = classes['Term']
    
    directory = FSDirectory.open(JavaPaths.get(index_path))
    reader = DirectoryReader.open(directory)
    
    target_id = "FBIS3-10082"
    term = Term("id", target_id)
    count = reader.docFreq(term)
    
    print(f"Document ID: {target_id}")
    print(f"Count in Index: {count}")
    
    if count > 0:
        print("VERIFICATION SUCCESS: ID found.")
    else:
        print("VERIFICATION FAILURE: ID not found.")

if __name__ == "__main__":
    main()
