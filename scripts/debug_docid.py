
import sys
from src.utils.lucene_utils import initialize_lucene

initialize_lucene(lucene_path="./lucene_jars")

from jnius import autoclass

index_path = "data/msmarco_index"
FSDirectory = autoclass("org.apache.lucene.store.FSDirectory")
DirectoryReader = autoclass("org.apache.lucene.index.DirectoryReader")
Paths = autoclass("java.nio.file.Paths")
IndexSearcher = autoclass("org.apache.lucene.search.IndexSearcher")

directory = FSDirectory.open(Paths.get(index_path))
reader = DirectoryReader.open(directory)
searcher = IndexSearcher(reader)

# Get a valid doc ID (not 0, maybe one from results?)
# Just try checking doc 0, 1, 100
stored_fields = searcher.storedFields()

for i in [0, 1, 100, 1000]:
    try:
        doc = stored_fields.document(i)
        
        # Check 'id' field
        f_id = doc.getField("id")
        print(f"Doc {i} 'id' field: {f_id}")
        if f_id:
            print(f"  stringValue: {f_id.stringValue()}")
            
        # Check 'docid' field
        f_docid = doc.getField("docid")
        print(f"Doc {i} 'docid' field: {f_docid}")
        if f_docid:
            bin_val = f_docid.binaryValue()
            print(f"  binaryValue: {bin_val}")
            if bin_val:
                print(f"  utf8ToString: {bin_val.utf8ToString()}")
                
    except Exception as e:
        print(f"Error on doc {i}: {e}")

reader.close()
