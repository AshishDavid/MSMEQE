
import logging
import os
import sys
from pathlib import Path
import jnius_config

# Ensure JVM config matching the environment
if not jnius_config.vm_running:
    try:
        jnius_config.add_options('-Djava.awt.headless=true')
        jnius_config.add_options(f"-Djava.library.path={os.environ.get('LD_LIBRARY_PATH')}")
        lucene_jars = list(Path("lucene_jars").glob("*.jar"))
        classpath = [str(p.absolute()) for p in lucene_jars]
        jnius_config.set_classpath('.', *classpath)
    except Exception as e:
        print(f"Warning on JVM init: {e}")

from jnius import autoclass, cast

def inspect_fields():
    index_path = "data/msmarco_index"
    JDirectoryReader = autoclass('org.apache.lucene.index.DirectoryReader')
    JFSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
    JPath = autoclass('java.nio.file.Paths')
    
    reader = JDirectoryReader.open(JFSDirectory.open(JPath.get(index_path)))
    # Cast to IndexReader
    index_reader = cast('org.apache.lucene.index.IndexReader', reader)
    
    # Get first doc
    if index_reader.numDocs() > 0:
        # Try retrieving doc 0
        doc = index_reader.storedFields().document(0)
        print("Fields in Doc 0:")
        fields = doc.getFields()
        # Specifically check docid
        print("\nChecking 'docid' field specifically:")
        docid_field = doc.getField("docid")
        if docid_field:
            print(f"  Field Class: {docid_field.getClass().getName()}")
            print(f"  stringValue: {docid_field.stringValue()}")
            print(f"  binaryValue: {docid_field.binaryValue()}")
            print(f"  numericValue: {docid_field.numericValue()}")
            # Try to force string conversion if BytesRef
            if docid_field.binaryValue():
                 print(f"  utf8ToString: {docid_field.binaryValue().utf8ToString()}")
        else:
            print("  'docid' field return null from getField()")
    else:
        print("Empty index")

if __name__ == "__main__":
    inspect_fields()
