
import ir_datasets
import sys

def check(dataset_id):
    print(f"Checking {dataset_id}...")
    try:
        ds = ir_datasets.load(dataset_id)
        if hasattr(ds, 'docs_iter'):
            # Try to fetch one doc
            next(ds.docs_iter())
            print(f"PASS: {dataset_id} docs accessible")
        else:
            print(f"FAIL: {dataset_id} has no docs_iter")
    except Exception as e:
        print(f"FAIL: {dataset_id} - {e}")

if __name__ == "__main__":
    check("disks45/nocr/trec-robust-2004")
