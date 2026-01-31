
#!/usr/bin/env python3
import time
import os
import sys

def count_lines(fname):
    if not os.path.exists(fname):
        return 0
    try:
        with open(fname, 'rb') as f:
            return sum(1 for _ in f)
    except:
        return 0

def main():
    train_file = "data/llm_candidates_train.jsonl"
    test_file = "data/llm_candidates_test.jsonl"
    train_target = 5000
    test_target = 6980 # Fixed count for test queries

    print(f"Monitoring Progress...")
    print(f"Target Train: {train_target}")
    print(f"Target Test:  {test_target}")

    start_time = time.time()
    
    while True:
        c_train = count_lines(train_file)
        c_test = count_lines(test_file)
        
        elapsed = time.time() - start_time
        
        # Calculate rate (queries per second)
        total_done = c_train + c_test
        rate = total_done / elapsed if elapsed > 0 else 0
        
        sys.stdout.write(f"\r[Time: {elapsed:.0f}s] Train: {c_train}/{train_target} | Test: {c_test}/{test_target} | Rate: {rate:.2f} q/s")
        sys.stdout.flush()
        
        if c_train >= train_target and c_test >= test_target:
            print("\nDone!")
            break
        
        time.sleep(5)

if __name__ == "__main__":
    main()
