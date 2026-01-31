import glob
import re
import os

def parse_log(filepath):
    """Extract MAP from log file."""
    with open(filepath, 'r') as f:
        content = f.read()
        match = re.search(r"map:\s+([0-9\.]+)", content)
        if match:
            return float(match.group(1))
    return 0.0

def main():
    print("| Depth (k) | RM3 (DL19) | MS-MEQE (DL19) | RM3 (DL20) | MS-MEQE (DL20) |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    budgets = [10, 30, 50, 70]
    
    for k in budgets:
        # DL19
        rm3_dl19 = parse_log(f"results/rq2/rm3_{k}_dl19.log")
        msmeqe_dl19 = parse_log(f"results/rq2/msmeqe_{k}_dl19.log")
        
        # DL20
        rm3_dl20 = parse_log(f"results/rq2/rm3_{k}_dl20.log")
        msmeqe_dl20 = parse_log(f"results/rq2/msmeqe_{k}_dl20.log")
        
        print(f"| {k} | {rm3_dl19:.4f} | {msmeqe_dl19:.4f} | {rm3_dl20:.4f} | {msmeqe_dl20:.4f} |")

if __name__ == "__main__":
    main()
