#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
VENV="./venv/bin/python3"

# Final specific points to test
ALPHAS=(0.0 0.1)
BETAS=(0.0 1.0 10.0 50.0 100.0)

mkdir -p breakthrough_results

for a in "${ALPHAS[@]}"; do
    for b in "${BETAS[@]}"; do
        echo "Processing a=$a, b=$b..."
        OUT="runs/bt_final_a${a}_b${b}.txt"
        EVAL="breakthrough_results/final_eval_a${a}_b${b}.json"
        
        $VENV scripts/run_oracle_enhanced.py \
            --dense-run runs/dense_baseline.txt \
            --msmeqe-run runs/msmeqe_oracle_advanced.txt \
            --oracle-stats results/oracle_stats_alignment.jsonl \
            --gain-model models/alignment_breakthrough/gain_model.pkl \
            --risk-model models/alignment_breakthrough/risk_model.pkl \
            --align-model models/alignment_breakthrough/align_model.pkl \
            --alpha $a --beta $b \
            --output $OUT > /dev/null
            
        $VENV best_result_reproduction/scripts/evaluate_comprehensive.py data/test_qrels.txt $OUT > $EVAL
        
        MAP=$(grep '"MAP"' $EVAL | awk '{print $2}' | tr -d ',')
        echo "Result: a=$a, b=$b -> MAP: $MAP"
    done
done
