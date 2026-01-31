#!/bin/bash
# scripts/run_sigir_final_all.sh

set -e

export JAVA_HOME="$PWD/jdk-21"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export LD_LIBRARY_PATH="$JAVA_HOME/lib/server:$LD_LIBRARY_PATH"
export PYTHONPATH=.
VENV="./venv/bin/python"
LUCENE_JARS="./lucene_jars"

mkdir -p results/sigir_final

echo "=========================================================="
echo "Starting SIGIR 2024 Final Experimental Suite"
echo "Date: $(date)"
echo "=========================================================="

echo "[1/4] Running RQ1: Value of Combinatorial Optimization..."
$VENV scripts/run_sigir_rq1.py --lucene-jars $LUCENE_JARS > results/sigir_final/rq1.log 2>&1

echo "[2/4] Running RQ2: Robustness and Safe Aggression..."
$VENV scripts/run_sigir_rq2.py --lucene-jars $LUCENE_JARS > results/sigir_final/rq2.log 2>&1

echo "[3/4] Running RQ3: Source-Agility and Cross-Domain Generalization..."
$VENV scripts/run_sigir_rq3.py --lucene-jars $LUCENE_JARS > results/sigir_final/rq3.log 2>&1

echo "[4/4] Running RQ4: Interpretability and Semantic Gap..."
$VENV scripts/run_sigir_rq4.py --lucene-jars $LUCENE_JARS > results/sigir_final/rq4.log 2>&1

echo "=========================================================="
echo "Final Evaluation Suite Completed!"
echo "Results available in results/sigir_final/"
echo "Date: $(date)"
echo "=========================================================="
