#!/bin/bash
# ============================================================
# V5 PORTFOLIO BATCH — Re-run all scenarios with versioned
# CSV + .npz output for portfolio aggregation analysis (Section 4.3.1).
# ============================================================
# Run AFTER:
#   1. Scalability v4 finishes (check: docker ps | wc -l)
#   2. Claude Code applies --pred-suffix change to run_fl.py
#   3. Verify: python pipeline/run_fl.py --help | grep pred-suffix
#
# Output:
#   CSV:  logs/fl_*_v5_portfolio.csv (new files, v4 preserved)
#   .npz: logs/predictions/v5_portfolio/*.npz (isolated subdir)
#
# Estimated time: ~5.5 hours total
# ============================================================

set -e

SUFFIX="v5_portfolio"
PRED_SUFFIX="v5_portfolio"
LOG="logs/v5_portfolio_batch_log.txt"

echo "============================================================" | tee $LOG
echo "V5 PORTFOLIO BATCH START: $(date)" | tee -a $LOG
echo "Suffix: $SUFFIX | Pred dir: logs/predictions/$PRED_SUFFIX/" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# --- 1/6: Local MLP (~25 min) ---
echo "" | tee -a $LOG
echo "--- 1/6: Local MLP — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario local_mlp --n 250 --horizon 1 \
    --suffix $SUFFIX --pred-suffix $PRED_SUFFIX 2>&1 | tee -a $LOG

# --- 2/6: FedAvg (~65 min) ---
echo "" | tee -a $LOG
echo "--- 2/6: FedAvg — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario fedavg --n 250 --rounds 20 \
    --suffix $SUFFIX --pred-suffix $PRED_SUFFIX --horizon 1 2>&1 | tee -a $LOG

# --- 3/6: FedAdam (~65 min) ---
echo "" | tee -a $LOG
echo "--- 3/6: FedAdam — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario fedadam --n 250 --rounds 20 \
    --suffix $SUFFIX --pred-suffix $PRED_SUFFIX --horizon 1 2>&1 | tee -a $LOG

# --- 4/6: FedProx (~50 min) ---
echo "" | tee -a $LOG
echo "--- 4/6: FedProx — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario fedprox --n 250 --rounds 20 --mu 0.01 \
    --suffix $SUFFIX --pred-suffix $PRED_SUFFIX --horizon 1 2>&1 | tee -a $LOG

# --- 5/6: Personalised FedAdam (~65 min) ---
echo "" | tee -a $LOG
echo "--- 5/6: Personalised FedAdam — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario personalised --n 250 --rounds 20 \
    --base-strategy fedadam --fine-tune-lr 0.001 \
    --suffix fedadam_v5_portfolio --pred-suffix $PRED_SUFFIX --horizon 1 2>&1 | tee -a $LOG

# --- 6/6: Personalised FedProx (~65 min) ---
echo "" | tee -a $LOG
echo "--- 6/6: Personalised FedProx — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario personalised --n 250 --rounds 20 \
    --base-strategy fedprox --fine-tune-lr 0.001 --mu 0.01 \
    --suffix fedprox_v5_portfolio --pred-suffix $PRED_SUFFIX --horizon 1 2>&1 | tee -a $LOG


echo "" | tee -a $LOG
echo "============================================================" | tee -a $LOG
echo "V5 PORTFOLIO BATCH COMPLETE: $(date)" | tee -a $LOG
echo "============================================================" | tee -a $LOG

# --- Post-run verification ---
echo "" | tee -a $LOG
echo "--- Verification ---" | tee -a $LOG
echo "NPZ files created:" | tee -a $LOG
ls logs/predictions/$PRED_SUFFIX/ | wc -l | tee -a $LOG
echo "CSV files created:" | tee -a $LOG
ls logs/*${SUFFIX}*.csv 2>/dev/null | tee -a $LOG
echo "" | tee -a $LOG
echo "Next: python pipeline/analyze_portfolio_aggregation_v2.py" | tee -a $LOG