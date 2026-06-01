#!/bin/bash
set -e
LOG="logs/v5_multihorizon_log.txt"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="dasfl-gpu:v1"

echo "=== V5 MULTI-HORIZON START: $(date) ===" | tee $LOG

for H in 6 24 168; do
  echo "" | tee -a $LOG
  echo "========== HORIZON h=$H ==========" | tee -a $LOG

  echo "--- Local MLP h=$H — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario local_mlp --n 250 --horizon $H \
    --suffix v5_portfolio_h${H} 2>&1 | tee -a $LOG

  echo "--- FedAvg h=$H — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario fedavg --n 250 --rounds 20 \
    --suffix v5_portfolio_h${H} --horizon $H 2>&1 | tee -a $LOG

  echo "--- FedAdam h=$H — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario fedadam --n 250 --rounds 20 \
    --suffix v5_portfolio_h${H} --horizon $H 2>&1 | tee -a $LOG

  echo "--- FedProx h=$H — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario fedprox --n 250 --rounds 20 --mu 0.01 \
    --suffix v5_portfolio_h${H} --horizon $H 2>&1 | tee -a $LOG

  echo "--- Pers FedAdam h=$H — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario personalised --n 250 --rounds 20 \
    --base-strategy fedadam --fine-tune-lr 0.001 \
    --suffix fedadam_v5_portfolio_h${H} --horizon $H 2>&1 | tee -a $LOG

  echo "--- Pers FedProx h=$H — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario personalised --n 250 --rounds 20 \
    --base-strategy fedprox --fine-tune-lr 0.001 --mu 0.01 \
    --suffix fedprox_v5_portfolio_h${H} --horizon $H 2>&1 | tee -a $LOG

  echo "--- Centralised MLP h=$H — $(date) ---" | tee -a $LOG
  docker run --gpus all --rm \
    -v ${PROJECT}:/workspace -w /workspace ${IMAGE} \
    python pipeline/train_centralised_mlp.py \
    --building-ids-file logs/fl_building_ids_250.txt \
    --horizon $H --epochs 50 --patience 5 --batch-size 1024 \
    --suffix v5_portfolio_h${H} 2>&1 | tee -a $LOG

done

echo "" | tee -a $LOG
echo "=== V5 MULTI-HORIZON COMPLETE: $(date) ===" | tee -a $LOG
