#!/bin/bash
set -e
LOG="logs/scalability_v5_complete_log.txt"
echo "=== SCALABILITY V5 START: $(date) ===" | tee $LOG

# Part 1: FedProx standalone at all K
for K in 20 50 100 200 500 988; do
  echo "--- FedProx K=$K — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario fedprox --n $K --rounds 20 --mu 0.01 \
    --suffix scalability_k${K}_v5 --horizon 1 2>&1 | tee -a $LOG
done

# Part 2: Pers FedProx at all K
for K in 20 50 100 200 500 988; do
  echo "--- Pers FedProx K=$K — $(date) ---" | tee -a $LOG
  bash run_gpu.sh --scenario personalised --n $K --rounds 20 \
    --base-strategy fedprox --fine-tune-lr 0.001 --mu 0.01 \
    --suffix scalability_k${K}_v5 --horizon 1 2>&1 | tee -a $LOG
done

# Part 3: K=988 for remaining strategies
echo "--- FedAvg K=988 — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario fedavg --n 988 --rounds 20 \
  --suffix scalability_k988_v5 --horizon 1 2>&1 | tee -a $LOG

echo "--- FedAdam K=988 — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario fedadam --n 988 --rounds 20 \
  --suffix scalability_k988_v5 --horizon 1 2>&1 | tee -a $LOG

echo "--- Pers FedAdam K=988 — $(date) ---" | tee -a $LOG
bash run_gpu.sh --scenario personalised --n 988 --rounds 20 \
  --base-strategy fedadam --fine-tune-lr 0.001 \
  --suffix scalability_k988_v5 --horizon 1 2>&1 | tee -a $LOG

echo "=== SCALABILITY V5 COMPLETE: $(date) ===" | tee -a $LOG
