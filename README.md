# das-flwr-heatdemand

**Federated Learning for Demand-Driven Estimation of District Heat Energy Consumption**

A research codebase that trains and evaluates federated, local, and centralized
models for forecasting building-level district-heating demand. Federated training
is built on [Flower](https://flower.ai/) (`flwr`), where **each building is an
independent FL client**.

> Target venue: SCAI 2026.

## Overview

The project compares several training regimes for hourly heat-demand forecasting
across many buildings:

| Scenario          | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| `local_only`      | Each building trains its own model in isolation                    |
| `centralized`     | A single model trained on pooled data from all buildings           |
| `fedavg`          | Federated averaging (McMahan et al., 2017)                         |
| `fedprox`         | FedAvg with a proximal term for client drift                       |
| `personalized_fl` | Federated training followed by per-building fine-tuning            |

Models include a lightweight **MLP** (used for the FL scenarios) and an **XGBoost**
baseline. Forecasts are evaluated at multiple horizons (t+1, t+6, t+24, t+168 hours)
using RMSE, MAE, MAPE, and R².

## Repository layout

```
dasfl/              Flower app (client/server) and shared FL logic
  client_app.py     ClientApp — one building per client
  server_app.py     ServerApp — coordinates rounds, aggregation, eval
  task.py           Shared data/model/train/eval functions
  models.py         MLP architecture
  data_utils.py     Data loading, cleaning, feature engineering
pipeline/           Experiment runners, analysis, and figure/table scripts
  run_fl.py         Main FL experiment orchestrator
  train_*.py        Local / centralized baselines (MLP, XGBoost)
  analyze_*.py      Result analysis
  generate_*.py     Paper figures and tables
configs/
  config.yaml       Central configuration (data, model, FL, evaluation)
Dockerfile          GPU image based on NVIDIA NGC TensorFlow 23.12
run_gpu.sh          Helper to run experiments in the GPU container
requirements.txt    Python dependencies
pyproject.toml      Flower app definition (heatdemand-estimate-app)
```

> **Not included in this repository:** raw/processed datasets (`data/`,
> `benchmark/`), logs (`logs/`), the paper draft (`das-fl-paper/`), and large
> binaries — see [`.gitignore`](.gitignore). Configure data locations under
> `paths:` in [`configs/config.yaml`](configs/config.yaml).

## Setup

Requires Python 3.10+ (the GPU image uses 3.10). Key dependencies:
`flwr[simulation]`, `tensorflow`, `xgboost`, `scikit-learn`, `pandas`, `pyarrow`.

```bash
pip install -e .            # installs the heatdemand-estimate-app package + deps
# or:
pip install -r requirements.txt
```

### GPU (NVIDIA NGC) container

The project is set up to run on the NVIDIA NGC TensorFlow image, which ships a
CUDA-native TensorFlow build:

```bash
docker build -t dasfl-gpu:v1 .
./run_gpu.sh --scenario fedavg --n 100
```

## Usage

Run federated experiments via the orchestrator:

```bash
# Quick smoke test: 20 buildings, 5 rounds, horizon t+1
python pipeline/run_fl.py --scenario fedavg --n 20 --rounds 5 --horizon 1

# Full FedAvg: 100 buildings, 50 rounds, all horizons
python pipeline/run_fl.py --scenario fedavg --n 100

# FedProx / Personalized FL
python pipeline/run_fl.py --scenario fedprox --n 100
python pipeline/run_fl.py --scenario personalised --n 100

# Scalability sweep, or all scenarios sequentially
python pipeline/run_fl.py --scenario scalability
python pipeline/run_fl.py --scenario all --n 100
```

Common flags: `--rounds`, `--horizon`, `--local-epochs`, `--batch-size`, `--lr`,
`--mu` (FedProx), `--seed`, `--data-dir`. See `python pipeline/run_fl.py --help`
for the full list.

Configuration defaults (features, quality filters, train/val/test split, FL
rounds, metrics) live in [`configs/config.yaml`](configs/config.yaml).

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
