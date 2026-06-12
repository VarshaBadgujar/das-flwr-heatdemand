# das-flwr-heatdemand

**Personalised Federated Learning for District Heating Demand Estimation under Portfolio Heterogeneity**

A research codebase that trains and evaluates federated, local, and centralized
models for forecasting building-level district-heating demand. Federated training
is built on [Flower](https://flower.ai/) (`flwr`), where **each building is an
independent FL client**.

> 📄 **Paper accepted at SCAI 2026** (Scandinavian Conference on Artificial
> Intelligence, Odense, Denmark, 15–16 June 2026), to appear in Springer
> *Communications in Computer and Information Science* (CCIS).
> Author-accepted version: [`paper/`](paper/) · Citation: see [below](#citation).

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
paper/              Author-accepted version of the SCAI 2026 paper
configs/
  config.yaml       Central configuration (data, model, FL, evaluation)
Dockerfile          GPU image based on NVIDIA NGC TensorFlow 23.12
run_gpu.sh          Helper to run experiments in the GPU container
requirements.txt    Python dependencies
pyproject.toml      Flower app definition (heatdemand-estimate-app)
```

> **Not included in this repository:** raw/processed datasets (`data/`,
> `benchmark/`), logs (`logs/`), and large binaries — see
> [`.gitignore`](.gitignore). Configure data locations under `paths:` in
> [`configs/config.yaml`](configs/config.yaml).

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

# FedProx / Personalized FL
python pipeline/run_fl.py --scenario fedprox --n 100
python pipeline/run_fl.py --scenario personalised --n 100

# Scalability sweep, or all scenarios sequentially
python pipeline/run_fl.py --scenario scalability
python pipeline/run_fl.py --scenario all --n 100
```

Common flags: `--rounds`, `--horizon`, `--local-epochs`, `--batch-size`, `--lr`,
`--mu` (FedProx), `--seed`, `--data-dir`, `--suffix`. See
`python pipeline/run_fl.py --help` for the full list.

Configuration defaults (features, quality filters, train/val/test split, FL
rounds, metrics) live in [`configs/config.yaml`](configs/config.yaml).

> **Note:** `--scenario scalability` covers FedAvg, FedAdam, and personalised
> FedAdam only. FedProx variants require individual scenario calls; pass an
> explicit `--suffix` to avoid overwriting result files.

## Reproducing the paper experiments

The shell scripts under `pipeline/` are the entry points used for the
experiments reported in the paper:

| Script | Paper experiments |
| --- | --- |
| `pipeline/run_v5_portfolio.sh` | Main benchmark and portfolio-level evaluation |
| `pipeline/run_v5_multihorizon.sh` | Multi-horizon evaluation (t+1, t+6, t+24, t+168) |
| `pipeline/run_v5_scalability_complete.sh` | Scalability sweep across portfolio sizes |
| `pipeline/wilcoxon_multiseed.py`, `pipeline/compute_multiseed_stats.py` | Multi-seed statistics and significance tests |
| `pipeline/simulate_rolling_168h.py` | Rolling weekly (168 h) forecast simulation |

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{badgujar2026personalised,
  author    = {Badgujar, Varsha Kiran and Mbiydzenyuy, Gideon},
  title     = {Personalised Federated Learning for District Heating Demand
               Estimation under Portfolio Heterogeneity},
  booktitle = {Scandinavian Conference on Artificial Intelligence (SCAI 2026)},
  series    = {Communications in Computer and Information Science},
  publisher = {Springer},
  address   = {Odense, Denmark},
  year      = {2026},
  note      = {To appear}
}
```

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).
The author-accepted version of the paper is available in [`paper/`](paper/);
the final authenticated version will be available on SpringerLink (the DOI
will be added here once the proceedings are published).

## Data availability

The smart-meter dataset used in this study was provided by an industrial
partner under a data-sharing agreement and cannot be redistributed. The
pipeline expects hourly substation meter readings plus outdoor temperature;
see `dasfl/data_utils.py` and `configs/config.yaml` for the expected schema,
which allows the code to be adapted to other district-heating datasets.

## Acknowledgements

This work is part of the project *Data Analytics for Peak Load Stabilization
in District Heating Networks* (DAS), funded by the Swedish Knowledge
Foundation (KKS), grant Dnr. 20230101. We thank our industrial partners and
colleagues at the University of Borås.

## Contact

- Varsha Kiran Badgujar — `extern-varsha_kiran.badgujar@hb.se` ([ORCID 0009-0009-7450-4563](https://orcid.org/0009-0009-7450-4563))
- Gideon Mbiydzenyuy — `gideon.mbiydzenyuy@hb.se` ([ORCID 0000-0002-9685-7775](https://orcid.org/0000-0002-9685-7775))

## License

Code is licensed under the Apache License 2.0 — see [LICENSE](LICENSE).
The paper PDF in `paper/` is **not** covered by this license; see
[`paper/README.md`](paper/README.md) for its copyright notice.
