# This script computes the mean and standard deviation of MAE and R² for different scenarios, excluding certain building IDs.
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.anonymise_buildings import real_ids

exclude = [int(x) for x in real_ids("B001", "B037", "B238", "B028")]

scenarios = [
    ('logs/centralised_xgboost_results_v2.csv', 'Centr XGB'),
    ('logs/local_mlp_matched_results_250.csv', 'Local MLP'),
    ('logs/fl_personalised_final_fedadam_v4.csv', 'Pers FL FedAdam'),
    ('logs/fl_personalised_final_fedprox_v4.csv', 'Pers FL FedProx'),
    ('logs/local_baseline_results_v2.csv', 'Local XGB'),
    ('logs/centralised_mlp_results_v2.csv', 'Centr MLP'),
    ('logs/fl_fedprox_final_v4.csv', 'FedProx'),
    ('logs/fl_fedavg_final_v4.csv', 'FedAvg'),
    ('logs/fl_fedadam_final_v4.csv', 'FedAdam'),
]

print('=== RESULTS WITH STD (t+1, 246 buildings) ===\n')
print(f'{"Scenario":<22} {"MAE±std":>16} {"Median":>8} {"R²±std":>16} {"Median":>8}')
print('-' * 75)

for f, name in scenarios:
    try:
        df = pd.read_csv(f)
        df = df[(df['horizon']==1) & (~df['building_id'].isin(exclude))]
        print(f'{name:<22} {df["mae"].mean():.3f}±{df["mae"].std():.3f} '
              f'{df["mae"].median():>8.3f} '
              f'{df["r2"].mean():.3f}±{df["r2"].std():.3f} '
              f'{df["r2"].median():>8.3f}')
    except:
        print(f'{name:<22} NOT FOUND')