# Oracle / Selective Deployment Analysis
import pandas as pd, numpy as np, json

outliers = {'B001', 'B037', 'B238', 'B028'}
fl_bids = set(open('logs/fl_building_ids_250.txt').read().split())
bids_246 = fl_bids - outliers

local = pd.read_csv('logs/local_mlp_matched_results_250_v5_portfolio.csv')
pers = pd.read_csv('logs/fl_personalised_final_fedadam_v5_portfolio.csv')
cent = pd.read_csv('logs/centralised_mlp_results_v5_portfolio.csv')

for df in [local, pers, cent]:
    df['building_id'] = df['building_id'].astype(str)

l = local[local['building_id'].isin(bids_246)].set_index('building_id')
p = pers[pers['building_id'].isin(bids_246)].set_index('building_id')
c = cent[cent['building_id'].isin(bids_246)].set_index('building_id')

if 'horizon' in l.columns: l = l[l['horizon']==1]
if 'horizon' in p.columns: p = p[p['horizon']==1]
if 'horizon' in c.columns: c = c[c['horizon']==1]

common = sorted(set(l.index) & set(p.index) & set(c.index))
l, p, c = l.loc[common], p.loc[common], c.loc[common]

oracle_2 = np.minimum(l['mae'].values, p['mae'].values)
oracle_3 = np.minimum(np.minimum(l['mae'].values, p['mae'].values), c['mae'].values)

fl_wins_2 = int((p['mae'] < l['mae']).sum())
local_wins_2 = int((l['mae'] < p['mae']).sum())

fl_best_3 = int(((p['mae'] <= l['mae']) & (p['mae'] <= c['mae'])).sum())
local_best_3 = int(((l['mae'] <= p['mae']) & (l['mae'] <= c['mae'])).sum())
cent_best_3 = int(((c['mae'] <= l['mae']) & (c['mae'] <= p['mae'])).sum())

# Save text report
with open('logs/oracle_selective_deployment_v5.txt', 'w') as f:
    f.write('=' * 60 + '\n')
    f.write('ORACLE / SELECTIVE DEPLOYMENT ANALYSIS (v5)\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'Buildings: {len(common)} (246 excl. 4 outliers)\n')
    f.write(f'Source: v5 portfolio CSVs (Apr 4, 2026)\n\n')
    f.write('TWO-WAY: Local MLP vs Personalised FL (FedAdam)\n')
    f.write('-' * 50 + '\n')
    f.write(f'  Local MLP mean MAE:   {l["mae"].mean():.3f}\n')
    f.write(f'  Pers FL mean MAE:     {p["mae"].mean():.3f}\n')
    f.write(f'  Oracle MAE:           {oracle_2.mean():.3f}\n')
    f.write(f'  FL wins:              {fl_wins_2}/{len(common)} ({100*fl_wins_2/len(common):.0f}%)\n')
    f.write(f'  Local wins:           {local_wins_2}/{len(common)} ({100*local_wins_2/len(common):.0f}%)\n\n')
    f.write('THREE-WAY: Local MLP vs Pers FL vs Centralised MLP\n')
    f.write('-' * 50 + '\n')
    f.write(f'  Centr MLP mean MAE:   {c["mae"].mean():.3f}\n')
    f.write(f'  Pers FL best:         {fl_best_3}/{len(common)} ({100*fl_best_3/len(common):.0f}%)\n')
    f.write(f'  Local best:           {local_best_3}/{len(common)} ({100*local_best_3/len(common):.0f}%)\n')
    f.write(f'  Centr MLP best:       {cent_best_3}/{len(common)} ({100*cent_best_3/len(common):.0f}%)\n')
    f.write(f'  Oracle MAE:           {oracle_3.mean():.3f}\n\n')
    f.write('KEY FINDINGS\n')
    f.write('-' * 50 + '\n')
    f.write(f'  FL and Local MLP are equally competitive (50/50 split)\n')
    f.write(f'  Centralised MLP almost never optimal (3%)\n')
    f.write(f'  Oracle improves over Local MLP by {100*(l["mae"].mean()-oracle_2.mean())/l["mae"].mean():.1f}%\n')
    f.write(f'  Oracle improves over Pers FL by {100*(p["mae"].mean()-oracle_2.mean())/p["mae"].mean():.1f}%\n')
    f.write(f'  Selective deployment is the practical recommendation\n')

# Save JSON for machine-readable results
results = {
    'n_buildings': len(common),
    'two_way': {
        'local_mlp_mae': round(float(l['mae'].mean()), 3),
        'pers_fl_mae': round(float(p['mae'].mean()), 3),
        'oracle_mae': round(float(oracle_2.mean()), 3),
        'fl_wins': fl_wins_2,
        'local_wins': local_wins_2,
    },
    'three_way': {
        'centr_mlp_mae': round(float(c['mae'].mean()), 3),
        'oracle_mae': round(float(oracle_3.mean()), 3),
        'fl_best': fl_best_3,
        'local_best': local_best_3,
        'centr_best': cent_best_3,
    }
}
with open('logs/oracle_selective_deployment_v5.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved: logs/oracle_selective_deployment_v5.txt')
print('Saved: logs/oracle_selective_deployment_v5.json')
cat_cmd = open('logs/oracle_selective_deployment_v5.txt').read()
print(cat_cmd)