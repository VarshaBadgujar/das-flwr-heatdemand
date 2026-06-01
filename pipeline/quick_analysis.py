import pandas as pd
import numpy as np

exclude = [B001, B037, B238, B028]

local = pd.read_csv('logs/local_mlp_matched_results_250.csv')
pers = pd.read_csv('logs/fl_personalised_final_fedadam_v4.csv')
central = pd.read_csv('logs/centralised_mlp_results_v2.csv')

l = local[local['horizon']==1].set_index('building_id')
p = pers[pers['horizon']==1].set_index('building_id')
c = central[central['horizon']==1].set_index('building_id')

common = [b for b in l.index.intersection(p.index).intersection(c.index) 
          if b not in exclude]

l, p, c = l.loc[common], p.loc[common], c.loc[common]

fl_best = ((p['mae'] <= l['mae']) & (p['mae'] <= c['mae'])).sum()
local_best = ((l['mae'] <= p['mae']) & (l['mae'] <= c['mae'])).sum()
central_best = ((c['mae'] <= l['mae']) & (c['mae'] <= p['mae'])).sum()

print(f"=== THREE-WAY COMPARISON (246 buildings) ===")
print(f"Pers FL wins:     {fl_best} ({fl_best/len(common)*100:.0f}%)")
print(f"Local MLP wins:   {local_best} ({local_best/len(common)*100:.0f}%)")
print(f"Centralised wins: {central_best} ({central_best/len(common)*100:.0f}%)")

oracle = np.minimum(np.minimum(l['mae'], p['mae']), c['mae'])
print(f"\nOracle (best-of-three): MAE={oracle.mean():.3f}")
print(f"vs Local MLP:           MAE={l['mae'].mean():.3f}")
print(f"vs Pers FL:             MAE={p['mae'].mean():.3f}")
print(f"vs Centralised MLP:     MAE={c['mae'].mean():.3f}")
