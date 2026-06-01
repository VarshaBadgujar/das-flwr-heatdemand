"""Rerun Local MLP on the same 100 buildings used for FL."""
import pandas as pd
import numpy as np
import sys
import os
import gc
import time
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dasfl.task import build_model, load_client_data, get_building_ids

# Load FL building list
fl_buildings = open("logs/fl_building_ids.txt").read().strip().split("\n")
print(f"FL buildings to evaluate: {len(fl_buildings)}")

HORIZONS = [1, 6, 24, 168]
results = []
start = time.time()

for horizon in HORIZONS:
    print(f"\n--- Horizon t+{horizon} ---")
    
    for i, bid in enumerate(fl_buildings):
        try:
            data = load_client_data(i, fl_buildings, horizon=horizon)
            
            model = build_model(data["n_features"], [64, 32], 0.2, 0.001)
            
            history = model.fit(
                data["X_train"], data["y_train"],
                validation_data=(data["X_val"], data["y_val"]),
                epochs=30,
                batch_size=64,
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                )],
            )
            
            epochs_trained = len(history.history["loss"])
            
            y_pred = model.predict(data["X_test"], verbose=0).flatten()
            y_pred = np.clip(y_pred, 0, None)
            y_test = data["y_test"]
            
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            ss_res = np.sum((y_test - y_pred)**2)
            ss_tot = np.sum((y_test - np.mean(y_test))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            mape_mask = y_test > 0.5
            mape = np.mean(np.abs((y_test[mape_mask] - y_pred[mape_mask]) / y_test[mape_mask])) * 100 if mape_mask.sum() > 10 else np.nan
            
            results.append({
                "building_id": bid, "horizon": horizon, "status": "success",
                "mae": round(mae, 4), "rmse": round(rmse, 4),
                "mape": round(mape, 2) if not np.isnan(mape) else np.nan,
                "r2": round(r2, 4), "epochs": epochs_trained,
            })
            
            tf.keras.backend.clear_session()
            del model
            gc.collect()
            
        except Exception as e:
            results.append({"building_id": bid, "horizon": horizon, 
                          "status": "error", "error": str(e)})
        
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{len(fl_buildings)} done")

elapsed = time.time() - start
df = pd.DataFrame(results)
df.to_csv("logs/local_mlp_matched_results.csv", index=False)

print(f"\n{'='*60}")
print(f"LOCAL MLP — MATCHED TO FL BUILDINGS")
print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"{'='*60}")

df_ok = df[df["status"] == "success"]
for h in sorted(df_ok["horizon"].unique()):
    hdf = df_ok[df_ok["horizon"] == h]
    print(f"  t+{h:<4}  MAE={hdf['mae'].median():.3f}  "
          f"R²={hdf['r2'].median():.3f}  N={len(hdf)}")
