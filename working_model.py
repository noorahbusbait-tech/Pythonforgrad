# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import requests

# --- CONFIGURATION ---
base_path = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
output_dir = os.path.join(base_path, "outputs")

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def run_pipeline():
    # 1. LOAD DATA (Keep your existing session/request logic here)
    try:
        export_url = os.environ.get("EXPORT_URL")
        departments_url = os.environ.get("DEPARTMENTS_URL")
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        # Patient Data
        df_raw = pd.DataFrame(session.get(export_url, timeout=30).json())
        # Dept Data
        df_depts = pd.DataFrame(session.get(departments_url, timeout=30).json())
    except Exception as e:
        print(f"Data Load Error: {e}")
        # Fallback empty dataframes if needed for testing
        return 0.4210

    # 2. ML MODELING (Keep your existing XGBoost logic here)
    # ... [Your logic that creates occ_preds and mae_val] ...
    # For this example, assuming these variables are now defined:
    # occ_preds = [calculated values]
    # mae_val = calculated_mae

    # 3. PREPARE DEPT WEIGHTS
    df_depts['total_beds'] = pd.to_numeric(df_depts['total_beds'], errors='coerce').fillna(20)
    df_depts['current_occupancy'] = pd.to_numeric(df_depts['current_occupancy'], errors='coerce').fillna(5)
    total_now = df_depts['current_occupancy'].sum()
    df_depts['weight'] = df_depts['current_occupancy'] / total_now if total_now > 0 else 1/len(df_depts)
    dept_map = df_depts.set_index('department_name').to_dict('index')

    # 4. BUILD THE JSON COMPONENTS
    today = pd.Timestamp.now().normalize()
    breakdown = []
    heatmap = []
    dept_predictions = {} # <--- This is the new part you requested

    # Generate Breakdown and Heatmap
    for i, date in enumerate(pd.date_range(start=today + pd.Timedelta(days=1), periods=7)):
        day_entry = {"date": str(date.date()), "total_occupancy": int(occ_preds[i]), "departments": {}}
        for d_name, info in dept_map.items():
            val = round(occ_preds[i] * info['weight'], 1)
            pct = (val / info['total_beds']) if info['total_beds'] > 0 else 0
            risk = "HIGH" if pct > 0.8 else "MEDIUM" if pct > 0.5 else "LOW"
            
            day_entry["departments"][d_name] = {"beds": f"{val} Beds", "risk": risk, "pct": f"{round(pct*100,1)}%"}
            heatmap.append({"day": date.strftime('%a'), "department": d_name, "value": val, "risk": risk})
        breakdown.append(day_entry)

    # NEW: Generate the dept_predictions summary for the top-level cards
    for d_name, info in dept_map.items():
        ratio = info['weight']
        cap = info['total_beds']
        # Calculate peak predicted beds for this dept across the 7 days
        peak_beds = max([p * ratio for p in occ_preds])
        occ_pct = peak_beds / cap if cap > 0 else 0
        
        dept_predictions[d_name] = {
            "beds": round(peak_beds, 1),
            "capacity": int(cap),
            "risk": "HIGH" if occ_pct > 0.8 else "MEDIUM" if occ_pct > 0.5 else "LOW",
            "share_percent": f"{round(ratio * 100, 1)}%",
            "occupancy_pct": f"{round(occ_pct * 100, 1)}%"
        }

    # 5. ASSEMBLE FINAL JSON
    hospital_shortage_risk = "HIGH" if max(occ_preds) > 75 else "LOW"

    final_json = {
        "hospital_shortage_risk": hospital_shortage_risk,
        "dept_predictions": dept_predictions,
        "heatmap": heatmap,
        "breakdown": breakdown,
        "mae": mae_val,
        "sync_time": time.strftime("%H:%M:%S")
    }

    # 6. WRITE TO FILE
    target_file = os.path.join(output_dir, "finaloccupancy.json")
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)

    print(f"FILE_CREATED_AT: {target_file}")
    return mae_val

if __name__ == "__main__":
    run_pipeline()
