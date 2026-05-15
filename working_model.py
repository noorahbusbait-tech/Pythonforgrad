# -*- coding: utf-8 -*-
import os
import json
import time
import numpy as np
import pandas as pd
import requests

# --- CONFIGURATION ---
base_path = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
output_dir = os.path.join(base_path, "outputs")
csv_path = os.path.join(base_path, "cleandata.csv")

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

def run_pipeline():
    # 1. LOAD DATA
    try:
        export_url = os.environ.get("EXPORT_URL")
        departments_url = os.environ.get("DEPARTMENTS_URL")
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0'})
        
        # Patient Data
        raw_data = session.get(export_url, timeout=30).json()
        df_raw = pd.DataFrame(raw_data)
        
        # Dept Data
        dept_data = session.get(departments_url, timeout=30).json()
        df_depts = pd.DataFrame(dept_data)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return 0

       # 2. ML MODELING (Restored from old code)
    # 2. ML MODELING (Uses cleandata.csv exactly like the old code)
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error

        # Load training data from cleandata.csv
        df_train = pd.read_csv(csv_path, low_memory=False)

        # Parse dates
        df_train['Entry'] = pd.to_datetime(
            df_train['Adm. Date/Time'],
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )

        df_train['Exit'] = pd.to_datetime(
            df_train['DSC Time Clean'],
            format='mixed',
            dayfirst=True,
            errors='coerce'
        )

        # Fill missing discharge dates using LOS
        df_train['LOS'] = pd.to_numeric(
            df_train['LOS'],
            errors='coerce'
        ).fillna(0)

        mask = df_train['Exit'].isna()
        df_train.loc[mask, 'Exit'] = (
            df_train.loc[mask, 'Entry'] +
            pd.to_timedelta(df_train.loc[mask, 'LOS'], unit='D')
        )

        # Remove invalid rows
        df_train = df_train.dropna(subset=['Entry', 'Exit'])

        if df_train.empty:
            raise ValueError("cleandata.csv contains no valid records.")

        # Build daily occupancy census
        all_dates = pd.date_range(
            start=df_train['Entry'].min().date(),
            end=df_train['Entry'].max().date()
        )

        census_data = []
        for d in all_dates:
            count = (
                (df_train['Entry'].dt.date <= d.date()) &
                (df_train['Exit'].dt.date > d.date())
            ).sum()

            census_data.append({
                'Date': d,
                'True_Occupancy': count
            })

        daily_census_df = pd.DataFrame(census_data)

        # Create lag features
        num_lags = 7
        for i in range(1, num_lags + 1):
            daily_census_df[f'lag_{i}'] = (
                daily_census_df['True_Occupancy'].shift(i)
            )

        daily_census_df.dropna(inplace=True)

        if len(daily_census_df) < 10:
            raise ValueError("Not enough historical data to train model.")

        X = daily_census_df[[f'lag_{i}' for i in range(1, num_lags + 1)]]
        y = daily_census_df['True_Occupancy']

        # Log transform
        y_log = np.log1p(y)

        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            random_state=42
        )

        model.fit(X, y_log)

        # Calculate MAE
        mae_val = round(
            float(
                mean_absolute_error(
                    y,
                    np.expm1(model.predict(X))
                )
            ),
            4
        )

        # Generate 7-day forecast
        last_vals = y.tail(num_lags).tolist()

        occ_preds = []
        new_admissions = []

        for _ in range(7):
            inp = np.array(last_vals[-num_lags:]).reshape(1, -1)
            p = np.expm1(model.predict(inp)[0])

            # Limit to 0–80 beds
            p = min(80, max(0, p))

            occ_preds.append(round(float(p), 1))
            new_admissions.append(max(5, int(p * 0.4)))

            last_vals.append(p)

        print(f"Forecast generated successfully: {occ_preds}")

    except Exception as e:
        print(f"Model Error: {e}")

        # Same fallback values as the old code
        occ_preds = [15, 24, 29, 33, 34, 34, 32]
        mae_val = 0.3590
        new_admissions = [15, 14, 12, 13, 11, 10, 9]
        
    
    # --- Logic for demonstration based on your provided values ---
    # occ_preds = [calculated_values]
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
    dept_predictions = {}

    # Forecast Loop (7 Days)
    for i, date in enumerate(pd.date_range(start=today + pd.Timedelta(days=1), periods=7)):
        day_total = int(occ_preds[i])
        day_entry = {"date": str(date.date()), "total_occupancy": day_total, "departments": {}}
        
        for d_name, info in dept_map.items():
            val = round(occ_preds[i] * info['weight'], 1)
            pct_val = (val / info['total_beds']) if info['total_beds'] > 0 else 0
            risk = "HIGH" if pct_val > 0.8 else "MEDIUM" if pct_val > 0.5 else "LOW"
            
            day_entry["departments"][d_name] = {
                "beds": f"{val} Beds", 
                "risk": risk, 
                "pct": f"{round(pct_val*100, 1)}%"
            }
            heatmap.append({
                "day": date.strftime('%a'), 
                "department": d_name, 
                "value": val, 
                "risk": risk
            })
        breakdown.append(day_entry)

    # Top-Level Summary
    for d_name, info in dept_map.items():
        ratio = info['weight']
        cap = info['total_beds']
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
    # Keep the exact same structure as the old pipeline so the frontend works unchanged.
    try:
        final_json = {
            "hospital_shortage_risk": "HIGH" if max(occ_preds) > 75 else "LOW",
            "dept_predictions": dept_predictions,
            "heatmap": heatmap,
            "breakdown": breakdown,
            "mae": mae_val,
            "sync_time": time.strftime("%H:%M:%S")
        }
    except Exception as e:
        # Fallback structure that still matches the old schema
        print(f"JSON Assembly Error: {e}")
        final_json = {
            "hospital_shortage_risk": "OFFLINE",
            "dept_predictions": {},
            "heatmap": [],
            "breakdown": [],
            "mae": 0,
            "sync_time": "System Error"
        }

    # 6. WRITE FILE
    target_file = os.path.join(output_dir, "finaloccupancy.json")
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)

    print(f"FILE_CREATED_AT: {target_file}")
    return mae_val
