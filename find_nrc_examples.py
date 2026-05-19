import pandas as pd
import numpy as np

print("Loading predictions.csv...")
df = pd.read_csv("predictions.csv", dtype={"SITEREF": str})
df["DATETIME_HOUR"] = pd.to_datetime(df["DATETIME_HOUR"])
df["DATE"] = df["DATETIME_HOUR"].dt.date
df["HOUR"] = df["DATETIME_HOUR"].dt.hour
# Clip negative predictions — physically impossible for traffic flow
df["Predicted_Flow"] = df["Predicted_Flow"].clip(lower=0)
print(f"Loaded {len(df):,} rows\n")

# Load meta for region lookup
meta_df = pd.read_pickle("step_b1_meta.pkl")

def get_region(siteref):
    match = meta_df[meta_df["SITEREF"] == str(siteref)]
    if not match.empty:
        region = str(match.iloc[0]["REGION"])
        return region.split(" - ")[-1] if " - " in region else region
    # Fallback to REGION_NAME in df if present
    return "Unknown"

# --- Fix: convert all NRC/event/holiday columns to numeric ---
lockdown_cols = ["LOCKDOWN_L4_AUCKLAND", "LOCKDOWN_L3_AUCKLAND",
                 "LOCKDOWN_L4_NATIONAL", "LOCKDOWN_L3_NATIONAL",
                 "LOCKDOWN_L2_NATIONAL", "EVENT_Lockdown_Announced"]
extreme_cols  = [c for c in df.columns if c.startswith("EVENT_") and "Lockdown" not in c]
holiday_cols  = [c for c in df.columns if c.startswith("HOLIDAY_")]
weather_cols  = ["TEMP", "RH", "WDSP", "DEWP", "VISIB", "GUST"]

# Convert to numeric safely
all_nrc = lockdown_cols + extreme_cols + holiday_cols
for col in all_nrc:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

# Helper to print a scenario row
def print_scenario(label, row, note, active_cols=None):
    ts = row["DATETIME_HOUR"]
    region = get_region(row["SITEREF"])
    direction = "Incoming" if int(row.get("IS_DIRECTION_2", 0)) == 1 else "Outgoing"
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Region   : {region}")
    print(f"  Siteref  : {row['SITEREF']}")
    print(f"  Direction: {direction}")
    print(f"  Date     : {ts.strftime('%Y-%m-%d (%A)')}")
    print(f"  Time     : {ts.strftime('%H:%M')} — select {int(row['HOUR'])}:00 in dashboard")
    print(f"  Pred Flow: {row['Predicted_Flow']:.0f} vph")
    if active_cols:
        active = [c for c in active_cols if row.get(c, 0) == 1]
        print(f"  Active NRC flags: {active}")
    print(f"  Note     : {note}")

# ===== 1. LOCKDOWN =====
print("\n" + "="*65)
print("SCANNING: 1. LOCKDOWN")
lockdown_mask = df[lockdown_cols].max(axis=1) == 1
lockdown_df   = df[lockdown_mask & df["HOUR"].isin([8, 9, 17, 18])].copy()
lockdown_df   = lockdown_df.sort_values("Predicted_Flow", ascending=True)
print(f"  Found {lockdown_mask.sum():,} lockdown rows | {len(lockdown_df):,} at peak hours")
if len(lockdown_df) > 0:
    print_scenario("1. LOCKDOWN", lockdown_df.iloc[0],
                   "Lockdown suppresses AM/PM peak heavily — great SHAP demo",
                   active_cols=lockdown_cols)

# ===== 2. EXTREME EVENT =====
print("\n" + "="*65)
print("SCANNING: 2. EXTREME WEATHER EVENT")
extreme_mask = df[extreme_cols].max(axis=1) == 1
extreme_df   = df[extreme_mask & df["HOUR"].isin([8, 9, 17, 18])].copy()
extreme_df   = extreme_df.sort_values("Predicted_Flow", ascending=True)
print(f"  Found {extreme_mask.sum():,} extreme event rows | {len(extreme_df):,} at peak hours")

# One per event type for variety
if "EXTREME_ID" in df.columns:
    extreme_df_all = df[extreme_mask].copy()
    for etype in extreme_cols:
        subset = extreme_df_all[extreme_df_all[etype] == 1]
        if len(subset) > 0:
            row = subset.sort_values("Predicted_Flow").iloc[0]
            print_scenario(f"2. EXTREME EVENT ({etype.replace('EVENT_','')})", row,
                           f"Active flag: {etype}", active_cols=extreme_cols)
else:
    if len(extreme_df) > 0:
        print_scenario("2. EXTREME WEATHER EVENT", extreme_df.iloc[0],
                       "Extreme event suppresses traffic — check EVENT_ SHAP bars",
                       active_cols=extreme_cols)

# ===== 3. HOLIDAY =====
print("\n" + "="*65)
print("SCANNING: 3. HOLIDAYS")

def get_holiday_name(row):
    for col in holiday_cols:
        if row[col] == 1:
            return col.replace("HOLIDAY_", "").replace("_", " ")
    return "Unknown"

holiday_mask = df[holiday_cols].max(axis=1) == 1
holiday_df   = df[holiday_mask & df["HOUR"].isin([10, 11, 12])].copy()
holiday_df["HOLIDAY_NAME_CLEAN"] = holiday_df.apply(get_holiday_name, axis=1)
holiday_df   = holiday_df.sort_values("Predicted_Flow", ascending=True)
print(f"  Found {holiday_mask.sum():,} holiday rows | {len(holiday_df):,} at midday hours")

seen = set()
for _, row in holiday_df.iterrows():
    hname = row["HOLIDAY_NAME_CLEAN"]
    if hname not in seen:
        seen.add(hname)
        print_scenario(f"3. HOLIDAY ({hname})", row,
                       f"Holiday reduces weekday traffic — check HOLIDAY_{hname.replace(' ','_')} SHAP",
                       active_cols=holiday_cols)
    if len(seen) >= 4:  # Show top 4 different holidays
        break

# ===== 4. BAD WEATHER =====
print("\n" + "="*65)
print("SCANNING: 4. EXTREME WEATHER CONDITIONS")

low_temp_thresh  = df["TEMP"].quantile(0.05)
high_gust_thresh = df["GUST"].quantile(0.95)
low_vis_thresh   = df["VISIB"].quantile(0.05)

print(f"  Thresholds: TEMP ≤ {low_temp_thresh:.1f}°C | GUST ≥ {high_gust_thresh:.1f}km/h | VISIB ≤ {low_vis_thresh:.1f}km")

# High Gust
gust_df = df[(df["GUST"] >= high_gust_thresh) & df["HOUR"].isin([8,9,17,18])].sort_values("GUST", ascending=False)
if len(gust_df) > 0:
    print_scenario("4a. HIGH GUST", gust_df.iloc[0],
                   f"GUST = {gust_df.iloc[0]['GUST']:.1f} km/h — check GUST SHAP bar")

# Low Visibility
vis_df = df[(df["VISIB"] <= low_vis_thresh) & df["HOUR"].isin([8,9,17,18])].sort_values("VISIB", ascending=True)
if len(vis_df) > 0:
    print_scenario("4b. LOW VISIBILITY", vis_df.iloc[0],
                   f"VISIB = {vis_df.iloc[0]['VISIB']:.1f} km — check VISIB SHAP bar")

# Low Temperature
temp_df = df[(df["TEMP"] <= low_temp_thresh) & df["HOUR"].isin([8,9,17,18])].sort_values("TEMP", ascending=True)
if len(temp_df) > 0:
    print_scenario("4c. LOW TEMPERATURE", temp_df.iloc[0],
                   f"TEMP = {temp_df.iloc[0]['TEMP']:.1f}°C — check TEMP SHAP bar")
