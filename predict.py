import polars as pl
import numpy as np
import pandas as pd
import joblib
import os

print("Loading data...")
df = pl.read_csv(
    "cleaned_master_v2.csv",
    schema_overrides={"SITEREF": pl.String, "SH": pl.String}
)

print("Parsing DATETIME and extracting time features...")
df = df.with_columns(
    pl.col("DATETIME").str.to_datetime()
)

df = df.with_columns([
    pl.col("DATETIME").dt.hour().alias("HOUR"),
    pl.col("DATETIME").dt.weekday().alias("DAY_OF_WEEK"),
    pl.col("DATETIME").dt.month().alias("MONTH"),
    pl.col("DATETIME").dt.year().alias("YEAR")
])

print("Encoding traffic direction...")
df = df.with_columns(
    pl.when(pl.col('UNIQUE_ID').str.ends_with('_2'))
    .then(pl.lit(1))
    .otherwise(pl.lit(0))
    .cast(pl.Int8)
    .alias('IS_DIRECTION_2')
)

print("Aggregating 15-min records to hourly...")
df = df.with_columns(
    pl.col('DATETIME').dt.truncate('1h').alias('DATETIME_HOUR')
)

df_hourly = df.group_by(['DATETIME_HOUR', 'UNIQUE_ID', 'SITEREF']).agg([
    pl.col('FLOW').sum().alias('FLOW'),
    pl.col('TEMP').mean(),
    pl.col('RH').mean(),
    pl.col('WDSP').mean(),
    pl.col('DEWP').mean(),
    pl.col('VISIB').mean(),
    pl.col('GUST').mean(),
    pl.col('IS_EXTREME').max(),
    pl.col('IS_HOLIDAY').max(),
    pl.col('LOCKDOWN_L4_AUCKLAND').max(),
    pl.col('LOCKDOWN_L3_AUCKLAND').max(),
    pl.col('LOCKDOWN_L4_NATIONAL').max(),
    pl.col('LOCKDOWN_L3_NATIONAL').max(),
    pl.col('LOCKDOWN_L2_NATIONAL').max(),
    pl.col('EVENT_Lockdown_Announced').max(),
    pl.col('IS_DIRECTION_2').max(),
    pl.col('EXTREME_HAZARD').first(),
    pl.col('EXTREME_ID').first(),
    pl.col('HOLIDAY_NAME').first(),
    pl.col('HOUR').first(),
    pl.col('DAY_OF_WEEK').first(),
    pl.col('MONTH').first(),
    pl.col('YEAR').first(),
    pl.col('REGION').first(),
    pl.col('SH').first(),
    pl.col('LANE').first(),
]).sort(['UNIQUE_ID', 'DATETIME_HOUR'])

df_hourly = df_hourly.with_columns(
    pl.when(pl.col('DAY_OF_WEEK') >= 6)
    .then(1).otherwise(0)
    .alias('IS_WEEKEND')
)

df_hourly = df_hourly.with_columns(
    pl.col('REGION').str.replace(r'^\d+ - ', '').alias('REGION_NAME')
).with_columns([
    (pl.col('REGION_NAME') == 'Auckland').cast(pl.Int8).alias('IS_AUCKLAND'),
    (pl.col('REGION_NAME') == 'Wellington').cast(pl.Int8).alias('IS_WELLINGTON'),
    (pl.col('REGION_NAME') == 'Canterbury').cast(pl.Int8).alias('IS_CANTERBURY'),
])

print("Encoding holidays and extreme events...")
unique_holidays = (
    df.filter((pl.col('IS_HOLIDAY') == 1) & (pl.col('HOLIDAY_NAME').is_not_null()) & (pl.col('HOLIDAY_NAME') != 'None'))
    ['HOLIDAY_NAME'].unique().to_list()
)
unique_extremes = (
    df.filter((pl.col('IS_EXTREME') == 1) & (pl.col('EXTREME_HAZARD') != 'None'))
    ['EXTREME_HAZARD'].unique().to_list()
)

one_hot_exprs = []
for holiday in unique_holidays:
    clean_name = holiday.replace(" ", "_").replace("'", "")
    one_hot_exprs.append(
        pl.when((pl.col('IS_HOLIDAY') == 1) & (pl.col('HOLIDAY_NAME') == holiday))
        .then(1).otherwise(0).alias(f'HOLIDAY_{clean_name}')
    )
for extreme_id in unique_extremes:
    one_hot_exprs.append(
        pl.when((pl.col('IS_EXTREME') == 1) & (pl.col('EXTREME_HAZARD') == extreme_id))
        .then(1).otherwise(0).alias(f'EVENT_{extreme_id}')
    )

df_hourly = df_hourly.with_columns(one_hot_exprs)

print("Cyclical time encoding and lag features...")
df_hourly = df_hourly.with_columns([
    (2 * np.pi * pl.col("HOUR") / 24).sin().alias("HOUR_SIN"),
    (2 * np.pi * pl.col("HOUR") / 24).cos().alias("HOUR_COS"),
    (2 * np.pi * pl.col("DAY_OF_WEEK") / 7).sin().alias("DAY_SIN"),
    (2 * np.pi * pl.col("DAY_OF_WEEK") / 7).cos().alias("DAY_COS"),
    (2 * np.pi * pl.col("MONTH") / 12).sin().alias("MONTH_SIN"),
    (2 * np.pi * pl.col("MONTH") / 12).cos().alias("MONTH_COS")
])

df_hourly = df_hourly.with_columns([
    pl.col('FLOW').shift(1).over('UNIQUE_ID').alias('FLOW_lag_1h'),
    pl.col('FLOW').shift(24).over('UNIQUE_ID').alias('FLOW_lag_24h'),
    pl.col('FLOW').shift(168).over('UNIQUE_ID').alias('FLOW_lag_168h'),
    pl.col('FLOW').shift(1).rolling_mean(window_size=4).over('UNIQUE_ID').alias('FLOW_roll_mean_4h'),
])

WEATHER_FEATURES  = ['TEMP', 'RH', 'WDSP', 'DEWP', 'VISIB', 'GUST']

NRC_FEATURES = [
    'LOCKDOWN_L4_AUCKLAND', 'LOCKDOWN_L3_AUCKLAND', 'LOCKDOWN_L4_NATIONAL', 
    'LOCKDOWN_L3_NATIONAL', 'LOCKDOWN_L2_NATIONAL', 'EVENT_Lockdown_Announced', 
    'IS_WEEKEND',
    'HOLIDAY_ANZAC_Day', 'HOLIDAY_Christmas_Day', 'HOLIDAY_New_Years_Day',
    'HOLIDAY_Boxing_Day', 'HOLIDAY_Waitangi_Day', 'HOLIDAY_Day_after_New_Years_Day',
    'HOLIDAY_Queens_Birthday', 'HOLIDAY_Labour_Day', 'HOLIDAY_Easter_Monday',
    'HOLIDAY_Good_Friday',
    'EVENT_Multi Hazard, Snow / Ice', 'EVENT_High Wind / Gust',
    'EVENT_Multi Hazard', 'EVENT_Snow / Ice', 'EVENT_Flooding',
]

SPECIFIC_REGIONS  = ['IS_AUCKLAND', 'IS_WELLINGTON', 'IS_CANTERBURY']
TIME_FEATURES     = ['HOUR_SIN', 'HOUR_COS', 'DAY_SIN', 'DAY_COS', 'MONTH_SIN', 'MONTH_COS', 'YEAR']
TRAFFIC_FEATURES  = ['IS_DIRECTION_2']
LAG_FEATURES      = ['FLOW_lag_24h']
TARGET = ['FLOW']

ALL_FEATURES = TRAFFIC_FEATURES + NRC_FEATURES + SPECIFIC_REGIONS + TIME_FEATURES + WEATHER_FEATURES + LAG_FEATURES

# Ensure all expected columns exist (in case no events happened in the sample)
for feature in ALL_FEATURES:
    if feature not in df_hourly.columns:
        df_hourly = df_hourly.with_columns(pl.lit(0).alias(feature))

print("Dropping nulls and splitting data...")
df_hourly = df_hourly.drop_nulls(subset=ALL_FEATURES)

test = df_hourly.filter(pl.col('YEAR') == 2021)
test = test.to_pandas()
test = test.sort_values('DATETIME_HOUR')

X_test = test[ALL_FEATURES]

print("Loading model...")
baseline_xgboost = joblib.load('tuned_xgb(final).joblib')

print("Predicting...")
y_pred = baseline_xgboost.predict(X_test)

test['Predicted_Flow'] = y_pred

print("Saving predictions...")
test.to_csv('predictions.csv', index=False)
print("Done!")
