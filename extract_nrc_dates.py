import polars as pl

print("Loading data...")
df = pl.read_csv("cleaned_master.csv", schema_overrides={"SITEREF": pl.String, "SH": pl.String})

print("Filtering for NRCs...")
# Extract unique dates and the event name
df = df.with_columns(pl.col("DATETIME").str.slice(0, 10).alias("DATE"))

# Get unique holidays
holidays = df.filter((pl.col('IS_HOLIDAY') == 1) & (pl.col('HOLIDAY_NAME') != 'None')).select(['DATE', 'HOLIDAY_NAME']).unique().sort('DATE')

# Get unique extreme events
extremes = df.filter((pl.col('IS_EXTREME') == 1) & (pl.col('EXTREME_HAZARD') != 'None')).select(['DATE', 'EXTREME_HAZARD']).unique().sort('DATE')

with open("nrc_event_dates.txt", "w", encoding="utf-8") as f:
    f.write("=== PUBLIC HOLIDAYS ===\n")
    for row in holidays.iter_rows():
        f.write(f"{row[0]}: {row[1]}\n")
        
    f.write("\n=== EXTREME WEATHER EVENTS ===\n")
    for row in extremes.iter_rows():
        f.write(f"{row[0]}: {row[1]}\n")
        
print("Successfully generated nrc_event_dates.txt")
