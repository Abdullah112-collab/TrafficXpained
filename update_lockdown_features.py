import polars as pl
from datetime import datetime

print("Loading cleaned_master.csv...")
df = pl.read_csv("cleaned_master.csv", schema_overrides={"SITEREF": pl.String, "SH": pl.String})

print("Extracting dates and regions...")
# Extract Date string (YYYY-MM-DD)
df = df.with_columns(pl.col("DATETIME").str.slice(0, 10).alias("DATE_STR"))
# Extract Region Name
df = df.with_columns(pl.col('REGION').str.replace(r'^\d+ - ', '').alias('REGION_NAME'))

# Define date ranges for Auckland
AUCKLAND_L4 = [
    ("2020-03-26", "2020-04-27"),
    ("2021-08-18", "2021-09-21")
]
AUCKLAND_L3 = [
    ("2020-03-24", "2020-03-25"),
    ("2020-04-28", "2020-05-13"),
    ("2020-08-12", "2020-08-30"),
    ("2021-02-15", "2021-02-17"),
    ("2021-03-01", "2021-03-07"),
    ("2021-09-22", "2021-12-02")
]

# Define date ranges for Rest of NZ
NATIONAL_L4 = [
    ("2020-03-26", "2020-04-27"),
    ("2021-08-18", "2021-08-31")
]
NATIONAL_L3 = [
    ("2020-03-24", "2020-03-25"),
    ("2020-04-28", "2020-05-13"),
    ("2021-09-01", "2021-09-07")
]
NATIONAL_L2 = [
    ("2020-03-22", "2020-03-23"),
    ("2020-05-14", "2020-06-08"),
    ("2020-08-12", "2020-09-21"),
    ("2021-02-15", "2021-02-17"),
    ("2021-03-01", "2021-03-07"),
    ("2021-09-08", "2021-12-02")
]
WELLINGTON_L2 = [
    ("2021-06-23", "2021-06-29")
]

# Announcement days (evacuation effect)
ANNOUNCEMENT_DAYS = [
    "2020-03-21", "2020-03-23", "2020-03-25", "2020-08-11",
    "2021-02-14", "2021-02-28", "2021-08-17"
]

def is_in_ranges(date_col, ranges):
    expr = pl.lit(False)
    for start, end in ranges:
        expr = expr | ((date_col >= start) & (date_col <= end))
    return expr

print("Applying one-hot encoded lockdown features...")

date_col = pl.col("DATE_STR")
region_col = pl.col("REGION_NAME")

df = df.with_columns([
    # Announcements
    pl.when(date_col.is_in(ANNOUNCEMENT_DAYS))
      .then(1).otherwise(0).cast(pl.Int8).alias("EVENT_Lockdown_Announced"),
      
    # Auckland L4 & L3
    pl.when((region_col == "Auckland") & is_in_ranges(date_col, AUCKLAND_L4))
      .then(1).otherwise(0).cast(pl.Int8).alias("LOCKDOWN_L4_AUCKLAND"),
    
    pl.when((region_col == "Auckland") & is_in_ranges(date_col, AUCKLAND_L3))
      .then(1).otherwise(0).cast(pl.Int8).alias("LOCKDOWN_L3_AUCKLAND"),
      
    # National L4, L3, L2 (Applies to all regions except Auckland during these dates, but for simplicity we can restrict to non-Auckland)
    pl.when((region_col != "Auckland") & is_in_ranges(date_col, NATIONAL_L4))
      .then(1).otherwise(0).cast(pl.Int8).alias("LOCKDOWN_L4_NATIONAL"),
      
    pl.when((region_col != "Auckland") & is_in_ranges(date_col, NATIONAL_L3))
      .then(1).otherwise(0).cast(pl.Int8).alias("LOCKDOWN_L3_NATIONAL"),
      
    pl.when(
        ((region_col != "Auckland") & is_in_ranges(date_col, NATIONAL_L2)) |
        ((region_col == "Wellington") & is_in_ranges(date_col, WELLINGTON_L2))
    ).then(1).otherwise(0).cast(pl.Int8).alias("LOCKDOWN_L2_NATIONAL"),
])

# Drop temporary columns and the old IS_PANDEMIC
columns_to_drop = ["DATE_STR", "REGION_NAME"]
if "IS_PANDEMIC" in df.columns:
    columns_to_drop.append("IS_PANDEMIC")

df = df.drop(columns_to_drop)

print("Writing to cleaned_master_v2.csv...")
df.write_csv("cleaned_master_v2.csv")
print("Done! cleaned_master_v2.csv generated successfully.")
