import pandas as pd

meta_file = r"c:\Users\Afsar003\PycharmProjects\KiwiData\step_b1_meta.pkl"
flow_file = r"c:\Users\Afsar003\PycharmProjects\KiwiData\step_b1_flow.pkl"

print("--- META INFO ---")
try:
    meta = pd.read_pickle(meta_file)
    print("Type:", type(meta))
    if isinstance(meta, pd.DataFrame):
        print(meta.info())
        print(meta.head())
    elif isinstance(meta, dict):
        print("Keys:", meta.keys())
        for k, v in list(meta.items())[:3]:
            print(f"{k}: {type(v)}")
            if isinstance(v, pd.DataFrame):
                print(v.head(2))
    else:
        print(meta)
except Exception as e:
    print("Error loading meta:", e)

print("\n--- FLOW INFO ---")
try:
    flow = pd.read_pickle(flow_file)
    print("Type:", type(flow))
    if isinstance(flow, pd.DataFrame):
        print(flow.info())
        print(flow.head())
except Exception as e:
    print("Error loading flow:", e)
