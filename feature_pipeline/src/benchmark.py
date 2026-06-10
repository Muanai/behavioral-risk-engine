import time
import pandas as pd
from feature_pipeline.src.features_numba import rolling_mean_numba
from feature_pipeline.src.features_pandas import rolling_features
from feature_pipeline.src.rolling_baseline import rolling_mean_python

df = pd.read_csv("data/processed/transactions_processed.csv")

values = df["bill_amt"].values
user_ids = df["user_id"].values

start = time.perf_counter()
baseline = rolling_mean_python(df, user_ids, 30)
print("Python loops:", time.perf_counter() - start)

start = time.perf_counter()
pandas = rolling_features(df, 30)
print("Pandas:", time.perf_counter() - start)

rolling_mean_numba(values, user_ids, 30)

start = time.perf_counter()
rolling_mean_numba(values, user_ids, 30)
print("Numba with warmup:", time.perf_counter() - start)
