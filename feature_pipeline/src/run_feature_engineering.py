import pandas as pd
import numpy as np
import time
import os
import sys

# Add workspace root to sys.path so 'feature_pipeline' and 'model_training' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from feature_pipeline.src.features_numba import get_max_consecutive_late, get_velocity_slope
from feature_pipeline.src.sync_offline import push_to_offline_store
from feature_pipeline.src.sync_online import push_to_online_store


DATA_PATH = os.path.join("data", "processed", "transactions_processed.csv")
OUTPUT_PATH = os.path.join("data", "processed", "transactions_engineered.csv")


def load_data(path):
    print(f"[1/4] Loading data from {path}...")
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    return df


def prepare_numba_inputs_from_long_format(df):
    print("[2/4] Pivoting data to Matrix format...")

    df_pivot = df.pivot_table(
        index='user_id',
        columns='month_idx',
        values=['bill_amt', 'pay_amt']
    )

    df_pivot = df_pivot.fillna(0)

    bill_matrix = df_pivot['bill_amt'].values
    pay_matrix = df_pivot['pay_amt'].values
    user_ids = df_pivot.index.values

    is_late_matrix = np.zeros(bill_matrix.shape, dtype=np.int32)
    mask_late = (pay_matrix < (bill_matrix - 100)) & (bill_matrix > 0)
    is_late_matrix[mask_late] = 1

    epsilon = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_matrix = pay_matrix / (bill_matrix + epsilon)
    ratio_matrix_last_3 = ratio_matrix[:, -3:]

    return user_ids, is_late_matrix, ratio_matrix_last_3


def main():
    start_global = time.time()

    try:
        df = load_data(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_PATH}.")
        return

    required = ['user_id', 'month_idx', 'bill_amt', 'pay_amt']
    if not all(col in df.columns for col in required):
        print(f"Error: Dataset harus punya kolom: {required}")
        return

    max_month_per_user = df.groupby('user_id')['month_idx'].max().reset_index()
    max_month_per_user.columns = ['user_id', 'timestamp']

    user_ids, late_mat, ratio_mat = prepare_numba_inputs_from_long_format(df)

    print(f"      Matrices prepared. Users: {len(user_ids)}")

    print("[3/4] Executing Numba Engines...")

    df_res = pd.DataFrame({'user_id': user_ids})

    print("      -> Computing Late Streaks...")
    df_res['feat_max_late_streak'] = get_max_consecutive_late(late_mat)

    print("      -> Computing Payment Velocity...")
    df_res['feat_pay_velocity'] = get_velocity_slope(ratio_mat)

    print("      -> Skipping Critical Utilization (LIMIT_BAL missing)")

    print(f"[4/4] Saving engineered features to {OUTPUT_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_res.to_csv(OUTPUT_PATH, index=False)

    df_sync = pd.merge(df_res, max_month_per_user, on='user_id', how='left')

    print("[5/5] Synchronizing to Feature Store Layers...")
    push_to_offline_store(df_sync)
    push_to_online_store(df_sync, id_col="user_id", time_col="timestamp")

    print(f"\nMission Complete. Total runtime: {time.time() - start_global:.2f} seconds.")


if __name__ == "__main__":
    main()