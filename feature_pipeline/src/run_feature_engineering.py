import pandas as pd
import numpy as np
import time
import os
import sys

# Add workspace root to sys.path so 'feature_pipeline' and 'model_training' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from feature_pipeline.src.features_numba import get_max_consecutive_late, get_velocity_slope, get_critical_utilization
from feature_pipeline.src.sync_offline import push_to_offline_store
from feature_pipeline.src.sync_online import push_to_online_store


OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "transactions_engineered.csv"))


def load_and_prepare_matrices():
    print("[1/4 & 2/4] Loading and Pivoting Data...")
    RAW_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "default-of-credit-card-clients.xls"))
    PROCESSED_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "transactions_processed.csv"))

    raw_df = pd.read_excel(RAW_FILE, header=1)
    raw_df.columns = [c.lower() for c in raw_df.columns]
    raw_df.rename(columns={'id': 'user_id'}, inplace=True)
    limit_map = raw_df.set_index('user_id')['limit_bal'].to_dict()

    df_trans = pd.read_csv(PROCESSED_FILE)

    max_month_per_user = df_trans.groupby('user_id')['month_idx'].max().reset_index()
    max_month_per_user.columns = ['user_id', 'timestamp']

    df_pivot_bill = df_trans.pivot(index='user_id', columns='month_idx', values='bill_amt').fillna(0)
    df_pivot_pay = df_trans.pivot(index='user_id', columns='month_idx', values='pay_amt').fillna(0)

    users = df_pivot_bill.index.values
    limit_bal_array = np.array([limit_map.get(u, 10000.0) for u in users], dtype=np.float64)

    bill_matrix = df_pivot_bill.values
    pay_matrix = df_pivot_pay.values

    is_late_matrix = np.zeros(bill_matrix.shape, dtype=np.int32)
    mask_late = (pay_matrix < (bill_matrix - 100)) & (bill_matrix > 0)
    is_late_matrix[mask_late] = 1

    epsilon = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_matrix = pay_matrix / (bill_matrix + epsilon)
    ratio_matrix_last_3 = ratio_matrix[:, -3:]

    return users, is_late_matrix, ratio_matrix_last_3, bill_matrix, limit_bal_array, max_month_per_user


def main():
    start_global = time.time()

    try:
        user_ids, late_mat, ratio_mat, bill_matrix, limit_bal_array, max_month_per_user = load_and_prepare_matrices()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"      Matrices prepared. Users: {len(user_ids)}")

    print("[3/4] Executing Numba Engines...")

    df_res = pd.DataFrame({'user_id': user_ids})

    print("      -> Computing Late Streaks...")
    df_res['feat_max_late_streak'] = get_max_consecutive_late(late_mat)

    print("      -> Computing Payment Velocity...")
    df_res['feat_pay_velocity'] = get_velocity_slope(ratio_mat)

    print("      -> Computing Critical Utilization...")
    critical_util_features = get_critical_utilization(bill_matrix, limit_bal_array)
    df_res['critical_utilization_count'] = critical_util_features

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