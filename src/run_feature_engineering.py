import pandas as pd
import numpy as np
import time
import os
from src.features_numba import get_max_consecutive_late, get_velocity_slope


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
        print(f"Error: File not found at {DATA_PATH}. Cek path file CSV lo.")
        return

    required = ['user_id', 'month_idx', 'bill_amt', 'pay_amt']
    if not all(col in df.columns for col in required):
        print(f"Error: Dataset harus punya kolom: {required}")
        print(f"Kolom yang ditemukan: {df.columns.tolist()}")
        return

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

    print(f"\nMission Complete. Total runtime: {time.time() - start_global:.2f} seconds.")


if __name__ == "__main__":
    main()
