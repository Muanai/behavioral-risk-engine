import pandas as pd
from pathlib import Path


def wide_to_long_transactions(
    df: pd.DataFrame,
    bill_cols: list,
    pay_cols: list,
    user_col: str = "id",
    target_col: str = "default_payment_next_month"
) -> pd.DataFrame:

    records = []

    for _, row in df.iterrows():
        for t, bill_col in enumerate(bill_cols):
            records.append({
                "user_id": row[user_col],
                "month_idx": t,
                "bill_amt": row[bill_col],
                "pay_amt": row[pay_cols[t]],
                "default": row[target_col],
            })

    txn_df = pd.DataFrame(records)
    return txn_df.sort_values(["user_id", "month_idx"])


if __name__ == "__main__":

    RAW_PATH = Path("data/raw/default-of-credit-card-clients.xls")
    OUT_PATH = Path("data/processed/transactions_processed.csv")

    df = pd.read_excel(RAW_PATH, header=1)

    df.columns = (
        df.columns
          .str.lower()
          .str.strip()
          .str.replace(" ", "_")
    )

    bill_cols = [c for c in df.columns if "bill_amt" in c]
    pay_cols  = [c for c in df.columns if "pay_amt" in c]

    txn_df = wide_to_long_transactions(
        df=df,
        bill_cols=bill_cols,
        pay_cols=pay_cols,
    )

    # dtype contract
    txn_df["user_id"]   = txn_df["user_id"].astype("int64")
    txn_df["month_idx"] = txn_df["month_idx"].astype("int8")
    txn_df["bill_amt"]  = txn_df["bill_amt"].astype("float64")
    txn_df["pay_amt"]   = txn_df["pay_amt"].astype("float64")
    txn_df["default"]   = txn_df["default"].astype("int8")

    # safety checks
    assert txn_df["month_idx"].max() <= 11
    assert set(txn_df["default"].unique()) <= {0, 1}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    txn_df.to_csv(OUT_PATH, index=False)

    print(f"Saved transactional data to {OUT_PATH}")
