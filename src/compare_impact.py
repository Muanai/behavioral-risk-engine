import pandas as pd
from src.models.train_lgbm import train_lightgbm

RAW_PATH = "data/processed/transactions_processed.csv"
ENGINEERED_PATH = "data/processed/transactions_engineered.csv"
TARGET_COL = "default"


def get_baseline_features(df_raw):
    print("   -> Generating Baseline features (Mean Aggregates)...")
    df_raw.columns = [c.lower() for c in df_raw.columns]
    baseline = df_raw.groupby('user_id')[['bill_amt', 'pay_amt']].mean().reset_index()
    baseline.columns = ['user_id', 'mean_bill', 'mean_pay']
    return baseline


def main():
    print("[1/4] Loading Data...")
    try:
        raw = pd.read_csv(RAW_PATH)
        df_eng_feats = pd.read_csv(ENGINEERED_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    raw.columns = [c.lower() for c in raw.columns]

    target_actual = TARGET_COL
    if TARGET_COL not in raw.columns:
        candidates = ['default_payment_next_month', 'default_flag', 'target']
        for c in candidates:
            if c in raw.columns:
                target_actual = c
                break

    targets = raw[['user_id', target_actual]].drop_duplicates()

    print("\n[2/4] Scenario A: BASELINE (Magnitude Only)")
    df_base_feats = get_baseline_features(raw)
    df_base = df_base_feats.merge(targets, on='user_id', how='inner')
    res_base = train_lightgbm(df_base, target_actual)
    print(f"   >>> BASELINE AUC: {res_base['auc']:.4f}")

    print("\n[3/4] Scenario B: NUMBA BEHAVIOR (Trend Only)")
    df_eng = df_eng_feats.merge(targets, on='user_id', how='inner')
    res_eng = train_lightgbm(df_eng, target_actual)
    print(f"   >>> BEHAVIOR AUC: {res_eng['auc']:.4f}")

    print("\n[4/4] Scenario C: HYBRID (Magnitude + Trend)")
    df_hybrid = df_base_feats.merge(df_eng_feats, on='user_id', how='inner')
    df_hybrid = df_hybrid.merge(targets, on='user_id', how='inner')

    res_hybrid = train_lightgbm(df_hybrid, target_actual)
    print(f"   >>> HYBRID AUC  : {res_hybrid['auc']:.4f}")

    print("\n" + "=" * 40)
    print("FINAL LEADERBOARD")
    print("=" * 40)
    print(f"1. Hybrid AUC   : {res_hybrid['auc']:.4f}")
    print(f"2. Baseline AUC : {res_base['auc']:.4f}")
    print(f"3. Behavior AUC : {res_eng['auc']:.4f}")

    uplift = res_hybrid['auc'] - res_base['auc']
    print(f"\nFinal Uplift (Hybrid vs Baseline): {uplift:+.4f} ({uplift / res_base['auc']:.1%} improvement)")


if __name__ == "__main__":
    main()
