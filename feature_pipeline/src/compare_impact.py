import pandas as pd
from model_training.src.train_lgbm import train_lightgbm
import shap
import numpy as np
import matplotlib.pyplot as plt

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

    print("\n[Visual] Generating SHAP Feature Importance Plot...")

    best_model = res_hybrid['model']
    X_test_hybrid = res_hybrid['X_test']

    explainer = shap.TreeExplainer(best_model)
    X_display = X_test_hybrid.iloc[:5000]
    shap_values = explainer.shap_values(X_display)

    if isinstance(shap_values, list):
        shap_values_target = shap_values[1]
    else:
        shap_values_target = shap_values

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    bg_color = '#1A202C'
    teal_accent = '#4FD1C5'

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    from shap.plots import colors
    colors.blue_rgb = np.array([79 / 255, 209 / 255, 197 / 255])

    shap.summary_plot(shap_values_target, X_display, plot_type="bar",
                      max_display=12, show=False, color=teal_accent)

    ax.set_title('Hybrid Model: Feature Importance', fontsize=16, fontweight='bold', color='#F7FAFC', pad=20)
    ax.tick_params(axis='x', colors='#A0AEC0')
    ax.tick_params(axis='y', colors='#F7FAFC', labelsize=12)

    plt.tight_layout()
    output_filename = 'proof_shap_hybrid.svg'
    plt.savefig(output_filename, dpi=300, facecolor=bg_color)
    print(f"   >>> PROOF SAVED: {output_filename}")

    plt.savefig(output_filename, format='svg', transparent=True)
    # plt.show()

if __name__ == "__main__":
    main()
