import pandas as pd
from sklearn.model_selection import train_test_split
from model_training.src.train_lr import train_logistic_regression
from model_training.src.train_lgbm import train_lightgbm
from feature_pipeline.src.monitoring import check_model_stability


def run_model_evaluation(feature_path, raw_data_path, target_col):
    print(f"[Data] Loading features from {feature_path}...")
    features = pd.read_csv(feature_path)

    print(f"[Data] Loading targets from {raw_data_path}...")
    raw = pd.read_csv(raw_data_path)

    raw.columns = [c.lower() for c in raw.columns]

    targets = raw[["user_id", target_col]].drop_duplicates()

    df = features.merge(
        targets,
        on="user_id",
        how="inner"
    )

    print(f"[Data] Final dataset shape: {df.shape}")

    print("[Model] Training Logistic Regression...")
    lr_result = train_logistic_regression(df, target_col)

    print("[Model] Training LightGBM...")
    lgbm_result = train_lightgbm(df, target_col)

    print("[Audit] Checking Model Stability (PSI)...")
    X = df.drop(columns=[target_col, "user_id"], errors="ignore")
    y = df[target_col]

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    psi_val, status = check_model_stability(lgbm_result["model"], X_train, X_test)

    summary = pd.DataFrame({
        "Model": ["Logistic Regression", "LightGBM"],
        "AUC": [lr_result["auc"], lgbm_result["auc"]]
    })

    return {
        "summary": summary,
        "lr_coefficients": lr_result["coefficients"],
        "lgbm_importance": lgbm_result["importance"],
        "psi_score": psi_val,
        "psi_status": status
    }


if __name__ == "__main__":
    FEATURE_PATH = "data/processed/transactions_engineered.csv"
    RAW_PATH = "data/processed/transactions_processed.csv"
    TARGET_COL = "default"

    try:
        results = run_model_evaluation(FEATURE_PATH, RAW_PATH, TARGET_COL)

        print("\n=== Model Performance Summary ===")
        print(results["summary"])

        print("\n=== Top Logistic Regression Coefficients ===")
        print(results["lr_coefficients"].head(10))

        print("\n=== Top LightGBM Feature Importance ===")
        print(results["lgbm_importance"].head(10))

        print("\n=== Model Stability (PSI) ===")
        print(f"PSI Score: {results['psi_score']:.4f}")
        print(f"Status   : {results['psi_status']}")

    except KeyError as e:
        print(f"\nCRITICAL ERROR: Kolom tidak ditemukan. {e}")
        print("Saran: Cek nama kolom di file CSV raw lo. Apakah 'default' atau 'default_payment_next_month'?")