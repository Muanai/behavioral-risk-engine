import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import joblib

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

TARGET_COL = "default"
RAW_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed",
                 "transactions_processed.csv"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))


def fetch_features():
    db_user = os.getenv("DB_USER", "user")
    db_pass = os.getenv("DB_PASSWORD", "password")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5433")
    db_name = os.getenv("DB_NAME", "feature_store")

    db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    engine = create_engine(db_url)
    return pd.read_sql("SELECT * FROM behavioral_features", engine)


def fetch_labels():
    df = pd.read_csv(RAW_DATA_PATH)
    df.columns = [c.lower() for c in df.columns]

    available_cols = df.columns.tolist()
    if TARGET_COL not in available_cols:
        fallback_targets = [c for c in available_cols if 'default' in c or 'target' in c or 'y' == c]
        if fallback_targets:
            df.rename(columns={fallback_targets[0]: TARGET_COL}, inplace=True)

    return df[['user_id', TARGET_COL]].drop_duplicates(subset=['user_id'])


def main():
    df_feat = fetch_features()
    df_label = fetch_labels()

    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_df.columns = [c.lower() for c in raw_df.columns]

    df_magnitude = raw_df.groupby('user_id').agg(
        mean_bill_amt=('bill_amt', 'mean'),
        max_bill_amt=('bill_amt', 'max'),
        mean_pay_amt=('pay_amt', 'mean'),
        sum_pay_amt=('pay_amt', 'sum')
    ).reset_index()

    df_train = pd.merge(df_feat, df_label, on="user_id", how="inner")
    df_train = pd.merge(df_train, df_magnitude, on="user_id", how="inner")

    features = [c for c in df_train.columns if c not in ["user_id", "timestamp", "default"]]
    X = df_train[features]
    y = df_train["default"]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    models = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            importance_type='gain'
        )

        clf.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )

        oof_preds[valid_idx] = clf.predict_proba(X_valid)[:, 1]
        models.append(clf)

    auc_score = roc_auc_score(y, oof_preds)
    print(f"Hybrid Scenario OOF ROC-AUC Score: {auc_score:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(models[0], os.path.join(MODEL_DIR, "lgbm_model_fold0.pkl"))
    print(f"Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()