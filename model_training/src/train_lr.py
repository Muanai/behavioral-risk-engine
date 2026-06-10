import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_logistic_regression(df, target_col):
    X = df.drop(columns=[target_col, 'user_id'], errors='ignore')
    y = df[target_col]

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)

    coef_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_[0]
    }).assign(abs_coef=lambda x: x.coefficient.abs()) \
      .sort_values("abs_coef", ascending=False)

    return {
        "model": model,
        "auc": auc,
        "coefficients": coef_df
    }
