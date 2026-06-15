import os
import pandas as pd
from sqlalchemy import create_engine

def push_to_offline_store(df: pd.DataFrame, table_name: str = "behavioral_features"):
    db_user = os.getenv("DB_USER", "user")
    db_pass = os.getenv("DB_PASSWORD", "password")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5433")
    db_name = os.getenv("DB_NAME", "feature_store")
    
    db_url = os.getenv(
        "OFFLINE_STORE_URL",
        f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)