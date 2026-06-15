import os
import pandas as pd
from sqlalchemy import create_engine

def push_to_offline_store(df: pd.DataFrame, table_name: str = "behavioral_features"):
    db_url = os.getenv(
        "OFFLINE_STORE_URL",
        "postgresql://user:password@localhost:5432/feature_store"
    )
    engine = create_engine(db_url)
    df.to_sql(table_name, engine, if_exists="replace", index=False)