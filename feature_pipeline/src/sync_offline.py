import pandas as pd
from sqlalchemy import create_engine

def push_to_offline_store(df: pd.DataFrame, table_name: str = "behavioral_features"):
    engine = create_engine("postgresql://user:password@localhost:5432/feature_store")
    df.to_sql(table_name, engine, if_exists="replace", index=False)