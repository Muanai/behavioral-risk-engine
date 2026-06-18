import os
import pandas as pd
import redis


def push_to_online_store(df: pd.DataFrame, id_col: str = "user_id", time_col: str = "timestamp"):
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=0,
        decode_responses=True
    )
    latest_features = df.sort_values(time_col).groupby(id_col).tail(1)

    pipeline = r.pipeline()

    for _, row in latest_features.iterrows():
        cid = str(int(row[id_col]))
        feature_dict = row.drop(labels=[id_col, time_col]).fillna(0).to_dict()
        feature_dict_str = {str(k): str(v) for k, v in feature_dict.items()}
        pipeline.hset(f"customer:{cid}", mapping=feature_dict_str)

    pipeline.execute()