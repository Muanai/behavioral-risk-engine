import pandas as pd

def load_data(path):
    df = pd.read_excel(path)
    df.columns = df.columns.str.lower()
    return df
