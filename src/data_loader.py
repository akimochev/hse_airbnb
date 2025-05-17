import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Загружает CSV-файл в pandas.DataFrame.
    """
    df = pd.read_csv(path)
    return df
