import pandas as pd
from pathlib import Path

DATA_RAW = Path("data/raw/restaurant_ratings.csv")
DATA_PROCESSED = Path("data/processed/restaurant_ratings_clean.csv")

def load_raw(path: str = None) -> pd.DataFrame:
    p = path or DATA_RAW
    df = pd.read_csv(p)
    return df

def save_processed(df: pd.DataFrame, path: str = None):
    p = path or DATA_PROCESSED
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print(f"Saved processed data to {p}")
