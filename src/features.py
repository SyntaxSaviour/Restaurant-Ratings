import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X[self.columns]

def simple_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fill missing cuisines
    if 'Cuisines' in df.columns:
        df['Cuisines'] = df['Cuisines'].fillna("Unknown")
    # Example: extract the primary cuisine (first in list)
    if 'Cuisines' in df.columns:
        df['PrimaryCuisine'] = df['Cuisines'].apply(lambda x: str(x).split(",")[0].strip())
    # Convert booleans like 'Yes'/'No' to 1/0
    for col in ['Has Table booking','Has Online delivery','Is delivering now']:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1, 'No':0}).fillna(0).astype(int)
    return df
 
