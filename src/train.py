import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

from src.data_utils import load_raw, save_processed
from src.features import simple_feature_engineering

MODEL_PATH = Path("models/best_model.pkl")

def main():
    # Load and preprocess data
    df = load_raw()
    df = simple_feature_engineering(df)
    save_processed(df)

    # Define target and drop unhelpful columns
    target = "Aggregate rating"
    drop_cols = [
        'Restaurant ID','Restaurant Name','Address',
        'Locality Verbose','Rating color','Rating text'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Feature selection
    numeric_features = ['Average Cost for two','Votes','Longitude','Latitude','Price range']
    numeric_features = [c for c in numeric_features if c in df.columns]

    categorical_features = ['City','PrimaryCuisine','Currency']
    categorical_features = [c for c in categorical_features if c in df.columns]

    X = df[numeric_features + categorical_features].copy()
    y = df[target].copy()

    # Pipelines for preprocessing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Handle sklearn version differences (sparse vs sparse_output)
    try:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except TypeError:
        # For older sklearn
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Full pipeline with Random Forest
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation (handle old sklearn that doesn’t support squared=False)
    from sklearn.metrics import mean_squared_error, r2_score

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2:", r2)

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Saved model to", MODEL_PATH)

if __name__ == "__main__":
    main()
