import joblib
import pandas as pd
from pathlib import Path
MODEL_PATH = Path("models/best_model.pkl")

def predict(input_csv: str, output_csv: str = "outputs/predictions.csv"):
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(input_csv)
    # Do same engineering as training
    from src.features import simple_feature_engineering
    df = simple_feature_engineering(df)
    # select same features
    numeric_features = ['Average Cost for two','Votes','Longitude','Latitude','Price range']
    categorical_features = ['City','PrimaryCuisine','Currency']
    features = [c for c in numeric_features+categorical_features if c in df.columns]
    preds = model.predict(df[features])
    df['predicted_rating'] = preds
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print("Saved predictions to", output_csv)

if __name__ == "__main__":
    import sys
    input_csv = sys.argv[1]
    predict(input_csv)
 
