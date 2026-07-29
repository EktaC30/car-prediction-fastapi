import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "3_Models" / "car_price_model.pkl"
ENCODER_PATH = BASE_DIR / "3_Models" / "encoder.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "3_Models" / "feature_columns.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

def get_model_info():
    return {
        "model_type": type(model).__name__,
        "n_features": model.n_features_in_
    }

def predict_price(input_data: dict):

    input_df = pd.DataFrame([input_data])

    encoded_data = encoder.transform(input_df)

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=feature_columns
    )

    prediction = model.predict(encoded_df)

    return round(float(prediction[0]), 2)