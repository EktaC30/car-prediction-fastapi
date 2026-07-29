import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "car_price_model.pkl"
ENCODER_PATH = BASE_DIR / "encoder.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

print("=" * 60)
print("LOADING ARTIFACTS")
print("=" * 60)

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

print("\nModel Type:")
print(type(model))

print("\nEncoder Type:")
print(type(encoder))

print("\nModel expects features:")
print(model.n_features_in_)

print("\nFeatures saved:")
print(len(feature_columns))

print("\nFirst 20 feature names:")
for col in feature_columns[:20]:
    print(col)

print("\nEncoder output features:")
try:
    generated_features = encoder.get_feature_names_out()

    print(f"Count: {len(generated_features)}")

    if len(generated_features) == len(feature_columns):
        print("\n✅ feature_columns matches encoder output")
    else:
        print("\n❌ feature_columns does NOT match encoder output")

except Exception as e:
    print("Could not retrieve encoder feature names")
    print(e)

print("\nModel Feature Check")

if model.n_features_in_ == len(feature_columns):
    print(
        "✅ Model expects exactly the same number of "
        "features as feature_columns."
    )
    print(
        "✅ Current predict.py is likely correct."
    )
else:
    print(
        "❌ Model expects more/fewer features than "
        "feature_columns."
    )
    print(
        "❌ We probably need numeric columns merged "
        "with encoded columns before prediction."
    )

print("=" * 60)