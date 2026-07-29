import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# ==========================================================
# PATHS
# ==========================================================

DATA_PATH = "../1_Data/car_cleaned.csv"

MODEL_PATH = "car_price_model.pkl"
ENCODER_PATH = "encoder.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"


# ==========================================================
# LOAD DATA
# ==========================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    print("\nDataset Loaded Successfully")
    print(f"Shape : {df.shape}")

    return df


# ==========================================================
# PREPARE DATA
# ==========================================================

def prepare_data(df):

    target_column = "price"

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # categorical columns
    categorical_columns = X.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    # numerical columns
    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    print("\nCategorical Columns:")
    print(categorical_columns)

    print("\nNumerical Columns:")
    print(numerical_columns)

    return X, y, categorical_columns, numerical_columns


# ==========================================================
# ENCODING
# ==========================================================

def encode_features(
    X_train,
    X_test,
    categorical_columns
):

    encoder = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ],
        remainder="passthrough"
    )

    X_train_encoded = encoder.fit_transform(X_train)

    X_test_encoded = encoder.transform(X_test)

    return (
        X_train_encoded,
        X_test_encoded,
        encoder
    )


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


# ==========================================================
# EVALUATE MODEL
# ==========================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    print("\n========== MODEL METRICS ==========")
    print(f"R² Score : {r2:.4f}")
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print("===================================")

    return predictions


# ==========================================================
# SAVE ARTIFACTS
# ==========================================================

def save_artifacts(model, encoder):

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        encoder,
        ENCODER_PATH
    )

    feature_columns = list(
        encoder.get_feature_names_out()
    )

    joblib.dump(
        feature_columns,
        FEATURE_COLUMNS_PATH
    )
    
    print("\nArtifacts Saved Successfully")

    print(f"Model    : {MODEL_PATH}")
    print(f"Encoder  : {ENCODER_PATH}")
    print(f"Features : {FEATURE_COLUMNS_PATH}")


# ==========================================================
# MAIN
# ==========================================================

def main():

    try:

        # load data
        df = load_data()

        # prepare
        (
            X,
            y,
            categorical_columns,
            numerical_columns
        ) = prepare_data(df)

        # split
        (
            X_train,
            X_test,
            y_train,
            y_test
        ) = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # encode
        (
            X_train_encoded,
            X_test_encoded,
            encoder
        ) = encode_features(
            X_train,
            X_test,
            categorical_columns
        )

        # train
        model = train_model(
            X_train_encoded,
            y_train
        )

        # evaluate
        evaluate_model(
            model,
            X_test_encoded,
            y_test
        )



        # ==========================================================
        # VERIFY TRAINING OUTPUT
        # ==========================================================

        print("\nTRAINING SUMMARY")
        print("=" * 50)

        print(f"Original X Shape       : {X.shape}")

        print(
            f"Encoded Train Shape    : "
            f"{X_train_encoded.shape}"
        )

        print(
            f"Encoded Test Shape     : "
            f"{X_test_encoded.shape}"
        )

        print(
            f"Model Features Expected: "
            f"{model.n_features_in_}"
        )

        print(
            f"Saved Feature Count    : "
            f"{len(encoder.get_feature_names_out())}"
        )

        print("=" * 50)

        # save files
        save_artifacts(
            model,
            encoder
)

        print(
            "\nTraining Completed Successfully ✅"
        )

    except Exception as e:

        print(f"\nError : {e}")


if __name__ == "__main__":
    main()