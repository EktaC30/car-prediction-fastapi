import gradio as gr
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================================
# CONFIGURATION
# ==========================================================

API_URL = "http://127.0.0.1:8000/predict"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "1_Data" / "car_cleaned.csv"

df = pd.read_csv(DATA_PATH)

# ==========================================================
# CHARTS
# ==========================================================


def price_distribution():

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=df,
        x="price",
        kde=True,
        color="steelblue",
        ax=ax
    )

    ax.set_title("Car Price Distribution")
    ax.set_xlabel("Price")

    return fig


def horsepower_vs_price():

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=df,
        x="horsepower",
        y="price",
        hue="horsepower-binned",
        palette="viridis",
        ax=ax
    )

    ax.set_title("Horsepower vs Price")

    return fig


def average_price_by_make():

    avg_df = (
        df.groupby("make")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=avg_df,
        x="price",
        y="make",
        hue="make",
        legend=False,
        palette="crest",
        ax=ax
    )

    ax.set_title("Top 10 Makes by Average Price")

    return fig


# ==========================================================
# PREDICTION
# ==========================================================

def predict_car_price(
    make,
    fuel_type,
    aspiration,
    doors,
    body_style,
    drive_wheels,
    engine_size,
    horsepower,
    peak_rpm
):

    payload = {
        "make": make,
        "fuel_type": fuel_type,
        "aspiration": aspiration,
        "doors": doors,
        "body_style": body_style,
        "drive_wheels": drive_wheels,
        "engine_size": engine_size,
        "horsepower": horsepower,
        "peak_rpm": peak_rpm
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        response.raise_for_status()

        prediction = response.json()["predicted_price"]

        if horsepower <= 128:
            hp_category = "Low"
        elif horsepower <= 208:
            hp_category = "Medium"
        else:
            hp_category = "High"

        return f"""
# ✅ Prediction Result

## 🚗 Estimated Price

### ${prediction:,.2f}

### Vehicle Summary

- **Make:** {make}
- **Fuel Type:** {fuel_type}
- **Aspiration:** {aspiration}
- **Doors:** {doors}
- **Body Style:** {body_style}
- **Drive Wheels:** {drive_wheels}
- **Engine Size:** {engine_size}
- **Horsepower:** {horsepower}
- **Horsepower Category:** {hp_category}
- **Peak RPM:** {peak_rpm}
"""

    except Exception as e:

        return f"""
# ❌ Error

{str(e)}

### Ensure:

- FastAPI server is running
- URL is http://127.0.0.1:8000
- /predict endpoint is accessible
"""


# ==========================================================
# UI
# ==========================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
# 🚗 Car Price Prediction Dashboard

End-to-End Machine Learning Project

✅ Data Cleaning & EDA  
✅ Feature Engineering  
✅ Random Forest Regression  
✅ FastAPI Backend  
✅ Gradio Frontend  

---

### Model Performance

| Metric | Score |
|----------|----------|
| R² Score | 0.9405 |
| MAE | 1441.05 |
| RMSE | 2153.45 |

---
"""
    )

    # ======================================================
    # TAB 1 : PREDICTION
    # ======================================================

    with gr.Tab("🚗 Predict Price"):

        with gr.Row():

            with gr.Column():

                make = gr.Dropdown(
                    choices=[
                        "alfa-romero",
                        "audi",
                        "bmw",
                        "chevrolet",
                        "dodge",
                        "honda",
                        "isuzu",
                        "jaguar",
                        "mazda",
                        "mercedes-benz",
                        "mercury",
                        "mitsubishi",
                        "nissan",
                        "peugot",
                        "plymouth",
                        "porsche",
                        "renault",
                        "saab",
                        "subaru",
                        "toyota",
                        "volkswagen",
                        "volvo"
                    ],
                    value="toyota",
                    label="Make"
                )

                fuel_type = gr.Dropdown(
                    ["gas", "diesel"],
                    value="gas",
                    label="Fuel Type"
                )

                aspiration = gr.Dropdown(
                    ["std", "turbo"],
                    value="std",
                    label="Aspiration"
                )

                doors = gr.Dropdown(
                    ["two", "four"],
                    value="four",
                    label="Doors"
                )

                body_style = gr.Dropdown(
                    [
                        "convertible",
                        "hardtop",
                        "hatchback",
                        "sedan",
                        "wagon"
                    ],
                    value="sedan",
                    label="Body Style"
                )

                drive_wheels = gr.Dropdown(
                    [
                        "fwd",
                        "rwd",
                        "4wd"
                    ],
                    value="fwd",
                    label="Drive Wheels"
                )

                engine_size = gr.Slider(
                    minimum=50,
                    maximum=350,
                    value=130,
                    step=1,
                    label="Engine Size"
                )

                horsepower = gr.Slider(
                    minimum=40,
                    maximum=300,
                    value=111,
                    step=1,
                    label="Horsepower"
                )

                peak_rpm = gr.Slider(
                    minimum=4000,
                    maximum=7000,
                    value=5000,
                    step=100,
                    label="Peak RPM"
                )

                predict_button = gr.Button(
                    "🚀 Predict Price",
                    variant="primary"
                )

            with gr.Column():

                output = gr.Markdown()

        predict_button.click(
            fn=predict_car_price,
            inputs=[
                make,
                fuel_type,
                aspiration,
                doors,
                body_style,
                drive_wheels,
                engine_size,
                horsepower,
                peak_rpm
            ],
            outputs=output
        )

    # ======================================================
    # TAB 2 : EDA
    # ======================================================

    with gr.Tab("📊 EDA Insights"):

        gr.Markdown(
            """
# Exploratory Data Analysis

Visual insights from the cleaned automobile dataset.
"""
        )

        gr.Plot(value=price_distribution)

        gr.Plot(value=horsepower_vs_price)

        gr.Plot(value=average_price_by_make)

    # ======================================================
# TAB 3 : MODEL INFO
# ======================================================

with gr.Tab("🤖 Model Information"):

    gr.Markdown(
    """
    # Model Details
     
    ### Algorithm

    Random Forest Regressor

    ### Dataset

    Automobile Imports Dataset

    ### Records

    205 Rows


    ### Features
    77 Engineered Features
    ### Performance
    - **R² Score:** 0.9405
    - **MAE:** 1441.05
    - **RMSE:** 2153.45### Architecture
    30
     
    31
    ```text
    32
    User
    33
    ↓
    34
    Gradio Frontend
    35
    ↓
    36
    FastAPI Backend
    37
    ↓
    38
    Encoder
    39
    ↓
    40
    Random Forest Model
    41
    ↓
    42
    Predicted Price
    Show more lines

    """ )

    demo.launch()