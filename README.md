# 🚗 Car Price Prediction System

## Overview

This project is an end-to-end Machine Learning solution for predicting automobile prices using historical vehicle attributes.

The solution includes:

- Data Cleaning & EDA
- Feature Engineering
- Model Training using Random Forest Regressor
- FastAPI Backend
- Gradio Frontend Dashboard
- Interactive Data Visualizations

---

## Project Structure

```text
Car_Price_Prediction/
│
├── 1_Data/
│   ├── imports-85.data
│   └── car_cleaned.csv
│
├── 2_EDA/
│   └── EDA.py
│
├── 3_Models/
│   ├── train.py
│   ├── car_price_model.pkl
│   ├── encoder.pkl
│   └── feature_columns.pkl
│
├── 4_FastAPI/
│   ├── app.py
│   ├── predict.py
│   └── defaults.py
│
├── 5_Gradio/
│   └── gradio_app.py
│
├── requirements.txt
├── README.md
└── .gitignore