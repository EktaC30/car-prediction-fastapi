import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "1_Data" / "car_cleaned.csv"

df = pd.read_csv(DATA_PATH)

DEFAULTS = {
    "symboling": int(df["symboling"].median()),
    "normalized-losses": float(df["normalized-losses"].median()),
    "wheel-base": float(df["wheel-base"].median()),
    "length": float(df["length"].median()),
    "width": float(df["width"].median()),
    "height": float(df["height"].median()),
    "curb-weight": float(df["curb-weight"].median()),
    "engine-type": "ohc",
    "num-of-cylinders": "four",
    "fuel-system": "mpfi",
    "bore": float(df["bore"].median()),
    "stroke": float(df["stroke"].median()),
    "compression-ratio": float(df["compression-ratio"].median()),
    "city-L/100 km": float(df["city-L/100 km"].median()),
    "highway-L/100 km": float(df["highway-L/100 km"].median())
}