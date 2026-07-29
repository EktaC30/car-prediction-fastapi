from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_price, get_model_info
from defaults import DEFAULTS

app = FastAPI(
    title="Car Price Prediction API"
)


class CarInput(BaseModel):

    make: str
    fuel_type: str
    aspiration: str
    doors: str
    body_style: str
    drive_wheels: str
    engine_size: float
    horsepower: float
    peak_rpm: float


@app.get("/")
def health_check():

    return {
        "status": "API Running"
    }


@app.post("/predict")
def predict(data: CarInput):

    car = DEFAULTS.copy()

    car.update({
        "make": data.make,
        "aspiration": data.aspiration,
        "doors": data.doors,
        "body-style": data.body_style,
        "drive-wheels": data.drive_wheels,
        "engine-size": data.engine_size,
        "horsepower": data.horsepower,
        "peak-rpm": data.peak_rpm,
        "engine-location": "front"
    })

    # horsepower bin
    if data.horsepower <= 128:
        hp_bin = "Low"
    elif data.horsepower <= 208:
        hp_bin = "Medium"
    else:
        hp_bin = "High"

    car["horsepower-binned"] = hp_bin

    # fuel dummies
    if data.fuel_type.lower() == "diesel":
        car["fuel-type-diesel"] = 1
        car["fuel-type-gas"] = 0
    else:
        car["fuel-type-diesel"] = 0
        car["fuel-type-gas"] = 1

    price = predict_price(car)

    return {
        "predicted_price": price
    }

@app.get("/model-info")

def model_info():
    return get_model_info()