import gradio as gr
import requests


API_URL = "http://127.0.0.1:8000/predict"


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
            json=payload
        )

        response.raise_for_status()

        prediction = response.json()[
            "predicted_price"
        ]

        return (
            f"Estimated Car Price: "
            f"${prediction:,.2f}"
        )

    except Exception as e:

        return f"Error: {str(e)}"


with gr.Blocks(
    title="Car Price Prediction"
) as demo:

    gr.Markdown(
        """
        # 🚗 Car Price Prediction
        
        Enter vehicle details and predict the car price.
        """
    )

    with gr.Row():

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
            label="Make"
        )

        fuel_type = gr.Dropdown(
            choices=["gas", "diesel"],
            label="Fuel Type"
        )

    with gr.Row():

        aspiration = gr.Dropdown(
            choices=["std", "turbo"],
            label="Aspiration"
        )

        doors = gr.Dropdown(
            choices=["two", "four"],
            label="Doors"
        )

    with gr.Row():

        body_style = gr.Dropdown(
            choices=[
                "convertible",
                "hardtop",
                "hatchback",
                "sedan",
                "wagon"
            ],
            label="Body Style"
        )

        drive_wheels = gr.Dropdown(
            choices=[
                "fwd",
                "rwd",
                "4wd"
            ],
            label="Drive Wheels"
        )

    engine_size = gr.Number(
        label="Engine Size",
        value=130
    )

    horsepower = gr.Number(
        label="Horsepower",
        value=111
    )

    peak_rpm = gr.Number(
        label="Peak RPM",
        value=5000
    )

    predict_button = gr.Button(
        "Predict Price"
    )

    output = gr.Textbox(
        label="Prediction"
    )

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

demo.launch()