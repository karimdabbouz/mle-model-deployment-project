from fastapi import FastAPI
from data_model import TaxiRide, TaxiRidePrediction
from predict import predict


app = FastAPI()


@app.get('/')
def index():
    return {'message': 'hi there'}


@app.post('/predict', response_model=TaxiRidePrediction)
def predict_duration(data: TaxiRide):
    prediction = predict('random-forest-duration', data)
    return TaxiRidePrediction(**data.dict(), predicted_duration=prediction)