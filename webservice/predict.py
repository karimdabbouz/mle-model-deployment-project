import mlflow
import os
from dotenv import load_dotenv
import pandas as pd



def prepare_features(ride):
    features = pd.DataFrame([{'trip_route': f'{ride.PULocationID}_{ride.DOLocationID}', 'trip_distance': ride.trip_distance}])
    print(features)
    return features


def load_model(model_name):
    stage = 'Production'
    model_uri = f'models:/{model_name}/{stage}'
    model = mlflow.pyfunc.load_model(model_uri)
    return model


def predict(model_name, data):
    load_dotenv()
    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_input = prepare_features(data)
    model = load_model(model_name)
    prediction = model.predict(model_input)
    return float(prediction[0])