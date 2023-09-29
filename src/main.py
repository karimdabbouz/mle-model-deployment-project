from random_forest_regressor import Preprocessing, RandomForest, RegisterModelMLFlow
import click, os
import pandas as pd
from dotenv import load_dotenv



@click.group()
def cli():
    pass


@cli.command()
@click.option('--bucket', '-b', help='Name of GCS bucket')
@click.option('--year', '-y', help='Year of trips to train model on', default=2021)
@click.option('--month', '-m', help='Month of trips to train model on', default=1)
@click.option('--mlflow_tracking_uri', help='URI of the MLFlow tracking server', default='http://34.107.124.157:5000/')
@click.option('--experiment_name', help='Name of the experiment', default='yellow-taxi-duration-random-forest-regression')
def train(bucket, year, month, mlflow_tracking_uri, experiment_name, run_id, model_name, model_version, stage):
    load_dotenv()
    df = pd.read_parquet(f'gs://{bucket}/training_data/yellow_tripdata_{year}-{month:02d}.parquet')
    df = df.iloc[:10000] # random forest seems to take quite long...
    preprocessor = Preprocessing()
    X_train, X_test, y_train, y_test = preprocessor.transform(df)
    model = RandomForest(mlflow_tracking_uri, experiment_name)
    model.fit(X_train, X_test, y_train, y_test)


@cli.command()
@click.option('--mlflow_tracking_uri', help='URI of the MLFlow tracking server', default='http://34.107.124.157:5000/')
@click.option('--run_id', help='ID of MLFlow run to register as model', default='2d4680fc9be04d9aa0a26d77e8544246')
@click.option('--model_name', help='Name to register model as', default='random-forest-duration')
@click.option('--model_version', help='Version of model to register')
@click.option('--stage', help='Stage of model to register', default='Production')
def register_model(run_id, mlflow_tracking_uri, model_name, model_version, stage):
    register_model = RegisterModelMLFlow(run_id, mlflow_tracking_uri, model_name, model_version, stage)
    register_model.register()


if __name__ == '__main__':
    cli()