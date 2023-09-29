from random_forest_regressor import Preprocessing, RandomForestRegressor, RegisterModelMLFlow
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
def train(bucket, year, month):
    load_dotenv()
    # 1) Load data
    df = pd.read_parquet(f'gs://{bucket}/training_data/yellow_tripdata_{year}-{month:02d}.parquet')
    df = df.iloc[:10000]

    # 2) Preprocess data
    preprocessor = Preprocessing()
    X_train, X_test, y_train, y_test = preprocessor.transform(df)

    # 3) Training
    model = RandomForestRegressor()
    model.fit(X_train, X_test, y_train, y_test)
    


if __name__ == '__main__':
    cli()