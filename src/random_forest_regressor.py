import os, mlflow
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from mlflow.tracking.client import MlflowClient



class Preprocessing():
    def __init__(self):
        pass

    def calculate_trip_duration_in_minutes(self, df):
        df['trip_duration_minutes'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
        df = df[(df['trip_duration_minutes'] >= 1) & (df['trip_duration_minutes'] <= 60)]
        return df

    def transform(self, df):
        df = self.calculate_trip_duration_in_minutes(df)
        categorical_features = ['PULocationID', 'DOLocationID']
        df[categorical_features] = df[categorical_features].astype(str)
        df['trip_route'] = df['PULocationID'] + '_' + df['DOLocationID']
        df = df[['trip_route', 'trip_distance', 'trip_duration_minutes']]
        y = df['trip_duration_minutes']
        X = df.drop(columns=['trip_duration_minutes'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
        return X_train, X_test, y_train, y_test


class RandomForestRegressor():
    def __init__(self):
        # CONNECT MLFLOW
        pass

    def fit(self, X_train, X_test, y_train, y_test):
        with mlflow.start_run():
            tags = {
                'model': 'random forest regression - GridSearchCV',
                'developer': 'karim',
                'dataset': 'yellow-taxi',
                'year': 2021,
                'month': 1,
                'features': features,
                'target': target
            }
            mlflow.set_tags(tags)

            model = RandomForestRegressor()
            param_grid = {
                'n_estimators': [20, 50, 100],
                'criterion': ['squared_error', 'absolute_error']
            }
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            y_score = best_model.score(X_test, y_test)
            y_pred = best_model.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            mlflow.log_metric('rmse', rmse)
            mlflow.sklearn.log_model(best_model, 'model')     


class RegisterModelMLFlow():
    def __init__(self, run_id, mlflow_tracking_uri, model_name, model):
        self.run_id = run_id
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_name = model_name
        self.model = model

        

    