# Model deployment project

RMSE is 3.35. I only trained it on a fraction of the January 2021 trips because mlflow.sklearn.log_model() would time out and I didn't have the time to find a better "fix".

If I had more time I would train it on a larger portion of the data (at least one month). I also would have scheduled monthly runs, ideally so that a prediction via the API for september would have been based on a model trained on data for sept of the previous year.