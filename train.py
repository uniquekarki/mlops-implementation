import click
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from utils import fetch_logged_data
from logger import rootlog
from pprint import pprint

RND_SEED = 42
EXP_NAME = "DemoBoston"


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


@click.command(help="Perform train (main entry point).")
@click.option("--seed", type=click.INT, default=RND_SEED,
              help="Seed for the random generator")
@click.argument("data_url")
def main(data_url, seed):
    # Data Example Preparation
    raw_df = pd.read_csv(data_url, delim_whitespace=True,
                         skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.33, random_state=seed
        )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    exps = mlflow.search_experiments()
    exp_names = [e.name for e in exps]
    if EXP_NAME in exp_names:
        experiment_id = exps[exp_names.index(EXP_NAME)].experiment_id
    else:
        experiment_id = mlflow.create_experiment(EXP_NAME)
        # , \
        # artifact_location=Path.cwd().joinpath("artifacts").as_uri(),
        # tags={"version": "v1", "priority": "P0"}
        # )
    print(f"Created New Experiment > ID: {experiment_id}")
    # mlflow.sklearn.autolog(registered_model_name="RF-BostonPredict"
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name="RF-1train",
                          experiment_id=experiment_id) as r:
        rootlog.info("---------- MLFlow cfg ---------")
        rootlog.info(f"Tracking URI: {mlflow.get_tracking_uri()}")
        rootlog.info(f"Run id:  {r.info.run_id}")
        rootlog.info(f"Experiment ID: {r.info.experiment_id}")
        rootlog.info(f"Artifact URI: {r.info.artifact_uri}")
        n_estimators = 100
        max_features = 3
        max_depth = 6
        rf = RandomForestRegressor(n_estimators=n_estimators,
                                   max_depth=max_depth,
                                   max_features=max_features,
                                   random_state=RND_SEED
                                   )
        rf.fit(X_train, y_train)
        # Make predictions
        predictions = rf.predict(X_test)

        # Create metrics
        metrics = eval_metrics(y_test, predictions)
        # Log Model
        mlflow.sklearn.log_model(sk_model=rf,
                                 artifact_path="model-artifact",
                                 registered_model_name="BostonPredict-RF")
        # Log Param
        mlflow.log_param("num_trees", n_estimators)
        mlflow.log_param("maxdepth", max_depth)
        mlflow.log_param("max_feat", max_features)
        # Log Metric
        mlflow.log_metrics(metrics)
        # mlflow.log_artifact(local_path='local-artifacts')

        # show logged data - Auto Log case
        for key, data in fetch_logged_data(r.info.run_id).items():
            rootlog.info("---------- logged {} ----------".format(key))
            pprint(data)


if __name__ == "__main__":
    main()
