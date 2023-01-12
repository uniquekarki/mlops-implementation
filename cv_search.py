import os
import tempfile
import click
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from logger import rootlog
from utils import fetch_logged_data

RND_SEED = 42
EXP_NAME = "DemoBoston"


def log_run(gridsearch: GridSearchCV, best_metrics: dict, *,
            run_name: str = '', experiment_id: int = '0',
            model_name: str = '', conda_env: str | None = None,
            register_model: bool = False, tags: dict = {}):
    """Logging of cross validation results to mlflow tracking server
    From: https://gist.github.com/liorshk/9dfcb4a8e744fc15650cbd4c2b0955e5

    Args:
        experiment_id (int): experiment ID
        model_name (str): Name of the model
        run_index (int): Index of the run (in Gridsearch)
        conda_env (str): A dictionary that describes the conda environment)
        tags (dict): Dictionary of extra data and tags (usually features)
    """
    cv_results = gridsearch.cv_results_
    with mlflow.start_run(run_name=run_name,
                          description='Gridsearch cross-validation',
                          experiment_id=experiment_id) as parent_run:
        run_index = gridsearch.best_index_
        mlflow.log_param("folds", gridsearch.cv)

        rootlog.info("Logging parameters")
        params = list(gridsearch.param_grid.keys())
        for param in params:
            mlflow.log_param(param,
                             cv_results["param_%s" % param][run_index])

        rootlog.info("Logging metrics")
        for score_name in [score for score in cv_results
                           if "mean_test" in score]:
            mlflow.log_metric(
                score_name,
                cv_results[score_name][run_index]
                )
            mlflow.log_metric(
                score_name.replace("mean", "std"),
                cv_results[score_name
                           .replace("mean", "std")][run_index])

        # If None, no model (no new version) will be registered
        registered_model_name = None
        if register_model:
            registered_model_name = 'BostonDemo-Opt-RF'
        rootlog.info("Logging model")
        mlflow.sklearn.log_model(gridsearch.best_estimator_,
                                 model_name,
                                 registered_model_name=registered_model_name
                                 )
        mlflow.log_metrics(best_metrics)

        rootlog.info("Logging CV results matrix")
        tempdir = tempfile.TemporaryDirectory().name
        rootlog.debug(f'Creating artifact temp directory {tempdir}')
        os.mkdir(tempdir)
        timestamp = datetime.now().isoformat() \
                            .split(".")[0].replace(":", ".")
        filename = f"{timestamp}-cv_results_{model_name}.csv"
        csv = os.path.join(tempdir, filename)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        pd.DataFrame(cv_results).to_csv(csv, index=False)
        rootlog.info("Logging extra data related to the experiment")
        mlflow.log_artifact(csv, "cv_results")
        mlflow.set_tags(tags)

        run_id = parent_run.info.run_uuid
        experiment_id = parent_run.info.experiment_id
        # mlflow.end_run()
        rootlog.debug("---------- MLFlow cfg ---------")
        rootlog.debug(f"Parent RunID: {run_id} - ExpID: {experiment_id}")
        rootlog.debug(f"Artifact URI: {mlflow.get_artifact_uri()}")

        for key, data in fetch_logged_data(run_id).items():
            rootlog.debug("---------- logged {} ----------".format(key))
            rootlog.debug(data)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}


@click.command(help="Perform grid search over train (main entry point).")
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
        data, target, test_size=0.2, random_state=seed
        )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    exps = mlflow.search_experiments()
    exp_names = [e.name for e in exps]
    if EXP_NAME in exp_names:
        experiment_id = exps[exp_names.index(EXP_NAME)].experiment_id
    else:
        experiment_id = mlflow.create_experiment(EXP_NAME)
        # artifact_location=Path.cwd().joinpath("artifacts").as_uri()
        # tags={"version": "v1", "priority": "P0"}
        # )
    rootlog.debug(f"Created New Experiment > ID: {experiment_id}")
    mlflow.set_experiment(experiment_id=experiment_id)
    param_grid = {
        'n_estimators': [50, 200],
        'max_features': [1, 'sqrt', 'log2'],
        'max_depth': [5, 6, 7, 8, 9],
        'criterion': ['squared_error', 'poisson']
        }
    rf = RandomForestRegressor()
    CV_rf = GridSearchCV(estimator=rf, param_grid=param_grid,
                         cv=5, scoring='r2')
    CV_rf.fit(X_train, y_train)
    predictions = CV_rf.best_estimator_.predict(X_test)
    metrics = eval_metrics(y_test, predictions)
    log_run(CV_rf, metrics, experiment_id=experiment_id,
            run_name="GridSearch CV",
            model_name="BostonPredict-RF",
            register_model=True)


if __name__ == "__main__":
    main()
