# MLFlow Experiment Example

Example of how to run a simple random forest algorithm on an example dataset ([Boston house price data](http://lib.stat.cmu.edu/datasets/boston)) and a hyperparameter tuning of a random forest algorithm using a cross-validated grid search. We use scikit learn library to perform the machine learnign and statistics part and MLflow for tracking and saving the outcomes.

## Set up and Installation

### Setting up

Clone repo and create virtual environment (make sure you have virtualenv installed)

```bash
git clone git@github.com:RESILEYES/mlflow-experiment-example.git &&
cd mlflow-experiment-example &&
python3 -m venv .venv &&
source .venv/bin/activate
```

### Installing MLflow

Run one of the following commands:

- default MLflow: `pip install mlflow`

- MLflow with the experimental MLflow Pipelines component `pip install mlflow[pipelines]`

- MLflow with extra ML libraries and 3rd-party tools `pip install mlflow[extras]`

- a lightweight version of MLflow `pip install mlflow-skinny`

### Installing dependencies

```bash
pip install -r requirements.txt
```

In this example, mlflow is already in the `requirements.txt` file.

### Launch MLflow UI

Here is the fast command to launch the UI:

```bash
mlflow ui
```

The commandline output should look like this:

```plain
[2022-10-19 13:48:47 +0200] [4102] [INFO] Starting gunicorn 20.1.0
[2022-10-19 13:48:47 +0200] [4102] [INFO] Listening at: http://127.0.0.1:5000 (4102)
[2022-10-19 13:48:47 +0200] [4102] [INFO] Using worker: sync
[2022-10-19 13:48:47 +0200] [4104] [INFO] Booting worker with pid: 4104
```

 > **_NOTE:_** For more details, and to be able to configure the mlflow tracking server check the [Tracking UI](#tracking-ui) section.

To access to the web based interface, click or sopy the IP address in your browser. If this is run on a remote machine, an ssh tunnel is needed to access from your local machine. In the general case, the ssh command is the following:

```bash
ssh -L local_port:remote_address:remote_port username@server.com
```

## Content and Purpose

This MLflow project contains two entries each one corresponds to an example and they are defined in the `MLproject` file.
The latter file should be structured as follows:

```yaml
name: <project-name>

python_env: python_env.yaml # or conda_env

entry_points:
  <entry-name>:
    parameters:
      ## some parameters and their default values
      seed: {type: int, default: 42}
    command: "python <myscript.py> <some arguments> --seed {seed}"
```

In this project, the `MLproject` has 2 targets:

- `train`: train simple random forest model on the boston housing dataset.

- `optimize`: perform a cross validated random search over a prespecified parameter space and saves all cross valisations results and the best performing model. This example tries to optimize the $R^2$ metric of  dataset.

`train` entry calls the script `train.py` while optimize calls `cv_search.py`. For those two entries, the random seed is fixed (seed=42) and the dataset URL is given as a parameter.
 > **_NOTE:_** Passing the dataset URL or path as an argument to a ML script can be useful to allow dataset versionning in the future, especially when used inside a a ML tracking tool/platform.

## Running this Example

You can run any of the two targets defined in `MLProject` file as a standard MLflow run. For these two exaples, we choose to use the local virtual environment instead of conda (which the default virtual environment manager for mlflow) using the `--env-manager=local` argument. We specify also the name of the experiment in the MLFlow command using `--experiment-name` argument.

> **_NOTE:_**  To manage experiments (and by that I mean creating a new experiment, selecting an existing one listing existing experiments), there are two manners:
>
> 1. The first one consists of doing that through MLFLow commands in the terminal. For instance, to create a new experiment named 'my-fantastic-experiment', it is possible to use:<br/>
>```mlflow experiments create -n my-fantastic-experiment```
>
> 2. The second way is to use the Tracking Service API. Through a client SDK in the `mlflow.client` module, it is possible to query data about past runs, log additional information about them, create experiments, add tags to a run, and more.
>
> In The following examples, we use the `mlflow.create_experiment()` Python API. Example: `mlflow.create_experiment("my-fantastic-experiment")`

### Individual RandomForest example

```{bash}
mlflow run -e train --env-manager=local --experiment-name=DemoBoston ./
```

<details>
  <summary><i>Command-line output:</i></summary>

```{plain}
2022/09/30 14:31:40 INFO mlflow.projects: 'DemoBoston' does not exist. Creating a new experiment
2022/09/30 14:31:41 INFO mlflow.projects.utils: === Created directory /tmp/tmpxow8paww for downloading remote URIs passed to arguments of type 'path' ===
2022/09/30 14:31:41 INFO mlflow.projects.backend.local: === Running command 'python train.py http://lib.stat.cmu.edu/datasets/boston --seed 42' in run with ID '44462f4ceaaf48f1ba085aa54853ec6c' === 
Created New Experiment > ID: 1
[root][INFO]  ---------- MLFlow cfg --------- (train.py:57)
[root][INFO]  Tracking URI: http://127.0.0.1:5000 (train.py:58)
[root][INFO]  Run id:  44462f4ceaaf48f1ba085aa54853ec6c (train.py:59)
[root][INFO]  Experiment ID: 1 (train.py:60)
[root][INFO]  Artifact URI: file:///home/inarighas/Projects/mlflow-experiment/mlruns/1/44462f4ceaaf48f1ba085aa54853ec6c/artifacts (train.py:61)
Successfully registered model 'BostonPredict-RF'.
2022/09/30 14:31:45 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: BostonPredict-RF, version 1
Created version '1' of model 'BostonPredict-RF'.
[root][INFO]  ---------- logged params ---------- (train.py:90)
{'data_url': 'http://lib.stat.cmu.edu/datasets/boston',
 'max_feat': '3',
 'maxdepth': '6',
 'num_trees': '100',
 'seed': '42'}
[root][INFO]  ---------- logged metrics ---------- (train.py:90)
{'MAE': 2.299510577409418, 'R2': 0.8364934352175877, 'RMSE': 3.517660150133989}
[root][INFO]  ---------- logged tags ---------- (train.py:90)
{}
[root][INFO]  ---------- logged artifacts ---------- (train.py:90)
['model-artifact/MLmodel',
 'model-artifact/conda.yaml',
 'model-artifact/model.pkl',
 'model-artifact/python_env.yaml',
 'model-artifact/requirements.txt']
2022/09/30 14:31:45 INFO mlflow.projects: === Run (ID '44462f4ceaaf48f1ba085aa54853ec6c') succeeded ===
```

</details>

### GridSearch CV example

```bash
mlflow-experiment mlflow run -e optimize --env-manager=local --experiment-name=DemoBoston ./ 
```

<details>
  <summary><i>Command-line output:</i></summary>

```{plain}
2022/09/30 14:36:24 INFO mlflow.projects.utils: === Created directory /tmp/tmp3zum_96o for downloading remote URIs passed to arguments of type 'path' ===
2022/09/30 14:36:24 INFO mlflow.projects.backend.local: === Running command 'python cv_search.py http://lib.stat.cmu.edu/datasets/boston --seed 42' in run with ID 'cc4cc85ff76e422ebac274ec8dc1e653' === 
[root][DEBUG]  Created New Experiment > ID: 1 (cv_search.py:129)
[root][INFO]  Logging parameters (cv_search.py:41)
[root][INFO]  Logging metrics (cv_search.py:47)
[root][INFO]  Logging model (cv_search.py:63)
Successfully registered model 'BostonDemo-Opt-RF'.
2022/09/30 14:36:55 INFO mlflow.tracking._model_registry.client: Waiting up to 300 seconds for model version to finish creation.                     Model name: BostonDemo-Opt-RF, version 1
Created version '1' of model 'BostonDemo-Opt-RF'.
[root][INFO]  Logging CV results matrix (cv_search.py:70)
[root][DEBUG]  Creating artifact temp directory /tmp/tmp2r1hfg_b (cv_search.py:72)
[root][INFO]  Logging extra data related to the experiment (cv_search.py:81)
[root][DEBUG]  ---------- MLFlow cfg --------- (cv_search.py:88)
[root][DEBUG]  Parent RunID: cc4cc85ff76e422ebac274ec8dc1e653 - ExpID: 1 (cv_search.py:89)
[root][DEBUG]  Artifact URI: file:///home/inarighas/Projects/mlflow-experiment/mlruns/1/cc4cc85ff76e422ebac274ec8dc1e653/artifacts (cv_search.py:90)
[root][DEBUG]  ---------- logged params ---------- (cv_search.py:93)
[root][DEBUG]  {'data_url': 'http://lib.stat.cmu.edu/datasets/boston', 'folds': '5', 'n_estimators': '200', 'max_depth': '9', 'criterion': 'poisson', 'max_features': 'log2', 'seed': '42'} (cv_search.py:94)
[root][DEBUG]  ---------- logged metrics ---------- (cv_search.py:93)
[root][DEBUG]  {'RMSE': 3.1589819580824265, 'R2': 0.8639212845507542, 'std_test_score': 0.04804912982520214, 'MAE': 2.007434222111125, 'mean_test_score': 0.8474457246627329} (cv_search.py:94)
[root][DEBUG]  ---------- logged tags ---------- (cv_search.py:93)
[root][DEBUG]  {} (cv_search.py:94)
[root][DEBUG]  ---------- logged artifacts ---------- (cv_search.py:93)
[root][DEBUG]  ['BostonPredict-RF/MLmodel', 'BostonPredict-RF/conda.yaml', 'BostonPredict-RF/model.pkl', 'BostonPredict-RF/python_env.yaml', 'BostonPredict-RF/requirements.txt', 'cv_results/2022-09-30T14.36.55-cv_results_BostonPredict-RF.csv'] (cv_search.py:94)
2022/09/30 14:36:55 INFO mlflow.projects: === Run (ID 'cc4cc85ff76e422ebac274ec8dc1e653') succeeded ===
```

</details>

## Tracking UI

Once the runs executed, the results are saved in the `./mlruns/` directorty and in the `mlruns.db` (SQLite) database.

<details><summary><i> unroll to see <code>mlruns/</code> tree</i></summary>

```{plain}
mlruns/
├── 0
│   └── meta.yaml
└── 1
    ├── 44462f4ceaaf48f1ba085aa54853ec6c
    │   ├── artifacts
    │   │   └── model-artifact
    │   │       ├── conda.yaml
    │   │       ├── MLmodel
    │   │       ├── model.pkl
    │   │       ├── python_env.yaml
    │   │       └── requirements.txt
    │   ├── meta.yaml
    │   ├── metrics
    │   │   ├── MAE
    │   │   ├── R2
    │   │   └── RMSE
    │   ├── params
    │   │   ├── data_url
    │   │   ├── maxdepth
    │   │   ├── max_feat
    │   │   ├── num_trees
    │   │   └── seed
    │   └── tags
    │       ├── mlflow.log-model.history
    │       ├── mlflow.project.backend
    │       ├── mlflow.project.entryPoint
    │       ├── mlflow.runName
    │       ├── mlflow.source.name
    │       ├── mlflow.source.type
    │       └── mlflow.user
    ├── cc4cc85ff76e422ebac274ec8dc1e653
    │   ├── artifacts
    │   │   ├── BostonPredict-RF
    │   │   │   ├── conda.yaml
    │   │   │   ├── MLmodel
    │   │   │   ├── model.pkl
    │   │   │   ├── python_env.yaml
    │   │   │   └── requirements.txt
    │   │   └── cv_results
    │   │       └── 2022-09-30T14.36.55-cv_results_BostonPredict-RF.csv
    │   ├── meta.yaml
    │   ├── metrics
    │   │   ├── MAE
    │   │   ├── mean_test_score
    │   │   ├── R2
    │   │   ├── RMSE
    │   │   └── std_test_score
    │   ├── params
    │   │   ├── criterion
    │   │   ├── data_url
    │   │   ├── folds
    │   │   ├── max_depth
    │   │   ├── max_features
    │   │   ├── n_estimators
    │   │   └── seed
    │   └── tags
    │       ├── mlflow.log-model.history
    │       ├── mlflow.note.content
    │       ├── mlflow.project.backend
    │       ├── mlflow.project.entryPoint
    │       ├── mlflow.runName
    │       ├── mlflow.source.name
    │       ├── mlflow.source.type
    │       └── mlflow.user
    └── meta.yaml

15 directories, 50 files
```

</details>

The tracking user inteface lets you visualize, search and compare runs, as well as download run artifacts or metadata for analysis in other tools.
If you log runs to a local mlruns directory, you can access by runing `mlflow ui` and it loads the corresponding runs. Alternatively, the MLflow tracking server serves the same UI and enables remote storage of run artifacts. In that case, you can view the UI using URL http://<ip address of your MLflow tracking server>:5000 in your browser from any machine, including any remote machine that can connect to your tracking server.

There are multiple configuration to set MLFlow tracking server (locally or remotely). Currently, we can either use the above `mlflow ui command` since everything is saved locally in `mlruns/` or use

```{bash}
mlflow server \
    --registry-store-uri sqlite:///mlruns.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 \
    --port 5000
```

which is more configurable. for more details, see ["MLFlow tracking documentation"](https://www.mlflow.org/docs/latest/tracking.html#how-runs-and-artifacts-are-recorded).

In this case, opening  <http://127.0.0.1:5000> in your favourite browser lets you access the tracking UI.
