## Set up and Installation

### Installing MLflow

Run one of the following commands:

- default MLflow: `pip install mlflow`

### Installing dependencies

```bash
pip install -r requirements.txt
```
### Launch MLflow UI

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

## Running this Project
### Individual RandomForest example

```{bash}
chmod +x train.sh
```

```{bash}
./train.sh
```

### In Browser

Go to link: <http://127.0.0.1:5000>
