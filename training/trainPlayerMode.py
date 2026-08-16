import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from pathlib import Path
from functools import partial

from mlflow.models import infer_signature
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from app.features import split_xy_playermode, split_xy_playermode_avg

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"


def objective(trial, train_x, train_y, val_x, val_y):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
    }
    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        model = XGBClassifier(
            objective='binary:logistic', tree_method='hist',
            enable_categorical=True, eval_metric='logloss', **params
        )
        model.fit(train_x, train_y)
        brier = brier_score_loss(val_y, model.predict_proba(val_x)[:, 1])
        mlflow.log_params(params)
        mlflow.log_metric('brier_loss', brier)
    return brier

def run_study(label, split_fn, train_df, val_df, n_trials=50):
    train_x, train_y = split_fn(train_df)
    val_x,   val_y   = split_fn(val_df)

    mlflow.set_experiment(f"NBA Shot Quality Model {label} Final")

    with mlflow.start_run(run_name=f"xgboost_optuna_study_{label}"):
        study = optuna.create_study(direction='minimize')
        study.optimize(
            partial(objective, train_x=train_x, train_y=train_y,
                    val_x=val_x, val_y=val_y),
            n_trials=n_trials,
        )
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_brier", study.best_value)
        best = XGBClassifier(
            objective='binary:logistic', tree_method='hist',
            enable_categorical=True, eval_metric='logloss',
            **study.best_params
        )
        best.fit(train_x, train_y)

        best.save_model(str(MODEL_DIR / f"{label}.json"))

        signature = infer_signature(train_x, best.predict_proba(train_x)[:, 1])
        mlflow.xgboost.log_model(best, name=f'{label}_final', signature=signature)

        for name, fn in [("param_importances", optuna.visualization.matplotlib.plot_param_importances),
                         ("slice", optuna.visualization.matplotlib.plot_slice)]:
            fn(study)
            plt.savefig(f"{label}_{name}.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact(f"{label}_{name}.png")
            plt.close()

    return best


if __name__ == '__main__':
    train_df = pd.read_parquet(DATA_DIR / 'train_playerMode.parquet')
    val_df   = pd.read_parquet(DATA_DIR / 'val_playerMode.parquet')

    mlflow.set_tracking_uri("http://localhost:5000")

    run_study("PlayerModeAvg", split_xy_playermode_avg, train_df, val_df)

    run_study("PlayerMode", split_xy_playermode, train_df, val_df)