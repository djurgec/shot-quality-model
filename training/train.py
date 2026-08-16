import mlflow
import optuna
import pandas as pd
from pathlib import Path

from mlflow.models import infer_signature
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

from app.features import split_xy

DATA_DIR = Path(__file__).parent.parent / "data"


def objective(trial):
    max_depth = trial.suggest_int("max_depth", 3, 10) #max depth of one tree
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True) #how much each tree contributes to the final prediction
    min_child_weight = trial.suggest_int("min_child_weight", 1, 50) #minimum weight one leaf has to hold
    subsample = trial.suggest_float("subsample", 0.6, 1.0) #fraction of training rows sampled for each tree
    n_estimators = trial.suggest_int("n_estimators", 100, 1000) #number of trees

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        model = XGBClassifier(objective='binary:logistic', tree_method='hist', enable_categorical=True,
                              eval_metric='logloss', max_depth=max_depth, learning_rate=learning_rate,
                              n_estimators=n_estimators, min_child_weight=min_child_weight, subsample=subsample)
        model.fit(train_x, train_y)
        probs = model.predict_proba(val_x)[:, 1]
        brier_loss = brier_score_loss(val_y, probs)
        mlflow.log_metric('brier_loss', brier_loss)
        mlflow.log_param("max_depth", model.max_depth)
        mlflow.log_param("learning_rate", model.learning_rate)
        mlflow.log_param("min_child_weight", model.min_child_weight)
        mlflow.log_param("subsample", model.subsample)
        mlflow.log_param("n_estimators", model.n_estimators)

    return brier_loss



if __name__ == '__main__':
    train_df = pd.read_parquet(DATA_DIR / 'train_cleanData.parquet')
    val_df = pd.read_parquet(DATA_DIR / 'val_cleanData.parquet')
    test_df = pd.read_parquet(DATA_DIR / 'test_cleanData.parquet')
    train_x, train_y = split_xy(train_df)
    test_x, test_y = split_xy(test_df)
    val_x, val_y = split_xy(val_df)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("NBA Shot Quality Model Final")

    with mlflow.start_run(run_name='xgboost_optuna_study_finalv2'):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_brier", study.best_value)
        best = XGBClassifier(
            objective='binary:logistic', tree_method='hist',
            enable_categorical=True, eval_metric='logloss',
            **study.best_params
        )
        best.fit(train_x, train_y)

        signature = infer_signature(train_x, best.predict_proba(train_x)[:, 1])
        mlflow.xgboost.log_model(best, name='TunedXGBoostModelFinal', signature=signature)

        for name, fn in [("param_importances", optuna.visualization.matplotlib.plot_param_importances),
                         ("slice", optuna.visualization.matplotlib.plot_slice)]:
            fn(study)
            plt.savefig(f"{name}.png", dpi=150, bbox_inches="tight")
            mlflow.log_artifact(f"{name}.png")
            plt.close()
