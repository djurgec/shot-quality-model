import mlflow.xgboost
from mlflow.models import infer_signature
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

from app.features import split_xy

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_PATH  = Path(__file__).parent.parent / "models" / "shot_mode.ubj"

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("NBA Shot Quality Model Final")
    train_df = pd.read_parquet(DATA_DIR / 'train_cleanData.parquet')
    val_df = pd.read_parquet(DATA_DIR / 'val_cleanData.parquet')
    test_df = pd.read_parquet(DATA_DIR / 'test_cleanData.parquet')
    train_x, train_y = split_xy(train_df)
    test_x, test_y = split_xy(test_df)
    val_x, val_y = split_xy(val_df)
    final_x = pd.concat([train_x, val_x])
    final_y = pd.concat([train_y, val_y])

    model = mlflow.xgboost.load_model(MODEL_PATH)
    best_params = model.get_params()
    final_model = XGBClassifier(**best_params)
    final_model.fit(final_x, final_y)
    probs = model.predict_proba(test_x)[:, 1]
    signature = infer_signature(final_x, final_model.predict_proba(final_x)[:, 1])
    mlflow.xgboost.log_model(final_model, name='TunedXGBoostModelFinal', signature=signature)
    final_model.save_model(str(Path(__file__).parent.parent / "models" / "shot_mode.ubj"))

    print(f"test brier:    {brier_score_loss(test_y, probs):.4f}")
    print(f"test log loss: {log_loss(test_y, probs):.4f}")
    print(f"test auc:      {roc_auc_score(test_y, probs):.4f}\n")