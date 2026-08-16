
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

from dataCleanup import compute_roster, data_cleanup_playerMode
from app.features import split_xy_playermode, split_xy_playermode_avg

DATA = Path(__file__).parent.parent / "data"
MODELS = Path(__file__).parent.parent / "models"

BEST = {
    "PlayerModeAvg": dict(max_depth=6,  learning_rate=0.07429014673298681,
                          min_child_weight=50, subsample=0.9951084994157238,
                          n_estimators=104),
    "PlayerMode":    dict(max_depth=10, learning_rate=0.01266928906877919,
                          min_child_weight=41, subsample=0.7001456545996178,
                          n_estimators=340),
}

if __name__ == "__main__":
    raw = pd.concat([pd.read_parquet(DATA / "train_allShots.parquet"),
                     pd.read_parquet(DATA / "val_allShots.parquet")], ignore_index=True)

    clean = raw[(raw["LOC_Y"] < 470) & (raw["SHOT_ZONE_BASIC"] != "Backcourt")].copy()
    clean["SHOT_ZONE_BASIC"] = clean["SHOT_ZONE_BASIC"].replace(
        {"Left Corner 3": "Corner 3", "Right Corner 3": "Corner 3"})
    id_to_cat = compute_roster(clean)
    print(f"roster: {len(id_to_cat)} players")

    data_cleanup_playerMode(raw, "trainval", id_to_cat)
    data_cleanup_playerMode(pd.read_parquet(DATA / "test_allShots.parquet"), "test", id_to_cat)

    df = pd.read_parquet(DATA / "trainval_playerMode.parquet")
    print(f"training on {len(df):,} shots")

    for label, split in [("PlayerModeAvg", split_xy_playermode_avg),
                         ("PlayerMode", split_xy_playermode)]:
        x, y = split(df)
        model = XGBClassifier(objective="binary:logistic", tree_method="hist",
                              enable_categorical=True, eval_metric="logloss",
                              **BEST[label])
        model.fit(x, y)
        model.save_model(str(MODELS / f"{label}.json"))
        print(f"  saved {label}.json")
