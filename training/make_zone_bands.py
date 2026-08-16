
import json
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from app.features import split_xy_playermode, split_xy_playermode_avg

DATA = Path(__file__).parent.parent / "data"
MODELS = Path(__file__).parent.parent / "models"
OUT = Path(__file__).parent.parent / "shared" / "zone_bands.json"
MIN_SHOTS = 20

if __name__ == "__main__":
    df = pd.read_parquet(DATA / "trainval_playerMode.parquet")

    player, avg = XGBClassifier(), XGBClassifier()
    player.load_model(str(MODELS / "player_mode.json"))
    avg.load_model(str(MODELS / "player_mode_avg.json"))
    df["delta"] = (player.predict_proba(split_xy_playermode(df)[0])[:, 1]
                   - avg.predict_proba(split_xy_playermode_avg(df)[0])[:, 1])

    bands = {}
    for zone, sub in df.groupby("SHOT_ZONE_BASIC"):
        per = sub.groupby("PLAYER_CAT")["delta"].agg(["mean", "size"])
        per = per[per["size"] >= MIN_SHOTS]["mean"] #transform to a series
        q = per.quantile([.05, .30, .70, .95])
        bands[zone] = {"elite": round(q[.95], 4), "above": round(q[.70], 4),
                       "below": round(q[.30], 4), "poor": round(q[.05], 4)}
        print(f"{zone:<22} n={len(per):<4} "
              f"elite>{q[.95]:+.4f}  above>{q[.70]:+.4f}  "
              f"below>{q[.30]:+.4f}  poor<={q[.05]:+.4f}")

    OUT.write_text(json.dumps(bands, indent=2) + "\n")
    print(f"\nwrote {OUT}")
