import json

from pathlib import Path
import pandas as pd

from xgboost import XGBClassifier

from app.features import split_xy_playermode, split_xy_playermode_avg

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
SHARED_DIR = Path(__file__).parent.parent / "shared"

def predict_at(loc_x, loc_y, dist, zone, player_cat=None):
    row = pd.DataFrame([{'LOC_X': loc_x, 'LOC_Y':loc_y, 'SHOT_DISTANCE': dist, }])
    if player_cat is not None:
        row['PLAYER_CAT'] = player_cat
        row['PLAYER_CAT'] = row['PLAYER_CAT'].astype('category')
    row['SHOT_ZONE_BASIC'] = zone
    row['SHOT_ZONE_BASIC'] = row['SHOT_ZONE_BASIC'].astype('category')
    m =model_player if player_cat is not None else model_avg
    return float(m.predict_proba(row)[:, 1][0])


if __name__ == '__main__':
    model_avg = XGBClassifier()
    model_player = XGBClassifier()
    test_df = pd.read_parquet(DATA_DIR / 'test_playerMode.parquet')

    model_avg.load_model(str(MODEL_DIR / 'player_mode_avg.json'))
    model_player.load_model(str(MODEL_DIR / 'player_mode.json'))
    player_zones = json.loads((SHARED_DIR / 'player_zones.json').read_text())
    playercode_dict = {v['name']:int(code) for code, v in player_zones.items()}
    playername_dict = {int(code):v['name'] for code, v in player_zones.items()}
    zones = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range',
            'Above the Break 3', 'Corner 3']

    league_avg = predict_at(0, 240, 24, 'Above the Break 3')
    print(f"league avg (Above the Break 3): {league_avg:.3f}\n")
    sample_players = ['Stephen Curry', 'Klay Thompson', 'Marcus Smart', 'Russell Westbrook']
    for player in sample_players:
        cat = playercode_dict.get(player)
        p = predict_at(0, 240, 24, 'Above the Break 3', player_cat=cat)
        print(f"{player}: {p:.3f}, delta {p-league_avg:.3f}")

    xp, _ = split_xy_playermode(test_df)
    xa, _ = split_xy_playermode_avg(test_df)

    test_df = test_df.copy()
    test_df['p_player'] = model_player.predict_proba(xp)[:, 1]
    test_df['p_league'] = model_avg.predict_proba(xa)[:, 1]
    test_df['delta'] = test_df['p_player'] - test_df['p_league']

    roster = test_df[test_df['PLAYER_CAT'] != 0]
    for zone in zones:
        roster_zone = roster[roster['SHOT_ZONE_BASIC'] == zone]
        per_player = roster_zone.groupby('PLAYER_CAT')['delta'].agg(['mean', 'size'])
        per_player['name'] =per_player.index.map(playername_dict)
        per_player = per_player[per_player['size'] >= 20]
        print(f"delta distribution ({zone}):\n")
        print(per_player['mean'].describe().round(3))
        print(f"biggest positive deltas ({zone}):  \n")
        print(per_player.sort_values('mean', ascending=False).head(10))
        print(f"biggest negative deltas ({zone}): \n")
        print(per_player.sort_values('mean').head(10))
    #
    # CalibrationDisplay.from_predictions(test_df['SHOT_MADE_FLAG'], test_df['p_league'],
    #                                     n_bins=15, strategy='quantile', name='league')
    # CalibrationDisplay.from_predictions(test_df['SHOT_MADE_FLAG'], test_df['p_player'],
    #                                     n_bins=15, strategy='quantile', name='player', ax=plt.gca())
    # plt.show()



