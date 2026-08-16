from nba_api.stats.endpoints import shotchartdetail
from nba_api.stats.static import teams
import time
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
ALL_TEAM_IDS = [t['id'] for t in teams.get_teams()]


def fetch_season(season):
    season_shots = []
    for team_id in ALL_TEAM_IDS:
        try:
            shots = shotchartdetail.ShotChartDetail(
                player_id=0, team_id=team_id,
                context_measure_simple='FGA',
                season_nullable=season, timeout=45
            )
            df = shots.get_data_frames()[0]
            if len(df) > 0:
                season_shots.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"  {season} | team {team_id} failed: {type(e).__name__}")
            time.sleep(1)
            continue

    result = pd.concat(season_shots, ignore_index=True)
    return result


def create_data(season_list, mode):
    all_shots = []
    for season in season_list:
        df = fetch_season(season)
        if len(df) > 0:
            all_shots.append(df)

    final_result = pd.concat(all_shots, ignore_index=True)
    out = DATA_DIR / f"{mode}_allShots.parquet"
    final_result.to_parquet(out)


train_season_list = ['2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']
val_season_list = ['2024-25', '2023-24']
test_season_list = ['2025-26']

if __name__ == '__main__':
    create_data(train_season_list, 'train')
    create_data(val_season_list, 'val')
    create_data(test_season_list, 'test')
