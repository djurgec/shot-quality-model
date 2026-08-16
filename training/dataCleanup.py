import pandas as pd
import numpy as np

from nba_api.stats.static import teams
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent.parent / "data"
SHARED_DIR = Path(__file__).parent.parent / "shared"

TEAM_ABBR = {t['id']: t['abbreviation'] for t in teams.get_teams()}
CATEGORY_MAP = {
    'Cutting Dunk Shot': 'Dunk',
    'Running Dunk Shot': 'Dunk',
    'Driving Dunk Shot': 'Dunk',
    'Alley Oop Dunk Shot': 'Dunk',
    'Running Alley Oop Dunk Shot': 'Dunk',
    'Reverse Dunk Shot': 'Dunk',
    'Driving Reverse Dunk Shot': 'Dunk',
    'Running Reverse Dunk Shot': 'Dunk',
    'Dunk Shot': 'Dunk',

    'Jump Shot': 'Jump Shot',
    'Turnaround Jump Shot': 'Jump Shot',
    'Running Jump Shot': 'Jump Shot',
    'Jump Bank Shot': 'Jump Shot',
    'Driving Bank shot': 'Jump Shot',

    'Pullup Jump shot': 'Pull-up',
    'Pullup Bank shot': 'Pull-up',
    'Running Pull-Up Jump Shot': 'Pull-up',

    'Step Back Jump shot': 'Step Back',
    'Step Back Bank Jump Shot': 'Step Back',

    'Layup Shot': 'Layup',
    'Driving Layup Shot': 'Layup',
    'Driving Reverse Layup Shot': 'Layup',
    'Reverse Layup Shot': 'Layup',
    'Finger Roll Layup Shot': 'Layup',
    'Driving Finger Roll Layup Shot': 'Layup',
    'Cutting Layup Shot': 'Layup',
    'Cutting Finger Roll Layup Shot': 'Layup',
    'Running Layup Shot': 'Layup',
    'Running Reverse Layup Shot': 'Layup',
    'Running Finger Roll Layup Shot': 'Layup',
    'Alley Oop Layup shot': 'Layup',
    'Running Alley Oop Layup Shot': 'Layup',

    'Putback Layup Shot': 'Putback',
    'Tip Layup Shot': 'Putback',
    'Tip Dunk Shot': 'Putback',
    'Putback Dunk Shot': 'Putback',

    'Driving Floating Jump Shot': 'Floater',
    'Floating Jump shot': 'Floater',
    'Driving Floating Bank Jump Shot': 'Floater',

    'Turnaround Hook Shot': 'Hook Shot',
    'Driving Hook Shot': 'Hook Shot',
    'Hook Shot': 'Hook Shot',
    'Driving Bank Hook Shot': 'Hook Shot',
    'Turnaround Bank Hook Shot': 'Hook Shot',
    'Hook Bank Shot': 'Hook Shot',

    'Fadeaway Jump Shot': 'Fadeaway',
    'Turnaround Fadeaway shot': 'Fadeaway',
    'Fadeaway Bank shot': 'Fadeaway',
    'Turnaround Fadeaway Bank Jump Shot': 'Fadeaway',
    'Turnaround Bank shot': 'Fadeaway',
}


MODIFIERS = ['driving', 'running', 'cutting', 'turnaround', 'fadeaway',
             'stepback', 'pullup', 'fingerroll', 'reverse', 'alleyoop',
             'putback', 'tip', 'bank', 'floating']

def parse_action_type(s):
    t = s.lower()
    if 'dunk' in t:    base = 'Dunk'
    elif 'layup' in t: base = 'Layup'
    elif 'hook' in t:  base = 'Hook Shot'
    elif 'float' in t: base = 'Floater'
    else:              base = 'Jump Shot'
    return pd.Series({
        'BASE_TYPE':  base,
        'driving':    int('driving' in t),
        'running':    int('running' in t),
        'cutting':    int('cutting' in t),
        'turnaround': int('turnaround' in t),
        'fadeaway':   int('fadeaway' in t),
        'stepback':   int('step back' in t),
        'pullup':     int('pullup' in t or 'pull-up' in t),
        'fingerroll': int('finger roll' in t),
        'reverse':    int('reverse' in t),
        'alleyoop':   int('alley oop' in t),
        'putback':    int('putback' in t),
        'tip':        int(t.startswith('tip ')),
        'bank':       int('bank' in t),
        'floating':   int('float' in t),
    })


def flag_analysis(df, min_n=500):
    parsed = df['ACTION_TYPE'].apply(parse_action_type)
    d = pd.concat([df[['ACTION_TYPE', 'SHOT_MADE_FLAG', 'SHOT_DISTANCE']], parsed], axis=1)

    print("=" * 78)
    print("1. ALL ACTION TYPES")
    print("=" * 78)
    full = (d.groupby('ACTION_TYPE')
             .agg(fg=('SHOT_MADE_FLAG', 'mean'),
                  n=('SHOT_MADE_FLAG', 'size'),
                  dist=('SHOT_DISTANCE', 'mean'))
             .sort_values('n', ascending=False).round(3))
    print(full.to_string())

    print("\n" + "=" * 78)
    print("2. BASE TYPE ONLY")
    print("=" * 78)
    base = (d.groupby('BASE_TYPE')
             .agg(fg=('SHOT_MADE_FLAG', 'mean'),
                  n=('SHOT_MADE_FLAG', 'size'),
                  dist=('SHOT_DISTANCE', 'mean'))
             .sort_values('fg', ascending=False).round(3))
    print(base.to_string())

    print("\n" + "=" * 78)
    print("3. MODIFIER EFFECT OVERALL")
    print("=" * 78)
    rows = []
    #checks if shot modifiers impact fg% regardless of shot distance
    for m in MODIFIERS:
        on, off = d[d[m] == 1], d[d[m] == 0]
        if len(on) < min_n:
            continue
        rows.append({
            'modifier': m, 'n': len(on),
            'fg_with': on['SHOT_MADE_FLAG'].mean(),
            'fg_without': off['SHOT_MADE_FLAG'].mean(),
            'delta': on['SHOT_MADE_FLAG'].mean() - off['SHOT_MADE_FLAG'].mean(),
            'dist_with': on['SHOT_DISTANCE'].mean(),
            'dist_without': off['SHOT_DISTANCE'].mean(),
        })
    overall = pd.DataFrame(rows).sort_values('delta', ascending=False).round(3)
    print(overall.to_string(index=False))

    print("\n" + "=" * 78)
    print(f"4. MODIFIER EFFECT WITHIN BASE TYPE (min n={min_n})")
    print("=" * 78)
    rows = []
    for bt, g in d.groupby('BASE_TYPE'):
        for m in MODIFIERS:
            on, off = g[g[m] == 1], g[g[m] == 0]
            if len(on) < min_n or len(off) < min_n:
                continue
            rows.append({
                'base': bt, 'modifier': m, 'n': len(on),
                'fg_with': on['SHOT_MADE_FLAG'].mean(),
                'fg_without': off['SHOT_MADE_FLAG'].mean(),
                'delta': on['SHOT_MADE_FLAG'].mean() - off['SHOT_MADE_FLAG'].mean(),
                'd_dist': on['SHOT_DISTANCE'].mean() - off['SHOT_DISTANCE'].mean(),
            })
    within = pd.DataFrame(rows).sort_values(['modifier', 'base']).round(3)
    print(within.to_string(index=False))

    print("\n" + "=" * 78)
    print("5. CONSISTENCY: does each modifier point the same way in every base type?")
    print("=" * 78)
    cons = (within.groupby('modifier')
                  .agg(n_bases=('base', 'size'),
                       mean_delta=('delta', 'mean'),
                       min_delta=('delta', 'min'),
                       max_delta=('delta', 'max'))
                  .round(3))
    cons['consistent'] = np.sign(cons['min_delta']) == np.sign(cons['max_delta'])
    print(cons.sort_values('mean_delta', ascending=False).to_string())

    return full, base, overall, within, cons

def compute_roster(df):
    FIVE_ZONES = ['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range',
                  'Above the Break 3', 'Corner 3']
    df = df.dropna(subset=['PLAYER_ID', 'SHOT_ZONE_BASIC'])
    counts = df.groupby(['PLAYER_ID', 'SHOT_ZONE_BASIC']).size().unstack('SHOT_ZONE_BASIC').fillna(0)
    qualifies = (counts >= 400).sum(axis=1) >= 2 #keep players according to 400/2 rule
    roster_ids = counts.index[qualifies]
    id_to_cat = {pid: i + 1 for i, pid in enumerate(roster_ids)} #recompute player id-s to smaller numbers

    names = (df.drop_duplicates('PLAYER_ID')
             .set_index('PLAYER_ID')['PLAYER_NAME'].to_dict())

    selectable = {}
    for pid in roster_ids:
        zones = [z for z in FIVE_ZONES if counts.loc[pid, z] >= 200]
        selectable[str(id_to_cat[pid])] = {
            "name": names.get(pid, str(int(pid))),
            "zones": zones,
        }

    out = Path(SHARED_DIR / "player_zones.json")
    out.write_text(json.dumps(selectable, indent=2))

    return id_to_cat

def data_cleanup_playerMode(df, mode, id_to_cat):
    df = df[['PLAYER_ID', 'PLAYER_NAME', 'SHOT_ZONE_BASIC' ,'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']].copy()
    df = df[df['LOC_Y'] <470] #remove shots from beyond the half court
    df = df[df['SHOT_ZONE_BASIC'] != 'Backcourt']
    df['SHOT_ZONE_BASIC'] = df['SHOT_ZONE_BASIC'].replace(
        {'Left Corner 3': 'Corner 3', 'Right Corner 3': 'Corner 3'}
    )
    df = df.dropna()
    df['PLAYER_CAT'] = df['PLAYER_ID'].map(id_to_cat).fillna(0).astype(int)
    df.to_parquet(DATA_DIR / f"{mode}_playerMode.parquet")

def data_cleanup(df, mode):
    df = df[['ACTION_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_MADE_FLAG']].copy()
    df['SHOT_CATEGORY'] = df['ACTION_TYPE'].map(CATEGORY_MAP)
    at = df['ACTION_TYPE'].str.lower()
    df['IS_MOVING']    = (at.str.contains('running') | at.str.contains('cutting')).astype(int)
    df = df.drop(['ACTION_TYPE'], axis=1)
    df = df[df['LOC_Y'] < 470]
    df = df.dropna()
    df.to_parquet(DATA_DIR / f"{mode}_cleanData.parquet")


if __name__ == '__main__':
    df_train = pd.read_parquet(DATA_DIR / "train_allShots.parquet")
    df_val = pd.read_parquet(DATA_DIR / "val_allShots.parquet")
    df_test = pd.read_parquet(DATA_DIR / "test_allShots.parquet")
    ##flag_analysis(df_train, min_n=500)
    # data_cleanup(df_train, 'train')
    # data_cleanup(df_val, 'val')
    # data_cleanup(df_test, 'test')
    train_clean = df_train[df_train['LOC_Y'] < 470].copy()
    train_clean = train_clean[train_clean['SHOT_ZONE_BASIC'] != 'Backcourt']
    train_clean['SHOT_ZONE_BASIC'] = train_clean['SHOT_ZONE_BASIC'].replace(
        {'Left Corner 3': 'Corner 3', 'Right Corner 3': 'Corner 3'}
    )
    id_to_cat = compute_roster(train_clean)
    data_cleanup_playerMode(df_train, 'train', id_to_cat)
    data_cleanup_playerMode(df_val, 'val', id_to_cat)
    data_cleanup_playerMode(df_test, 'test', id_to_cat)