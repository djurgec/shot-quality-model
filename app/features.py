NUMERIC = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'IS_MOVING']
CATEGORICAL = ['SHOT_ZONE_BASIC', 'SHOT_CATEGORY']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'SHOT_MADE_FLAG'

NUMERIC_PLAYERMODE_AVG = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
CATEGORICAL_PLAYERMODE_AVG = ['SHOT_ZONE_BASIC']
FEATURES_PLAYERMODE_AVG= NUMERIC_PLAYERMODE_AVG + CATEGORICAL_PLAYERMODE_AVG

NUMERIC_PLAYERMODE = ['LOC_X', 'LOC_Y', 'SHOT_DISTANCE']
CATEGORICAL_PLAYERMODE = ['PLAYER_CAT', 'SHOT_ZONE_BASIC']
FEATURES_PLAYERMODE= NUMERIC_PLAYERMODE + CATEGORICAL_PLAYERMODE


def split_xy(df):
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype('category')
    return df[FEATURES], df[TARGET]

def split_xy_playermode(df):
    df = df.copy()
    for c in CATEGORICAL_PLAYERMODE:
        df[c] = df[c].astype('category')
    return df[FEATURES_PLAYERMODE], df[TARGET]

def split_xy_playermode_avg(df):
    df = df.copy()
    for c in CATEGORICAL_PLAYERMODE_AVG:
        df[c] = df[c].astype('category')
    return df[FEATURES_PLAYERMODE_AVG], df[TARGET]