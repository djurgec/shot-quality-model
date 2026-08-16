import json
import math
from pathlib import Path

import pandas as pd

from app.features import FEATURES, CATEGORICAL, FEATURES_PLAYERMODE, CATEGORICAL_PLAYERMODE, FEATURES_PLAYERMODE_AVG, CATEGORICAL_PLAYERMODE_AVG


from fastapi import HTTPException

MOVING_CATEGORIES = {'Dunk', 'Layup', 'Jump Shot'}
_BANDS_PATH = Path(__file__).parent.parent / "shared" / "zone_bands.json"
ZONE_BANDS = json.loads(_BANDS_PATH.read_text())


BASELINE_Y     = -52.5
HALF_COURT_Y   = 417.5
SIDELINE_X     = 250.0
RESTRICTED_R   = 4.0
PAINT_HALF_W   = 80.0
PAINT_TOP_Y    = 138.0
CORNER_3_X     = 220.0
CORNER_3_TOP_Y = 87.5
ARC_3_R        = 23.75
BACKCOURT_Y    = 420.0


def check_category_bounds(req, bounds, dist):
    """
    Rejects impossible distances and positions for specific shot types, e.g. a dunk cannot be taken from the three point line.
    """
    b = bounds.get(req.shot_category)
    if b is None:
        raise HTTPException(422, f"unknown shot category: {req.shot_category}")
    if not (b["dist_min"] <= dist <= b["dist_max"]):
        raise HTTPException(422, f"{req.shot_category} not realistic at {dist:.0f} ft")
    if not (b["x_min"] <= req.loc_x <= b["x_max"]):
        raise HTTPException(422, f"{req.shot_category} not realistic at this x position")
    if not (b["y_min"] <= req.loc_y <= b["y_max"]):
        raise HTTPException(422, f"{req.shot_category} not realistic at this y position")

def check_player_area(req, area, dist):
    """
    Rejects requests for spots on the court for which data is insufficient.
    """
    if req.loc_y < area['y_min']:
        raise HTTPException(422, "almost nobody shoots from that far behind the hoop")
    if abs(req.loc_x) > area['x_abs_max']:
        raise HTTPException(422, "almost nobody shoots from that close to the sideline")
    if dist > area['dist_max']:
        raise HTTPException(422, f"almost nobody shoots from {dist:.0f} ft")


def check_player_zone(player_code, zone, player_zones):
    """
    Checks if requested player has a minimum number of shots from the selected location.
    Returns the roster entry.
    """
    entry = player_zones.get(str(player_code))
    if entry is None:
        raise HTTPException(422, "unknown player")
    if zone not in entry["zones"]:
        raise HTTPException(422, f"{entry['name']} has insufficient shots from {zone}")
    return entry


def check_court_bounds(req):
    if req.loc_y < BASELINE_Y:
        raise HTTPException(422, "Selected spot is behind the baseline")
    if req.loc_y > HALF_COURT_Y or abs(req.loc_x) > SIDELINE_X:
        raise HTTPException(422, "Selected spot is not in the half court")


def classify_zone(loc_x, loc_y):
    """
    Classifies the selected location on the court in one of the main shot zones.
    """
    if loc_y < BASELINE_Y:
        return None

    r = math.hypot(loc_x, loc_y) / 10.0


    if loc_y <= CORNER_3_TOP_Y:
        if abs(loc_x) >= CORNER_3_X:
            return 'Left Corner 3' if loc_x < 0 else 'Right Corner 3'
    elif r >= ARC_3_R:
        return 'Backcourt' if loc_y >= BACKCOURT_Y else 'Above the Break 3'

    if r < RESTRICTED_R:
        return 'Restricted Area'
    if abs(loc_x) <= PAINT_HALF_W and loc_y <= PAINT_TOP_Y:
        return 'In The Paint (Non-RA)'
    return 'Mid-Range'


def classify_zone_player(loc_x, loc_y):
    """
    Classifies the selected location on the court in one of the main shot zones.
    Merges the Left/Right Corner 3 zones in one zone.
    """

    zone = classify_zone(loc_x, loc_y)
    return "Corner 3" if zone in ("Left Corner 3", "Right Corner 3") else zone

def get_shot_value(zone):
    if zone in ['Mid-Range', 'Restricted Area', 'In The Paint (Non-RA)']:
        return 2
    else:
        return 3

def build_x(req, zone, dist):
    is_moving = int(req.is_moving) if req.shot_category in MOVING_CATEGORIES else 0
    row = {
        'LOC_X': req.loc_x,
        'LOC_Y': req.loc_y,
        'SHOT_DISTANCE': dist,
        'IS_MOVING': is_moving,
        'SHOT_ZONE_BASIC': zone,
        'SHOT_CATEGORY': req.shot_category,
    }
    df = pd.DataFrame([row])[FEATURES]

    for c in CATEGORICAL:
        df[c] = df[c].astype('category')
    return df

def classify_quality(xpts):
    if xpts >= 1.30:
        return 'Elite'
    if xpts >= 1.10:
        return 'Good'
    if xpts >= 0.90:
        return 'Average'
    if xpts >= 0.75:
        return 'Below Average'
    return 'Poor'

def build_player_mode_x(req, zone, dist):
    row = {
        'LOC_X': req.loc_x,
        'LOC_Y': req.loc_y,
        'SHOT_DISTANCE': dist,
        'PLAYER_CAT': req.player_id,
        'SHOT_ZONE_BASIC': zone
    }
    df = pd.DataFrame([row])[FEATURES_PLAYERMODE]
    for c in CATEGORICAL_PLAYERMODE:
        df[c] = df[c].astype('category')
    return df

def build_player_mode_avg_x(req, zone, dist):
    row = {
        'LOC_X': req.loc_x,
        'LOC_Y': req.loc_y,
        'SHOT_DISTANCE': dist,
        'SHOT_ZONE_BASIC': zone
    }
    df = pd.DataFrame([row])[FEATURES_PLAYERMODE_AVG]
    for c in CATEGORICAL_PLAYERMODE_AVG:
        df[c] = df[c].astype('category')
    return df

def return_zone_descriptor(zone):
    if zone == 'Above the Break 3':
        return 'beyond the arc'
    elif zone == 'Corner 3':
        return 'in the corner'
    elif zone == 'Mid-Range':
        return 'in the mid-range'
    elif zone == 'Restricted Area':
        return 'at the rim'
    elif zone == 'In The Paint (Non-RA)':
        return 'in the paint'
    else:
        raise HTTPException(422, "unknown zone")

def shooter_band(delta, zone):
    """
    Classifies the selected player's shotmaking ability.
    """
    b = ZONE_BANDS.get(zone)
    if b is None:
        return "average" #fallback
    if delta > b["elite"]:
        return "elite"
    if delta > b["above"]:
        return "above average"
    if delta > b["below"]:
        return "average"
    if delta > b["poor"]:
        return "below average"
    return "poor"

def shot_distance_from_loc(loc_x, loc_y):
    """NBA stats API calculates SHOT_DISTANCE as the truncated foot distance from the hoop."""
    return math.floor(math.hypot(loc_x, loc_y) / 10.0)





