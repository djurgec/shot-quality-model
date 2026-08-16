from contextlib import asynccontextmanager
from pathlib import Path
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from xgboost import XGBClassifier
import app.utils as utils

ROOT = Path(__file__).parent.parent
SHOT_MODEL_PATH  = ROOT / "models" / "shot_mode.ubj"
PLAYER_MODEL_PATH = ROOT / "models" / "player_mode.ubj"
PLAYER_AVG_MODEL_PATH = ROOT / "models" / "player_mode_avg.ubj"
BOUNDS_PATH = ROOT / "shared" / "bounds.json"
PLAYER_BOUNDS_PATH = ROOT / "shared" / "player_bounds.json"
ROSTER_PATH = ROOT / "shared" / "player_zones.json"
FRONTEND_DIR = ROOT / "frontend"
SHARED_DIR = ROOT / "shared"
ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    shot_model = XGBClassifier()
    shot_model.load_model(str(SHOT_MODEL_PATH))
    player_model = XGBClassifier()
    player_model.load_model(str(PLAYER_MODEL_PATH))
    player_avg_model = XGBClassifier()
    player_avg_model.load_model(str(PLAYER_AVG_MODEL_PATH))
    ml["shot_model"] = shot_model
    ml["player_model"] = player_model
    ml["player_avg_model"] = player_avg_model
    ml["bounds"] = json.loads(BOUNDS_PATH.read_text())
    ml["player_bounds"] = json.loads(PLAYER_BOUNDS_PATH.read_text())
    ml["roster"] = json.loads(ROSTER_PATH.read_text())
    yield
    ml.clear()

app = FastAPI(title="NBA xPoints App", lifespan=lifespan)

class ShotRequest(BaseModel):
    loc_x: float
    loc_y: float
    is_moving: bool = False
    shot_category: str

class PlayerRequest(BaseModel):
    loc_x: float
    loc_y: float
    player_id: int


@app.post("/api/predict/shot")
def predict_points(request: ShotRequest):
    utils.check_court_bounds(request)
    dist = utils.shot_distance_from_loc(request.loc_x, request.loc_y)
    utils.check_category_bounds(request, ml["bounds"], dist)
    zone = utils.classify_zone(request.loc_x, request.loc_y)
    shot_value = utils.get_shot_value(zone)
    x = utils.build_x(request, zone, dist)
    xfg = round(float(ml["shot_model"].predict_proba(x)[:, 1][0]), 2)
    xpts = round(xfg * shot_value, 2)
    return {'xFg': xfg, 'xPts': xpts, 'quality': utils.classify_quality(xpts)}

@app.post("/api/predict/player")
def predict_player(request: PlayerRequest):
    utils.check_court_bounds(request)
    dist = utils.shot_distance_from_loc(request.loc_x, request.loc_y)
    utils.check_player_area(request, ml["player_bounds"], dist)
    zone = utils.classify_zone_player(request.loc_x, request.loc_y)
    entry = utils.check_player_zone(request.player_id, zone, ml["roster"])
    x = utils.build_player_mode_x(request, zone, dist)
    avg_x = utils.build_player_mode_avg_x(request, zone, dist)
    xfg = round(float(ml["player_model"].predict_proba(x)[:, 1][0]), 3)
    avg_xfg = round(float(ml["player_avg_model"].predict_proba(avg_x)[:, 1][0]), 3)
    delta = round(xfg - avg_xfg, 3)
    descriptor = utils.shooter_band(delta, zone)
    article = 'an' if descriptor in ('elite', 'above average', 'average') else 'a'
    message = f'{entry["name"]} is {article} {descriptor} shooter from this spot {utils.return_zone_descriptor(zone)}'
    return {'xFg': xfg, 'avg_xFg': avg_xfg, 'delta': delta, 'descriptor': descriptor, 'message': message}


# Static files, mounted after the routes above so the catch-all at "/" cannot shadow them.
app.mount("/shared", StaticFiles(directory=SHARED_DIR), name="shared")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
