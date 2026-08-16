'use strict';

const VB_W = 500, VB_H = 470;
const BASKET = { x: 250, y: 417.5 };
const FT = 10;

const locToSvg = (x, y) => ({ x: x + 250, y: 417.5 - y });
const svgToLoc = (sx, sy) => ({ x: sx - 250, y: 417.5 - sy });

const SHOT_API_URL = '/api/predict/shot';
const PLAYER_API_URL = '/api/predict/player';
const SVG_NS = 'http://www.w3.org/2000/svg';

const MODE_SHOT = 'shot', MODE_PLAYER = 'player';

const MODE_COPY = {
  [MODE_SHOT]:   'Expected points for an average NBA shooter, by shot type and spot on the floor.',
  [MODE_PLAYER]: 'How much better or worse a player shoots than an average player, spot by spot.',
};

const QUALITY_COLORS = {
  'elite':          '#0e9f6e',
  'good':           '#64a022',
  'average':        '#d59a00',
  'below average':  '#e06f1c',
  'poor':           '#d23b3b',
};
const QUALITY_UNKNOWN = '#6b7c93';
const qualityColor = (label) =>
  QUALITY_COLORS[String(label || '').trim().toLowerCase()] || QUALITY_UNKNOWN;

const QUALITY_ORDER = ['Elite', 'Good', 'Average', 'Below Average', 'Poor'];

// only these show the "Situation" control
const MOVING_CATEGORIES = new Set(['Dunk', 'Layup', 'Jump Shot']);

const PLAYER_DESCRIPTOR = {
  'elite':          { qualityKey: 'elite',         label: 'Elite' },
  'above average':  { qualityKey: 'good',           label: 'Above Average' },
  'average':        { qualityKey: 'average',        label: 'Average' },
  'below average':  { qualityKey: 'below average',  label: 'Below Average' },
  'poor':            { qualityKey: 'poor',           label: 'Poor' },
};

let appMode = MODE_SHOT;
let bounds = null;            // parsed bounds.json (category -> box)
let playerBounds = null;      // parsed player_bounds.json (zone -> outer-edge box)
let playerZones = null;       // parsed player_zones.json (player_id -> {name, zones})
let selectedCategory = null;  // shot mode: currently chosen shot type, or null
let selectedPlayer = null;    // player mode: {id, name}, or null
let lastShot = null;          // last valid click: {loc_x, loc_y, dist, svgX, svgY}

/* ---- DOM handles -------------------------------------------------------- */
const svg          = document.getElementById('court');
const modeButtons  = [...document.querySelectorAll('.mode-btn')];
const shotTypeBlock = document.getElementById('shotTypeBlock');
const categoryBox  = document.getElementById('categoryButtons');
const playerBlock  = document.getElementById('playerBlock');
const playerSelectEl = document.getElementById('playerSelect');
const brandTagline = document.getElementById('brandTagline');
const courtHint    = document.getElementById('courtHint');
const coordReadout = document.getElementById('coordReadout');
const toggleMoving = document.getElementById('toggleMoving');
const situationBlock = document.getElementById('situationBlock');
const banner       = document.getElementById('banner');

const resultEmpty     = document.getElementById('resultEmpty');
const resultEmptyText = document.getElementById('resultEmptyText');
const resultBodyShot   = document.getElementById('resultBodyShot');
const resultBodyPlayer = document.getElementById('resultBodyPlayer');
const resultError     = document.getElementById('resultError');
const resultErrorTitle= document.getElementById('resultErrorTitle');
const resultErrorMsg  = document.getElementById('resultErrorMsg');
const qualityBadge    = document.getElementById('qualityBadge');
const xfgValue        = document.getElementById('xfgValue');
const xptsValue       = document.getElementById('xptsValue');
const shotMeta        = document.getElementById('shotMeta');
const playerBadge     = document.getElementById('playerBadge');
const deltaValue      = document.getElementById('deltaValue');
const playerXfgValue  = document.getElementById('playerXfgValue');
const avgXfgValue     = document.getElementById('avgXfgValue');
const playerMessage   = document.getElementById('playerMessage');
const qualityLegend   = document.getElementById('qualityLegend');
const scorecard       = document.getElementById('scorecard');
const xfgRing         = document.getElementById('xfgRing');
const deltaFill       = document.getElementById('deltaFill');

const RING_LENGTH = 2 * Math.PI * 42;   // the ring's r in index.html
const DELTA_SCALE = 10;                 // percentage points at each end of the delta bar

let greyOverlay, boundRect, maxCircle, minCircle, marker;
let playerZoneShapes = {};
let areaBoxRect, areaDiscCircle;

function el(tag, attrs) {
  const n = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

function arcPath(cx, cy, r, a0, a1, steps = 96) {
  let d = '';
  for (let i = 0; i <= steps; i++) {
    const a = a0 + (a1 - a0) * (i / steps);
    const p = locToSvg(cx + r * Math.cos(a), cy + r * Math.sin(a));
    d += (i === 0 ? 'M' : 'L') + p.x.toFixed(2) + ' ' + p.y.toFixed(2) + ' ';
  }
  return d.trim();
}

function circlePath(cx, cy, r) {
  return arcPath(cx, cy, r, 0, 2 * Math.PI) + ' Z';
}

// Court geometry
const BASELINE_Y     = -52.5;   // court edge behind the hoop
const HALF_COURT_Y   = 417.5;
const SIDELINE_X     = 250.0;
const RESTRICTED_R   = 4.0;     // ft
const PAINT_HALF_W   = 80.0;
const PAINT_TOP_Y    = 138.0;   // free-throw line
const CORNER_3_X     = 220.0;   // straight 22 ft corner lines
const CORNER_3_TOP_Y = 87.5;    // where the corner line gives way to the arc
const ARC_3_R        = 23.75;   // ft
const BACKCOURT_Y    = 420.0;


// Mirrors classify_zone() + classify_zone_player() in app/utils.py.
function classifyZonePlayer(loc_x, loc_y) {
  if (loc_y < BASELINE_Y) return null;

  const r = Math.hypot(loc_x, loc_y) / 10;     // tenths of a foot -> feet

  if (loc_y <= CORNER_3_TOP_Y) {
    if (Math.abs(loc_x) >= CORNER_3_X) return 'Corner 3';
  } else if (r >= ARC_3_R) {
    return loc_y >= BACKCOURT_Y ? 'Backcourt' : 'Above the Break 3';
  }

  if (r < RESTRICTED_R) return 'Restricted Area';
  if (Math.abs(loc_x) <= PAINT_HALF_W && loc_y <= PAINT_TOP_Y) return 'In The Paint (Non-RA)';
  return 'Mid-Range';
}

// Build the court (markings + bounds-mask + marker) inside the <svg>.
function buildCourt() {
  const defs = el('defs', {});

  const clip = el('clipPath', { id: 'boundClip' });
  boundRect = el('rect', { x: 0, y: 0, width: 0, height: 0 });
  clip.appendChild(boundRect);
  defs.appendChild(clip);

  const mask = el('mask', { id: 'invalidMask' });
  mask.appendChild(el('rect', { x: 0, y: 0, width: VB_W, height: VB_H, fill: '#fff' }));
  const maskGroup = el('g', { 'clip-path': 'url(#boundClip)' });
  maxCircle = el('circle', { cx: BASKET.x, cy: BASKET.y, r: 0, fill: '#000' });
  minCircle = el('circle', { cx: BASKET.x, cy: BASKET.y, r: 0, fill: '#fff' });
  maskGroup.appendChild(maxCircle);
  maskGroup.appendChild(minCircle);
  mask.appendChild(maskGroup);
  defs.appendChild(mask);

  buildPlayerMask(defs);

  svg.appendChild(defs);

  buildHardwood(defs);
  svg.appendChild(el('rect', { x: 0, y: 0, width: VB_W, height: VB_H, fill: 'url(#woodBase)' }));
  svg.appendChild(el('g', { class: 'plank-seams', 'pointer-events': 'none' })).append(
    ...plankSeams());
  svg.appendChild(el('rect', {
    x: 0, y: 0, width: VB_W, height: VB_H,
    filter: 'url(#woodGrain)', opacity: .5, 'pointer-events': 'none',
  }));

  const g = el('g', { class: 'court-lines', 'pointer-events': 'none' });

  g.appendChild(el('rect', { class: 'ct-boundary', x: 1.5, y: 1.5, width: VB_W - 3, height: VB_H - 3 }));

  const paintTL = locToSvg(-8 * FT, 13.75 * FT);   // top-left in svg
  g.appendChild(el('rect', {
    class: 'ct-paint',
    x: paintTL.x, y: paintTL.y, width: 16 * FT, height: (13.75 + 5.25) * FT,
  }));

  const ftc = locToSvg(0, 13.75 * FT);
  g.appendChild(el('circle', { cx: ftc.x, cy: ftc.y, r: 6 * FT }));

  // Three-point line. Corners are straight at x = ±22 ft; the arc is 23.75 ft.
  // The straight corner meets the arc where sqrt(23.75^2 - 22^2) = 8.948 ft.
  const cornerX = 22 * FT, arcR = 23.75 * FT;
  const junctionY = Math.sqrt(arcR * arcR - cornerX * cornerX);
  const baseY = -5.25 * FT;
  for (const sgn of [-1, 1]) {
    const a = locToSvg(sgn * cornerX, baseY);
    const b = locToSvg(sgn * cornerX, junctionY);
    g.appendChild(el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y }));
  }
  const aR = Math.atan2(junctionY, cornerX);          // right junction angle
  const aL = Math.atan2(junctionY, -cornerX);          // left junction angle
  g.appendChild(el('path', { d: arcPath(0, 0, arcR, aR, aL) }));

  // Restricted-area arc: 4 ft radius semicircle around the hoop, with short
  // uprights back to the backboard plane (loc_y -1.25 ft).
  g.appendChild(el('path', { d: arcPath(0, 0, 4 * FT, 0, Math.PI) }));
  for (const sgn of [-1, 1]) {
    const a = locToSvg(sgn * 4 * FT, 0);
    const b = locToSvg(sgn * 4 * FT, -1.25 * FT);
    g.appendChild(el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y }));
  }

  // Backboard (6 ft wide at loc_y -1.25 ft) and hoop (0.75 ft radius at origin).
  const bbL = locToSvg(-3 * FT, -1.25 * FT), bbR = locToSvg(3 * FT, -1.25 * FT);
  g.appendChild(el('line', { class: 'ct-backboard', x1: bbL.x, y1: bbL.y, x2: bbR.x, y2: bbR.y }));
  const hoop = locToSvg(0, 0);
  g.appendChild(el('circle', { class: 'ct-hoop', cx: hoop.x, cy: hoop.y, r: 0.75 * FT }));
  // Rim neck: backboard -> top of hoop.
  const neckT = locToSvg(0, -1.25 * FT), neckB = locToSvg(0, 0.75 * FT);
  g.appendChild(el('line', { class: 'ct-hoop', x1: neckT.x, y1: neckT.y, x2: neckB.x, y2: neckB.y }));

  g.appendChild(el('path', { d: arcPath(0, 417.5, 6 * FT, Math.PI, 2 * Math.PI) }));

  svg.appendChild(g);

  // --- grey overlay for invalid regions (masked; never intercepts clicks) ---
  greyOverlay = el('rect', {
    class: 'invalid-overlay',
    x: 0, y: 0, width: VB_W, height: VB_H,
    mask: 'url(#invalidMask)', 'pointer-events': 'none',
  });
  svg.appendChild(greyOverlay);

  marker = el('circle', { class: 'shot-marker', cx: 0, cy: 0, r: 8,
                          fill: QUALITY_UNKNOWN, 'pointer-events': 'none',
                          visibility: 'hidden' });
  svg.appendChild(marker);
}

function buildHardwood(defs) {
  const grad = el('linearGradient', { id: 'woodBase', x1: 0, y1: 0, x2: 0, y2: 1 });
  grad.appendChild(el('stop', { class: 'w1', offset: 0 }));
  grad.appendChild(el('stop', { class: 'w2', offset: .55 }));
  grad.appendChild(el('stop', { class: 'w3', offset: 1 }));
  defs.appendChild(grad);

  const filter = el('filter', {
    id: 'woodGrain', x: 0, y: 0, width: '100%', height: '100%',
    'color-interpolation-filters': 'sRGB',
  });
  filter.appendChild(el('feTurbulence', {
    type: 'fractalNoise', baseFrequency: '0.85 0.012', numOctaves: 4, seed: 7, result: 'n',
  }));
  filter.appendChild(el('feColorMatrix', {
    in: 'n', type: 'matrix',
    values: '0 0 0 0 0.36  0 0 0 0 0.22  0 0 0 0 0.09  0 0 0 0.55 0',
  }));
  defs.appendChild(filter);
}

function plankSeams() {
  const seams = [];
  for (let x = 24; x < VB_W; x += 24) {
    seams.push(el('line', { class: 'plank-seam', x1: x, y1: 0, x2: x, y2: VB_H }));
  }
  return seams;
}

function buildPlayerMask(defs) {
  const clipBox = el('clipPath', { id: 'clipPlayerBox' });
  areaBoxRect = el('rect', { x: 0, y: 0, width: VB_W, height: VB_H });
  clipBox.appendChild(areaBoxRect);
  defs.appendChild(clipBox);

  const clipDisc = el('clipPath', { id: 'clipPlayerDisc' });
  areaDiscCircle = el('circle', { cx: BASKET.x, cy: BASKET.y, r: 0 });
  clipDisc.appendChild(areaDiscCircle);
  defs.appendChild(clipDisc);

  const clipArc = el('clipPath', { id: 'clipArc3' });
  clipArc.appendChild(el('rect', {
    x: 0, y: 0, width: VB_W, height: locToSvg(0, CORNER_3_TOP_Y).y }));
  defs.appendChild(clipArc);

  const mask = el('mask', { id: 'playerInvalidMask' });

  mask.appendChild(el('rect', { x: 0, y: 0, width: VB_W, height: VB_H, fill: '#fff' }));


  const box = el('g', { 'clip-path': 'url(#clipPlayerBox)' });
  const area = el('g', { 'clip-path': 'url(#clipPlayerDisc)' });
  box.appendChild(area);
  mask.appendChild(box);

  const midRange = el('rect', { x: 0, y: 0, width: VB_W, height: VB_H, fill: '#fff' });
  area.appendChild(midRange);

  const arc3 = el('path', {
    d: `M0 0 H${VB_W} V${VB_H} H0 Z ` + circlePath(0, 0, ARC_3_R * FT),
    'fill-rule': 'evenodd', 'clip-path': 'url(#clipArc3)', fill: '#fff',
  });
  area.appendChild(arc3);

  const c3Y = locToSvg(0, CORNER_3_TOP_Y).y;
  const c3H = CORNER_3_TOP_Y - BASELINE_Y;
  const c3W = SIDELINE_X - CORNER_3_X;
  const corner3R = el('rect', { x: locToSvg(CORNER_3_X, 0).x, y: c3Y, width: c3W, height: c3H, fill: '#fff' });
  const corner3L = el('rect', { x: locToSvg(-SIDELINE_X, 0).x, y: c3Y, width: c3W, height: c3H, fill: '#fff' });
  area.appendChild(corner3R);
  area.appendChild(corner3L);

  const paintTL = locToSvg(-PAINT_HALF_W, PAINT_TOP_Y);
  const paint = el('rect', {
    x: paintTL.x, y: paintTL.y,
    width: 2 * PAINT_HALF_W, height: PAINT_TOP_Y - BASELINE_Y, fill: '#fff',
  });
  area.appendChild(paint);

  const restrictedArea = el('circle', {
    cx: BASKET.x, cy: BASKET.y, r: RESTRICTED_R * FT, fill: '#fff' });
  area.appendChild(restrictedArea);

  defs.appendChild(mask);

  playerZoneShapes = {
    'Mid-Range': [midRange],
    'Restricted Area': [restrictedArea],
    'In The Paint (Non-RA)': [paint],
    'Above the Break 3': [arc3],
    'Corner 3': [corner3R, corner3L],
  };
}

function applyPlayerBounds() {
  areaBoxRect.setAttribute('x', locToSvg(-playerBounds.x_abs_max, 0).x);
  areaBoxRect.setAttribute('width', 2 * playerBounds.x_abs_max);
  areaBoxRect.setAttribute('y', 0);
  areaBoxRect.setAttribute('height', locToSvg(0, playerBounds.y_min).y);
  areaDiscCircle.setAttribute('r', playerBounds.dist_max * FT);
}

function updatePlayerMask(playerId) {
  const entry = playerId != null ? playerZones[String(playerId)] : null;
  const allowed = entry ? new Set(entry.zones) : new Set();
  for (const [zone, shapes] of Object.entries(playerZoneShapes)) {
    const fill = allowed.has(zone) ? '#000' : '#fff';
    for (const shape of shapes) shape.setAttribute('fill', fill);
  }
}


// A click is valid only if it satisfies the category's box AND distance band.
function isValid(cat, loc_x, loc_y, dist) {
  const b = bounds[cat];
  if (!b) return false;
  return dist >= b.dist_min && dist <= b.dist_max &&
         loc_x >= b.x_min && loc_x <= b.x_max &&
         loc_y >= b.y_min && loc_y <= b.y_max;
}

// Point the mask at the selected category's box; null clears it
function updateMask(cat) {
  if (!cat) {
    boundRect.setAttribute('width', 0);
    boundRect.setAttribute('height', 0);
    maxCircle.setAttribute('r', 0);
    minCircle.setAttribute('r', 0);
    return;
  }
  const b = bounds[cat];
  const tl = locToSvg(b.x_min, b.y_max);
  boundRect.setAttribute('x', tl.x);
  boundRect.setAttribute('y', tl.y);
  boundRect.setAttribute('width',  b.x_max - b.x_min);
  boundRect.setAttribute('height', b.y_max - b.y_min);
  maxCircle.setAttribute('r', b.dist_max * FT);
  minCircle.setAttribute('r', b.dist_min * FT);
}


// Map a pointer event to NBA loc + distance, via the SVG's own screen matrix.
function eventToShot(evt) {
  const ctm = svg.getScreenCTM().inverse();
  const p = new DOMPoint(evt.clientX, evt.clientY).matrixTransform(ctm);
  const loc = svgToLoc(p.x, p.y);
  const dist = Math.sqrt(loc.x * loc.x + loc.y * loc.y) / 10;   // tenths -> feet
  return { svgX: p.x, svgY: p.y, loc_x: loc.x, loc_y: loc.y, dist };
}

function hasSelection() {
  return appMode === MODE_SHOT ? !!selectedCategory : !!selectedPlayer;
}

function inPlayerArea(loc_x, loc_y, dist) {
  return loc_y >= playerBounds.y_min
      && Math.abs(loc_x) <= playerBounds.x_abs_max
      && dist <= playerBounds.dist_max;
}

function isPointValid(s) {
  if (appMode === MODE_SHOT) {
    return isValid(selectedCategory, s.loc_x, s.loc_y, s.dist);
  }
  if (!inPlayerArea(s.loc_x, s.loc_y, s.dist)) return false;
  const entry = playerZones[String(selectedPlayer.id)];
  return entry.zones.includes(classifyZonePlayer(s.loc_x, s.loc_y));
}

function onCourtMove(evt) {
  if (!hasSelection()) { svg.style.cursor = 'not-allowed'; return; }
  const s = eventToShot(evt);
  const ok = isPointValid(s);
  svg.style.cursor = ok ? 'crosshair' : 'not-allowed';
  coordReadout.classList.add('show');
  coordReadout.textContent =
    `x ${showInt(s.loc_x)}  y ${showInt(s.loc_y)}  ·  ${s.dist.toFixed(1)} ft` +
    (ok ? '' : '  ·  outside range');
}

function onCourtLeave() { coordReadout.classList.remove('show'); }

function onCourtClick(evt) {
  if (!hasSelection()) return;
  const s = eventToShot(evt);
  if (!isPointValid(s)) return;

  lastShot = s;
  placeMarker(s.svgX, s.svgY);
  if (appMode === MODE_SHOT) requestPrediction();
  else requestPlayerPrediction();
}

function placeMarker(sx, sy) {
  marker.setAttribute('cx', sx);
  marker.setAttribute('cy', sy);
  marker.setAttribute('visibility', 'visible');
}


async function requestPrediction() {
  if (!lastShot || !selectedCategory) return;
  showLoading();

  const payload = {
    loc_x: round(lastShot.loc_x, 2),
    loc_y: round(lastShot.loc_y, 2),
    is_moving: MOVING_CATEGORIES.has(selectedCategory) && toggleMoving.checked,
    shot_category: selectedCategory,
  };

  try {
    const data = await predict(SHOT_API_URL, payload);
    showShotResult(data);
  } catch (err) {
    showError(err);
  }
}

async function requestPlayerPrediction() {
  if (!lastShot || !selectedPlayer) return;
  showLoading();

  const payload = {
    loc_x: round(lastShot.loc_x, 2),
    loc_y: round(lastShot.loc_y, 2),
    player_id: selectedPlayer.id,
  };

  try {
    const data = await predict(PLAYER_API_URL, payload);
    showPlayerResult(data);
  } catch (err) {
    showError(err);
  }
}

async function predict(url, payload) {
  const slowHint = setTimeout(
    () => showBadge(activeBadge(), 'Waking the server…', true), 4000);

  let resp;
  try {
    resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  } catch (_) {
    throw { kind: 'down', message:
      'Can’t reach the model API. If this is the hosted demo the server may still be ' +
      'starting — give it a moment and click again. Running it locally, check that the ' +
      'container is up (docker compose up).' };
  } finally {
    clearTimeout(slowHint);
  }

  if (resp.status === 422) {
    let detail = 'That shot isn’t valid.';
    try {
      const j = await resp.json();
      if (typeof j.detail === 'string') detail = j.detail;
      else if (Array.isArray(j.detail) && j.detail.length) detail = j.detail.map(d => d.msg).join('; ');
    } catch (_) { /* keep default */ }
    throw { kind: 'invalid', message: detail };
  }

  if (!resp.ok) {
    throw { kind: 'server', message: `The API returned an error (HTTP ${resp.status}).` };
  }
  return resp.json();
}

function activeResultBody() {
  return appMode === MODE_SHOT ? resultBodyShot : resultBodyPlayer;
}
function activeBadge() {
  return appMode === MODE_SHOT ? qualityBadge : playerBadge;
}

function showBadge(badge, text, loading) {
  for (const b of [qualityBadge, playerBadge]) b.classList.add('hidden');
  badge.classList.remove('hidden');
  badge.classList.toggle('loading', !!loading);
  badge.textContent = text;
}

function showLoading() {
  resultEmpty.classList.add('hidden');
  resultError.classList.add('hidden');
  resultBodyShot.classList.add('hidden');
  resultBodyPlayer.classList.add('hidden');
  activeResultBody().classList.remove('hidden');

  showBadge(activeBadge(), 'Calculating…', true);

  if (appMode === MODE_SHOT) {
    xfgValue.textContent = '—';
    xptsValue.textContent = '—';
    setRing(0);
  } else {
    deltaValue.textContent = '—';
    playerXfgValue.textContent = '—';
    avgXfgValue.textContent = '—';
    setDeltaBar(0);
  }
  setResultAccent(QUALITY_UNKNOWN);
}

function setRing(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  xfgRing.style.strokeDashoffset = RING_LENGTH * (1 - clamped / 100);
}

function setDeltaBar(deltaPct) {
  const mag = Math.min(Math.abs(deltaPct), DELTA_SCALE) / DELTA_SCALE * 50;
  if (deltaPct >= 0) {
    deltaFill.style.left = '50%';
    deltaFill.style.right = 'auto';
  } else {
    deltaFill.style.right = '50%';
    deltaFill.style.left = 'auto';
  }
  deltaFill.style.width = mag + '%';
}

function showShotResult(data) {
  const color = qualityColor(data.quality);
  const pct = data.xFg <= 1.0001 ? data.xFg * 100 : data.xFg;

  resultEmpty.classList.add('hidden');
  resultError.classList.add('hidden');
  resultBodyPlayer.classList.add('hidden');
  resultBodyShot.classList.remove('hidden');

  showBadge(qualityBadge, data.quality, false);
  xfgValue.textContent = pct.toFixed(1) + '%';
  xptsValue.textContent = Number(data.xPts).toFixed(2);
  setResultAccent(color);
  setRing(pct);

  const moveTxt = MOVING_CATEGORIES.has(selectedCategory)
    ? (toggleMoving.checked ? ' · moving' : ' · set')
    : '';
  shotMeta.textContent =
    `${selectedCategory} · ${lastShot.dist.toFixed(1)} ft${moveTxt}` +
    `  (x ${showInt(lastShot.loc_x)}, y ${showInt(lastShot.loc_y)})`;

  setResultAccent(color);
  marker.setAttribute('fill', color);
}

function showPlayerResult(data) {
  const info = PLAYER_DESCRIPTOR[data.descriptor];
  const color = info ? qualityColor(info.qualityKey) : QUALITY_UNKNOWN;

  resultEmpty.classList.add('hidden');
  resultError.classList.add('hidden');
  resultBodyShot.classList.add('hidden');
  resultBodyPlayer.classList.remove('hidden');

  showBadge(playerBadge, info ? info.label : data.descriptor, false);

  const deltaPct = data.delta * 100;
  const sign = deltaPct > 0 ? '+' : deltaPct < 0 ? '−' : '';
  deltaValue.textContent = sign + Math.abs(deltaPct).toFixed(1) + '%';

  playerXfgValue.textContent = (data.xFg * 100).toFixed(1) + '%';
  avgXfgValue.textContent = (data.avg_xFg * 100).toFixed(1) + '%';
  playerMessage.textContent = data.message;

  setResultAccent(color);
  setDeltaBar(deltaPct);
  marker.setAttribute('fill', color);
}

function showError(err) {
  const invalid = err && err.kind === 'invalid';
  resultEmpty.classList.add('hidden');
  resultBodyShot.classList.add('hidden');
  resultBodyPlayer.classList.add('hidden');
  resultError.classList.remove('hidden');
  resultErrorTitle.textContent =
    err && err.kind === 'down' ? 'Backend unavailable' :
    invalid ? 'Not a valid shot' : 'Prediction failed';
  resultErrorMsg.textContent = (err && err.message) || 'Unexpected error.';
  for (const b of [qualityBadge, playerBadge]) b.classList.add('hidden');
  setNeutralAccent();
  if (invalid) marker.setAttribute('visibility', 'hidden');
}

function setResultAccent(color) {
  scorecard.style.setProperty('--q', color);
  scorecard.style.setProperty('--q-tint', hexToRgba(color, .07));
}

function hexToRgba(hex, alpha) {
  const h = hex.replace('#', '');
  const n = parseInt(h.length === 3 ? h.replace(/./g, c => c + c) : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}


function buildCategoryButtons() {
  const cats = Object.keys(bounds).filter(k => !k.startsWith('_'));
  for (const cat of cats) {
    const btn = document.createElement('button');
    btn.className = 'cat-btn';
    btn.type = 'button';
    btn.textContent = cat;
    btn.addEventListener('click', () => selectCategory(cat, btn));
    categoryBox.appendChild(btn);
  }
}

function resetSelection(promptText) {
  lastShot = null;
  marker.setAttribute('visibility', 'hidden');
  resultBodyShot.classList.add('hidden');
  resultBodyPlayer.classList.add('hidden');
  resultError.classList.add('hidden');
  resultEmpty.classList.remove('hidden');
  resultEmptyText.textContent = promptText;
  for (const b of [qualityBadge, playerBadge]) b.classList.add('hidden');
  setNeutralAccent();
}

function setNeutralAccent() {
  scorecard.style.removeProperty('--q');
  scorecard.style.setProperty('--q-tint', 'transparent');
}

function selectCategory(cat, btn) {
  selectedCategory = cat;
  for (const b of categoryBox.children) b.classList.toggle('selected', b === btn);

  updateMask(cat);
  courtHint.classList.add('hidden');

  const showMoving = MOVING_CATEGORIES.has(cat);
  situationBlock.hidden = !showMoving;
  if (!showMoving) toggleMoving.checked = false;

  resetSelection(`Click the court to place a ${cat.toLowerCase()}.`);
}

function buildPlayerSelect() {
  const entries = Object.entries(playerZones).sort((a, b) => a[1].name.localeCompare(b[1].name));

  const placeholder = document.createElement('option');
  placeholder.value = '';
  placeholder.textContent = `Choose a player… (${entries.length})`;
  placeholder.disabled = true;
  placeholder.selected = true;
  playerSelectEl.appendChild(placeholder);

  for (const [id, info] of entries) {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = info.name;
    playerSelectEl.appendChild(opt);
  }
}

function onPlayerSelectChange() {
  const id = playerSelectEl.value;
  if (!id) {
    selectedPlayer = null;
    updatePlayerMask(null);
    resetSelection('Pick a player, then click the floor.');
    return;
  }
  selectedPlayer = { id: Number(id), name: playerZones[id].name };
  updatePlayerMask(id);
  courtHint.classList.add('hidden');
  resetSelection(`Click a spot on the court to see how ${selectedPlayer.name} shoots from there.`);
}

function setMode(mode) {
  if (mode === appMode) return;
  appMode = mode;
  const isShot = mode === MODE_SHOT;

  for (const b of modeButtons) {
    const selected = b.dataset.mode === mode;
    b.classList.toggle('selected', selected);
    b.setAttribute('aria-selected', String(selected));
  }
  brandTagline.textContent = MODE_COPY[mode];

  shotTypeBlock.hidden = !isShot;
  playerBlock.hidden = isShot;
  situationBlock.hidden = true;

  selectedCategory = null;
  selectedPlayer = null;
  for (const b of categoryBox.children) b.classList.remove('selected');
  playerSelectEl.value = '';
  toggleMoving.checked = false;

  greyOverlay.setAttribute('mask', isShot ? 'url(#invalidMask)' : 'url(#playerInvalidMask)');
  updateMask(null);
  updatePlayerMask(null);

  courtHint.classList.remove('hidden');
  courtHint.textContent = isShot ? 'Choose a shot type to begin' : 'Choose a player to begin';
  resetSelection(isShot ? 'Pick a shot type, then click the floor.' : 'Pick a player, then click the floor.');
}

function onToggleChange() {
  if (lastShot && selectedCategory) requestPrediction();
}

function buildLegend() {
  for (const label of QUALITY_ORDER) {
    const item = document.createElement('span');
    item.className = 'legend-item';
    const dot = document.createElement('span');
    dot.className = 'legend-dot';
    dot.style.background = qualityColor(label);
    item.appendChild(dot);
    item.appendChild(document.createTextNode(label));
    qualityLegend.appendChild(item);
  }
}

function showBanner(html) {
  banner.innerHTML = html;
  banner.classList.remove('hidden');
}

const round = (v, dp) => { const f = 10 ** dp; return Math.round(v * f) / f; };
const showInt = (v) => String(Math.round(v));

async function init() {
  buildCourt();
  buildLegend();

  const missing = [];
  try {
    const resp = await fetch('/shared/bounds.json', { cache: 'no-store' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    bounds = await resp.json();
  } catch (_) { missing.push('bounds.json'); }

  try {
    const resp = await fetch('/shared/player_zones.json', { cache: 'no-store' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    playerZones = await resp.json();
  } catch (_) { missing.push('player_zones.json'); }

  try {
    const resp = await fetch('/shared/player_bounds.json', { cache: 'no-store' });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    playerBounds = await resp.json();
  } catch (_) { missing.push('player_bounds.json'); }

  if (missing.length) {
    const files = missing.map(f => `<code>${f}</code>`).join(' and ');
    showBanner(
      `Couldn’t load ${files} from <code>/shared/</code>. Serve the repo root over HTTP ` +
      '— e.g. <code>python -m http.server 3000</code> from the project root, then open ' +
      '<code>/frontend/</code> — and reload. Serving <code>frontend/</code> itself puts ' +
      '<code>/shared/</code> out of reach, and opening <code>index.html</code> directly ' +
      '(file://) blocks local fetches entirely.');
    resultEmptyText.textContent = 'Data failed to load — see the message above.';
    return;
  }

  buildCategoryButtons();
  buildPlayerSelect();
  applyPlayerBounds();      // size the zone shapes now that the bounds have loaded
  updateMask(null);         // start fully greyed until a shot type is chosen
  updatePlayerMask(null);   // same, for player mode's mask

  svg.addEventListener('click', onCourtClick);
  svg.addEventListener('mousemove', onCourtMove);
  svg.addEventListener('mouseleave', onCourtLeave);
  toggleMoving.addEventListener('change', onToggleChange);
  playerSelectEl.addEventListener('change', onPlayerSelectChange);
  for (const b of modeButtons) b.addEventListener('click', () => setMode(b.dataset.mode));
}

document.addEventListener('DOMContentLoaded', init);
