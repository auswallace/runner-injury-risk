"""Build dashboard/index.html - the weekly decision-support triage view.

Retrospective demo over the time-aware TEST period (the held-out final ~20% of the
timeline), scored by the exact 04 winner: logistic C=0.01 on w7 raw+z, fit on train
only, Platt-calibrated on validation, thresholds frozen on validation. The script
asserts its refit reproduces 04's saved test numbers before emitting anything.

The output HTML embeds flagged athlete-day rows, so it is gitignored per the rule
in data/README.md (committed artifacts must not carry individual-level rows).
Rebuild any time with:  python3 dashboard/build_dashboard.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = 42
ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / 'data' / 'processed'
OUT = Path(__file__).resolve().parent / 'index.html'

X = pd.read_csv(PROC / 'features_day.csv.gz')
SEL = json.load(open(PROC / 'model_selection.json'))
EVAL = json.load(open(PROC / 'evaluation_results.json'))
FEATS = SEL['features']

# ---- rebuild the 04 time-aware winner, byte-for-byte the same construction ----
T1, T2 = X.Date.quantile([.6, .8]).astype(int)
PURGE = SEL['time_aware']['purge']
ta_train = X[X.Date < T1]
ta_val   = X[(X.Date >= T1 + PURGE) & (X.Date < T2)]
ta_test  = X[X.Date >= T2 + PURGE]

pipe = Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                 ('scale', StandardScaler()),
                 ('clf', LogisticRegression(C=0.01, class_weight='balanced',
                                            max_iter=3000, random_state=RNG))])
pipe.fit(ta_train[FEATS], ta_train.injury)

EPS = 1e-12
def _logodds(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)

p_val = pipe.predict_proba(ta_val[FEATS])[:, 1]
platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logodds(p_val), ta_val.injury)

p_raw = pipe.predict_proba(ta_test[FEATS])[:, 1]
p_cal = platt.predict_proba(_logodds(p_raw))[:, 1]

# refit must reproduce 04's one-shot numbers exactly, or nothing gets built
head = EVAL['headline']['time-aware']
assert abs(average_precision_score(ta_test.injury, p_raw) - head['ap']) < 1e-9
assert abs(roc_auc_score(ta_test.injury, p_raw) - head['auc']) < 1e-9

THR = {int(k): float(v) for k, v in EVAL['thresholds_frozen_on_validation']['time-aware'].items()}
LOOSEST = THR[10]

# ---- per-flag "why": standardized logistic contributions (coef * z-value) ----
names = list(pipe.named_steps['impute'].get_feature_names_out(FEATS))
Z = pipe[:-1].transform(ta_test[FEATS])
contrib = Z * pipe.named_steps['clf'].coef_[0]

BASE = {'km_sum': '7-day distance', 'sessions': 'sessions (7d)',
        'km_mod': 'moderate-intensity km (7d)', 'km_hi': 'high-intensity km (7d)',
        'pct_mod': 'share of moderate-intensity km', 'pct_hi': 'share of high-intensity km',
        'rest_days': 'rest days (7d)', 'strength_n': 'strength sessions (7d)',
        'alt_hours': 'cross-training hours (7d)', 'exertion_avg': 'avg perceived exertion',
        'recovery_avg': 'avg perceived recovery', 'success_avg': 'avg session success'}

def label(name):
    if name.startswith('missingindicator_'):
        return label(name[len('missingindicator_'):]) + ' (insufficient history)'
    stem = name.split('__')[0]
    return BASE.get(stem, stem) + (' vs own history' if name.endswith('__z') else '')

# ---- assemble the embedded data: weekly aggregates + flagged athlete-weeks ----
test = ta_test.reset_index(drop=True)
test['week'] = test.Date // 7
week_min, week_max = int(test.week.min()), int(test.week.max())

weeks = []
for w in range(week_min, week_max + 1):
    s = test[test.week == w]
    weeks.append({'w': w - week_min,
                  'days': int(len(s)),
                  'injuries': int(s.injury.sum()),
                  'flags': {k: int((p_cal[s.index] >= t).sum()) for k, t in THR.items()}})

flags = []
flagged = test[p_cal >= LOOSEST]
for (ath, w), s in flagged.groupby(['Athlete ID', 'week']):
    idx = s.index.to_numpy()
    peak = idx[np.argmax(p_cal[idx])]
    top = np.argsort(-np.abs(contrib[peak]))[:3]
    flags.append({
        'athlete': int(ath),
        'week': int(w) - week_min,
        'days': [[int(test.Date[i]), round(float(p_cal[i]), 6), int(test.injury[i])] for i in idx],
        'peak_p': round(float(p_cal[peak]), 6),
        'context': {'km': round(float(test.km_sum__w7[peak]), 1),
                    'sessions': int(round(float(test.sessions__w7[peak]))),
                    'exertion': round(float(test.exertion_avg__w7[peak]), 2)},
        'signals': [{'label': label(names[j]), 'push': round(float(contrib[peak][j]), 3)}
                    for j in top],
    })

alert = {str(r['budget/wk']): {'precision': r['precision'], 'recall': r['recall'],
                               'achieved': r['achieved/wk']}
         for r in EVAL['alert_budget'] if r['split'] == 'time-aware'}

DATA = {'meta': {'n_weeks': len(weeks), 'test_rows': int(len(test)),
                 'test_injuries': int(test.injury.sum()),
                 'headline': {'ap': head['ap'], 'auc': head['auc'], 'chance': head['chance_ap']},
                 'alert': alert,
                 'thresholds': {str(k): v for k, v in THR.items()}},
        'weeks': weeks, 'flags': flags}

TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Runner injury-risk weekly triage</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb;
  --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --flag: #2a78d6; --flag-sel: #1c5cab; --injury: #eb6834;
  --up: #d03b3b; --down: #2a78d6;
  --band-high: #d03b3b; --band-elev: #ec835a; --band-watch: #fab219;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --flag: #3987e5; --flag-sel: #86b6ef; --injury: #d95926;
    --up: #d03b3b; --down: #3987e5;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
       font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1060px; margin: 0 auto; padding: 24px 20px 48px; }
header h1 { font-size: 21px; margin: 0 0 2px; }
header .sub { color: var(--ink-2); margin: 0; }
header .disclaimer { color: var(--muted); font-size: 12.5px; margin: 6px 0 0; }
.card { background: var(--surface); border: 1px solid var(--border);
        border-radius: 10px; padding: 16px 18px; }
.filters { display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
           margin: 18px 0 14px; }
.filters .group { display: flex; align-items: center; gap: 8px; }
.filters label { color: var(--ink-2); font-size: 13px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px;
       overflow: hidden; }
.seg button { border: 0; background: transparent; color: var(--ink-2);
              padding: 5px 12px; cursor: pointer; font: inherit; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button[aria-pressed="true"] { background: var(--flag); color: #fff; }
.stepper button { border: 1px solid var(--border); background: var(--surface);
                  color: var(--ink); border-radius: 8px; width: 28px; height: 28px;
                  cursor: pointer; font: inherit; }
select { font: inherit; color: var(--ink); background: var(--surface);
         border: 1px solid var(--border); border-radius: 8px; padding: 4px 8px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px; margin-bottom: 14px; }
.tile .label { color: var(--ink-2); font-size: 12.5px; }
.tile .value { font-size: 26px; font-weight: 600; margin: 2px 0; }
.tile .note { color: var(--muted); font-size: 12px; }
.chart-head { display: flex; justify-content: space-between; align-items: baseline;
              flex-wrap: wrap; gap: 8px; }
.chart-head h2, .week-view h2 { font-size: 15px; margin: 0 0 4px; }
.legend { display: flex; gap: 16px; color: var(--ink-2); font-size: 12.5px; }
.legend .key { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 10px; height: 10px; border-radius: 3px; background: var(--flag); }
.dot { width: 9px; height: 9px; border-radius: 50%; background: var(--injury); }
.chart-wrap { position: relative; }
svg { display: block; width: 100%; height: auto; }
svg:focus { outline: 2px solid var(--flag); outline-offset: 2px; border-radius: 4px; }
.tooltip { position: absolute; pointer-events: none; background: var(--surface);
           border: 1px solid var(--border); border-radius: 8px; padding: 7px 10px;
           font-size: 12.5px; color: var(--ink); box-shadow: 0 2px 10px rgba(0,0,0,.12);
           white-space: nowrap; display: none; z-index: 2; }
.tooltip .t-title { font-weight: 600; }
.tooltip .muted { color: var(--muted); }
details { margin-top: 10px; color: var(--ink-2); font-size: 13px; }
details table { margin-top: 8px; }
table { border-collapse: collapse; width: 100%; }
th { text-align: left; color: var(--ink-2); font-weight: 500; font-size: 12.5px;
     border-bottom: 1px solid var(--grid); padding: 6px 10px 6px 0; }
td { border-bottom: 1px solid var(--grid); padding: 8px 10px 8px 0;
     vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tr:last-child td { border-bottom: 0; }
.week-view { margin-top: 14px; }
.week-view .meta { color: var(--ink-2); font-size: 13px; margin: 0 0 8px; }
.band { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px;
        border: 1px solid var(--border); border-radius: 999px; padding: 2px 9px; }
.band i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.signal { display: grid; grid-template-columns: minmax(150px, 1fr) 74px;
          gap: 8px; align-items: center; font-size: 12.5px; margin: 2px 0; }
.signal .bar { position: relative; height: 8px; }
.signal .bar span { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
.outcome { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2);
           font-size: 12.5px; }
.empty { color: var(--muted); padding: 14px 0; }
.caveats { margin-top: 14px; }
.caveats h2 { font-size: 15px; margin: 0 0 6px; }
.caveats ul { margin: 0; padding-left: 18px; color: var(--ink-2); }
.caveats li { margin: 5px 0; }
footer { margin-top: 18px; color: var(--muted); font-size: 12.5px; line-height: 1.6; }
footer a { color: var(--ink-2); }
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Runner injury-risk weekly triage</h1>
  <p class="sub">Retrospective decision-support demo over the held-out final period,
  scored only with information available before each day.</p>
  <p class="disclaimer">Independent portfolio project on the public Groningen runner
  dataset (masked athlete IDs). Not affiliated with any team or program. Scores are
  shown as ranked risk bands, not probabilities &mdash; see &ldquo;when not to trust
  this&rdquo; below.</p>
</header>

<div class="filters" id="filters">
  <div class="group">
    <label for="weekSel">Week</label>
    <span class="stepper"><button id="wPrev" aria-label="previous week">&#8249;</button></span>
    <select id="weekSel" aria-label="select week"></select>
    <span class="stepper"><button id="wNext" aria-label="next week">&#8250;</button></span>
  </div>
  <div class="group">
    <label id="budgetLabel">Alert budget</label>
    <span class="seg" role="group" aria-labelledby="budgetLabel" id="budgetSeg"></span>
    <span style="color:var(--muted);font-size:12px">flags/week the staff can act on</span>
  </div>
</div>

<div class="kpis" id="kpis"></div>

<div class="card">
  <div class="chart-head">
    <h2>Flags issued vs injuries that occurred, by week</h2>
    <div class="legend">
      <span class="key"><span class="swatch"></span>flags issued</span>
      <span class="key"><span class="dot"></span>injuries occurred</span>
    </div>
  </div>
  <div class="chart-wrap">
    <svg id="chart" viewBox="0 0 960 240" tabindex="0" role="img"
         aria-label="Weekly flags and injuries. Use left and right arrow keys to move between weeks."></svg>
    <div class="tooltip" id="tip"></div>
  </div>
  <details>
    <summary>Data table (weekly counts)</summary>
    <table id="weekTable"><thead><tr>
      <th>week</th><th class="num">athlete-days logged</th>
      <th class="num">flags</th><th class="num">injuries</th>
    </tr></thead><tbody></tbody></table>
  </details>
</div>

<div class="card week-view">
  <h2 id="weekTitle"></h2>
  <p class="meta" id="weekMeta"></p>
  <div id="flagArea"></div>
</div>

<div class="card caveats">
  <h2>When not to trust this</h2>
  <ul>
    <li><strong>Bands, not probabilities.</strong> Calibrated scores are statistically
    no sharper than the injury base rate (~1&ndash;2%); the model earns its keep by
    <em>ranking</em> athlete-days, so risk is shown as bands and never as a percentage.</li>
    <li><strong>Most injuries arrive unflagged.</strong> At actionable budgets the
    model catches 2&ndash;16% of injuries in advance. This is a triage aid for where
    to look first, not a safety net.</li>
    <li><strong>Individual athletes can be misranked.</strong> On fully held-out
    runners, per-athlete AUC ranged 0.27&ndash;0.83 and 2 of 15 sat below chance.
    Check an athlete's track record before acting on their score.</li>
    <li><strong>Weaker for high-volume, long-tenure athletes</strong> (held-out AUC
    0.60&ndash;0.62 vs 0.66&ndash;0.67 for lower-volume, shorter-tenure groups).</li>
    <li><strong>Thresholds drift across populations.</strong> Budgets frozen on one
    group under- or over-flag another; re-anchor before any new deployment.</li>
    <li><strong>Associations, not causes.</strong> The signals shown are model
    evidence, not instructions &mdash; nothing here says changing a factor changes risk.</li>
  </ul>
</div>

<footer id="foot"></footer>
</div>

<script>
const DATA = %%DATA%%;

const $ = (s) => document.querySelector(s);
const meta = DATA.meta;
const BUDGETS = [1, 2, 3, 5, 10];
const state = { budget: 3, week: 0 };

const thr = (k) => meta.thresholds[String(k)];
const BANDS = [
  { name: 'High',     k: 1,  color: 'var(--band-high)' },
  { name: 'Elevated', k: 3,  color: 'var(--band-elev)' },
  { name: 'Watch',    k: 10, color: 'var(--band-watch)' },
];
const bandOf = (p) => BANDS.find(b => p >= thr(b.k)) || BANDS[BANDS.length - 1];
const fmtPct = (x) => (100 * x).toFixed(1) + '%';

// --- filters ---
const weekSel = $('#weekSel');
DATA.weeks.forEach((w, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = 'Test week ' + (i + 1);
  weekSel.appendChild(o);
});
weekSel.addEventListener('change', () => setWeek(+weekSel.value));
$('#wPrev').addEventListener('click', () => setWeek(state.week - 1));
$('#wNext').addEventListener('click', () => setWeek(state.week + 1));

const seg = $('#budgetSeg');
BUDGETS.forEach(k => {
  const b = document.createElement('button');
  b.textContent = k; b.setAttribute('aria-pressed', String(k === state.budget));
  b.addEventListener('click', () => {
    state.budget = k;
    seg.querySelectorAll('button').forEach(x =>
      x.setAttribute('aria-pressed', String(+x.textContent === k)));
    renderAll();
  });
  seg.appendChild(b);
});

function setWeek(i) {
  state.week = Math.max(0, Math.min(DATA.weeks.length - 1, i));
  weekSel.value = state.week;
  renderChart(); renderWeek();
}

// --- KPI tiles ---
function renderKPIs() {
  const a = meta.alert[String(state.budget)];
  const tiles = [
    { label: 'Ranking quality (average precision)', value: meta.headline.ap.toFixed(3),
      note: (meta.headline.ap / meta.headline.chance).toFixed(1) + 'x the chance level of ' + meta.headline.chance.toFixed(3) },
    { label: 'ROC AUC (held-out period)', value: meta.headline.auc.toFixed(2),
      note: 'published benchmark on its own split: 0.72' },
    { label: 'Precision at this budget', value: fmtPct(a.precision),
      note: 'flags where injury followed the next session' },
    { label: 'Recall at this budget', value: fmtPct(a.recall),
      note: a.achieved.toFixed(1) + ' flags/week achieved of ' + state.budget + ' budgeted' },
  ];
  $('#kpis').innerHTML = tiles.map(t =>
    '<div class="card tile"><div class="label">' + t.label + '</div>' +
    '<div class="value">' + t.value + '</div>' +
    '<div class="note">' + t.note + '</div></div>').join('');
}

// --- weekly chart (hand-rolled SVG: bars = flags, dots = injuries) ---
const svg = $('#chart'), tip = $('#tip');
const M = { l: 34, r: 8, t: 10, b: 26 }, W = 960, H = 240;
const innerW = W - M.l - M.r, innerH = H - M.t - M.b;
let hoverW = -1;

function roundedBar(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  return 'M' + x + ' ' + (y + h) + 'V' + (y + r) +
         'Q' + x + ' ' + y + ' ' + (x + r) + ' ' + y +
         'H' + (x + w - r) +
         'Q' + (x + w) + ' ' + y + ' ' + (x + w) + ' ' + (y + r) +
         'V' + (y + h) + 'Z';
}

function renderChart() {
  const flagsAt = DATA.weeks.map(w => w.flags[String(state.budget)]);
  const yMax = Math.max(2, ...flagsAt, ...DATA.weeks.map(w => w.injuries));
  const step = yMax > 12 ? 5 : (yMax > 6 ? 2 : 1);
  const top = Math.ceil(yMax / step) * step;
  const yScale = v => M.t + innerH - (v / top) * innerH;
  const slot = innerW / DATA.weeks.length;
  const barW = Math.min(24, Math.max(2, slot - 2));

  let s = '';
  for (let v = 0; v <= top; v += step) {
    const y = yScale(v);
    s += '<line x1="' + M.l + '" x2="' + (W - M.r) + '" y1="' + y + '" y2="' + y +
         '" stroke="' + (v === 0 ? 'var(--baseline)' : 'var(--grid)') + '" stroke-width="1"/>';
    s += '<text x="' + (M.l - 7) + '" y="' + (y + 3.5) + '" text-anchor="end" ' +
         'font-size="10.5" fill="var(--muted)">' + v + '</text>';
  }
  DATA.weeks.forEach((w, i) => {
    const x = M.l + i * slot + (slot - barW) / 2;
    const f = flagsAt[i];
    if (f > 0) {
      const y = yScale(f);
      s += '<path d="' + roundedBar(x, y, barW, yScale(0) - y, 4) + '" fill="' +
           (i === state.week ? 'var(--flag-sel)' : 'var(--flag)') + '"/>';
    }
    if (w.injuries > 0) {
      s += '<circle cx="' + (x + barW / 2) + '" cy="' + yScale(w.injuries) +
           '" r="4.5" fill="var(--injury)" stroke="var(--surface)" stroke-width="2"/>';
    }
    if (i === state.week) {
      s += '<path d="M' + (x + barW / 2 - 4) + ' ' + (H - 12) + ' h8 l-4 -6 Z" fill="var(--ink-2)"/>';
    }
    if (i === 0 || (i + 1) % 10 === 0) {
      s += '<text x="' + (x + barW / 2) + '" y="' + (H - 1) + '" text-anchor="middle" ' +
           'font-size="10.5" fill="var(--muted)">' + (i + 1) + '</text>';
    }
  });
  svg.innerHTML = s;

  const tb = $('#weekTable tbody');
  tb.innerHTML = DATA.weeks.map((w, i) =>
    '<tr><td>' + (i + 1) + '</td><td class="num">' + w.days + '</td>' +
    '<td class="num">' + w.flags[String(state.budget)] + '</td>' +
    '<td class="num">' + w.injuries + '</td></tr>').join('');
}

function weekAt(clientX) {
  const r = svg.getBoundingClientRect();
  const x = (clientX - r.left) / r.width * W - M.l;
  return Math.max(0, Math.min(DATA.weeks.length - 1,
    Math.floor(x / (innerW / DATA.weeks.length))));
}
function showTip(i) {
  const w = DATA.weeks[i];
  tip.innerHTML = '<div class="t-title">Test week ' + (i + 1) + '</div>' +
    w.flags[String(state.budget)] + ' flags &middot; ' + w.injuries + ' injuries' +
    '<div class="muted">' + w.days + ' athlete-days logged</div>';
  tip.style.display = 'block';
  const slot = innerW / DATA.weeks.length;
  const wrap = svg.parentElement.getBoundingClientRect();
  const r = svg.getBoundingClientRect();
  const px = (M.l + (i + 0.5) * slot) / W * r.width + (r.left - wrap.left);
  tip.style.left = Math.min(wrap.width - tip.offsetWidth - 4, Math.max(0, px + 10)) + 'px';
  tip.style.top = '8px';
}
svg.addEventListener('mousemove', e => { hoverW = weekAt(e.clientX); showTip(hoverW); });
svg.addEventListener('mouseleave', () => { tip.style.display = 'none'; hoverW = -1; });
svg.addEventListener('click', e => setWeek(weekAt(e.clientX)));
svg.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') { setWeek(state.week - 1); showTip(state.week); e.preventDefault(); }
  if (e.key === 'ArrowRight') { setWeek(state.week + 1); showTip(state.week); e.preventDefault(); }
});

// --- flagged-athletes view for the selected week ---
function renderWeek() {
  const t = thr(state.budget);
  const w = DATA.weeks[state.week];
  const entries = DATA.flags
    .filter(f => f.week === state.week && f.peak_p >= t)
    .sort((a, b) => b.peak_p - a.peak_p);
  $('#weekTitle').textContent = 'Test week ' + (state.week + 1) + ' - flagged athletes';
  $('#weekMeta').textContent = w.days + ' athlete-days logged, ' +
    entries.length + ' athlete' + (entries.length === 1 ? '' : 's') + ' flagged at this budget, ' +
    w.injuries + ' injur' + (w.injuries === 1 ? 'y' : 'ies') + ' occurred this week ' +
    '(outcomes shown because this is a retrospective demo).';
  if (!entries.length) {
    $('#flagArea').innerHTML = '<div class="empty">No athletes cross the ' +
      state.budget + '-flags/week threshold this week. A quiet week is a valid output.</div>';
    return;
  }
  const maxPush = Math.max(...entries.flatMap(f => f.signals.map(s => Math.abs(s.push))));
  const rows = entries.map(f => {
    const days = f.days.filter(d => d[1] >= t);
    const hit = days.some(d => d[2] === 1);
    const b = bandOf(f.peak_p);
    const sig = f.signals.map(s => {
      const wPct = Math.abs(s.push) / maxPush * 100;
      const pos = s.push >= 0;
      return '<div class="signal"><span>' + s.label + '</span>' +
        '<span class="bar"><span style="' +
        (pos ? 'left:0;background:var(--up)' : 'right:0;background:var(--down)') +
        ';width:' + Math.max(4, wPct) + '%"></span></span></div>';
    }).join('');
    return '<tr>' +
      '<td><strong>Athlete ' + f.athlete + '</strong></td>' +
      '<td><span class="band"><i style="background:' + b.color + '"></i>' + b.name + '</span></td>' +
      '<td class="num">' + days.length + '</td>' +
      '<td>' + sig + '</td>' +
      '<td class="num">' + f.context.km.toFixed(1) + '</td>' +
      '<td class="num">' + f.context.sessions + '</td>' +
      '<td class="num">' + f.context.exertion.toFixed(2) + '</td>' +
      '<td>' + (hit ? '<span class="outcome"><span class="dot"></span>injury followed</span>'
                    : '<span class="outcome">&mdash;</span>') + '</td></tr>';
  }).join('');
  $('#flagArea').innerHTML =
    '<table><thead><tr><th>Athlete</th><th>Risk band</th><th class="num">Days flagged</th>' +
    '<th>Why flagged (top signals, red = raises risk)</th><th class="num">7d km</th>' +
    '<th class="num">Sessions</th><th class="num">Exertion <span style="white-space:nowrap">(0&#8211;1)</span></th>' +
    '<th>Outcome (retrospective)</th></tr></thead><tbody>' + rows + '</tbody></table>';
}

function renderAll() { renderKPIs(); renderChart(); renderWeek(); }

$('#foot').innerHTML =
  'Model: logistic regression (C=0.01, class-weighted) on 24 features - 12 seven-day ' +
  'training-load aggregates plus the same 12 normalized against each athlete\'s own ' +
  'earlier history. Trained on the first 60% of the timeline, Platt-calibrated on the ' +
  'next 20%, alert thresholds frozen there, and evaluated once on this final held-out ' +
  'period (' + meta.test_rows.toLocaleString() + ' athlete-days, ' + meta.test_injuries +
  ' injuries). Risk bands: High &ge; the ~1-flag/week threshold, Elevated &ge; ~3/week, ' +
  'Watch &ge; ~10/week.<br>' +
  'Data: Lovdal, den Hartigh &amp; Azzopardi (2021), <em>Injury Prediction in Competitive ' +
  'Runners With Machine Learning</em>, IJSPP - public replication data, DataverseNL ' +
  'DOI 10.34894/uwu9pv. Methods, assumptions and limitations: ' +
  '<a href="https://github.com/auswallace/runner-injury-risk">github.com/auswallace/runner-injury-risk</a>.';

// initial state: most recent week with at least one flag at the default budget
for (let i = DATA.weeks.length - 1; i >= 0; i--) {
  if (DATA.weeks[i].flags[String(state.budget)] > 0) { state.week = i; break; }
}
weekSel.value = state.week;
renderAll();
</script>
</body>
</html>
'''

html = TEMPLATE.replace('%%DATA%%', json.dumps(DATA, separators=(',', ':')))
OUT.write_text(html)
n_aw = len(flags)
print(f'wrote {OUT} ({len(html)/1024:.0f} KB): {len(weeks)} weeks, '
      f'{n_aw} flagged athlete-weeks at the loosest budget, '
      f'{int(test.injury.sum())} injuries in period')
