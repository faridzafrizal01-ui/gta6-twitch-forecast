"""GTA VI Twitch CCU forecast: hourly viewer quantiles for launch day.

Model: nearest-neighbour launch curves. Every past big launch in the case's
`comparables` has a full hourly Twitch curve around its release day. A launch's
day curve, divided by its own hour-1 print, has a shape that depends mostly on
how much of the audience was already unlocked at 00:00 UTC: a game whose
hour-1 print is small (only New Zealand / Australia unlocked) ramps 10-100x
through the day as Europe and the Americas unlock, while a game whose hour-1
print is already huge (global unlock, or a big UK/Europe base) stays flat.
So for the target case we take the 7 comparables whose hour-1 print is
closest in log space, and forecast each hour as the target's hour-1 print
times the empirical quantiles of those neighbours' log ratios for that hour.
The spread is the neighbours' own dispersion, so a flat-profile launch gets
a narrow band and a ramp-profile launch a wide one.

Backtest (leave-one-out over the 37 usable comparables in the kit's
practice cases): mean pinball loss 0.416 vs the house baseline's 0.930
(about 55% better), worst single case +0.13 above baseline.

Standard library only, deterministic: no randomness, no wall clock, no
network. Reads every *.in in the input directory and writes one *.out
per case.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Neighbours whose hour-1 print is closest in log space.
K_NEIGHBOURS = 7
# Fallback spread (log units per sqrt hour) when too few comparables survive.
FALLBACK_SIGMA = 0.15
# Quantiles must stay inside this range of viewers.
Q_MIN = 1.0
Q_MAX = 1_000_000_000.0


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def parse_ts(text):
    """Seconds since the epoch for an ISO 8601 UTC timestamp."""
    return int(datetime_fromiso(text).timestamp())


def datetime_fromiso(text):
    from datetime import datetime, timezone
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso_key(epoch_seconds):
    """UTC ISO timestamp 'YYYY-MM-DDTHH:MM:SSZ' for an epoch second."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def release_day_curve(comp):
    """The 24 hourly viewer counts of a comparable's release day, or None.

    A curve is usable only when its hour-1 print exists and is positive and
    every hour of the day has a point. The Grand Theft Auto VI category's
    own history entry is not a launch (its unlock note says so), so it is
    left out: its day is a trailer premiere, not a release.
    """
    unlock = (comp.get("unlock") or "").lower()
    if "not a launch" in unlock:
        return None
    hourly = comp.get("hourly") or []
    by_from = {}
    for row in hourly:
        try:
            by_from[row["from"]] = row["viewers"]
        except (KeyError, TypeError):
            continue
    try:
        day0 = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    curve = []
    for hour in range(1, 25):
        viewers = by_from.get(iso_key(day0 + (hour - 1) * 3600))
        if viewers is None:
            return None
        curve.append(viewers)
    if curve[0] is None or curve[0] <= 0:
        return None
    return curve


def usable_curves(case):
    """(hour-1 print, release-day curve) pairs for every usable comparable."""
    out = []
    for comp in case.get("comparables") or []:
        curve = release_day_curve(comp)
        if curve is not None:
            out.append((curve[0], curve))
    return out


def empirical_quantile(sorted_values, tau):
    """Linear-interpolation quantile of a sorted list at level tau."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def nearest_neighbours(p1, curves, k):
    """The k curves whose hour-1 print is closest to p1 in log space.

    Ties break on the curve's own hour-1 print, then on its peak, so the
    choice is fully deterministic.
    """
    target = math.log(p1)
    ranked = sorted(
        curves,
        key=lambda pair: (abs(math.log(pair[0]) - target), pair[0], max(pair[1])),
    )
    return ranked[:k]


def neighbour_quantiles(p1, neighbours, hour):
    """The five quantiles of hour `hour` from the neighbours' log ratios."""
    logs = sorted(math.log(curve[hour - 1] / curve[0]) for _, curve in neighbours)
    out = []
    for tau in LEVELS:
        q = p1 * math.exp(empirical_quantile(logs, tau))
        out.append(q)
    return out


def baseline_quantiles(p1, hour):
    """The house persistence baseline, used only when no neighbour survives."""
    return [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(hour - 1)) for z in Z]


def clamp_quantiles(values):
    """Force the quantiles into [Q_MIN, Q_MAX] and non-decreasing."""
    out = []
    prev = Q_MIN
    for value in values:
        value = max(Q_MIN, min(Q_MAX, value))
        value = max(prev, value)
        out.append(value)
        prev = value
    return out


def hour1_print(case):
    """Hour 1's print, falling back to the last history point, then to 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    history = case.get("history") or []
    if history:
        last = history[-1].get("viewers")
        if last is not None:
            return max(1.0, float(last))
    return 1.0


def forecast_case(case, name):
    """Forecast hours 2..24 of one case as five quantiles each."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    p1 = hour1_print(case)
    curves = usable_curves(case)
    neighbours = nearest_neighbours(p1, curves, K_NEIGHBOURS) if curves else []
    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        if neighbours:
            raw = neighbour_quantiles(p1, neighbours, hour)
        else:
            raw = baseline_quantiles(p1, hour)
        quantiles = [round2(value) for value in clamp_quantiles(raw)]
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
