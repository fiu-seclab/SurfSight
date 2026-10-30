"""Baselines from related work, applied to the same 200 ms windows as the LSTM.

Each baseline keeps the feature definition and the model family of the cited
study and is trained on the same windows, session-grouped splits, and
balancing as the LSTM and the 1D-CNN. The decision unit is the 200 ms window
for every model, which is narrower than the unit the original studies used.
"""

import numpy as np
import pandas as pd

from pipeline.config import MAX_SEQUENCE_LENGTH, MIN_EVENTS_PER_SUBSESSION, SESSION_GAP_MINUTES
from pipeline.models import _with_coordinates

EVENT_TYPES = [
    "mousedown", "mouseup", "mousemove", "mouseover", "mouseout", "mousewheel", "wheel",
    "touchstart", "touchend", "touchmove", "deviceorientation", "keydown", "keyup", "keypress",
    "click", "dblclick", "scroll", "change", "select", "submit", "reset", "contextmenu",
    "cut", "copy", "paste", "load", "unload", "beforeunload", "blur", "focus", "resize",
    "error", "abort", "online", "offline", "storage", "popstate", "hashchange", "pagehide",
    "pageshow", "message", "beforeprint", "afterprint",
]
TYPE_INDEX = {name: index for index, name in enumerate(EVENT_TYPES)}
BUTTON = ("mousedown", "mouseup", "click", "dblclick")
KEY = ("keydown", "keyup", "keypress")
SCROLL = ("scroll", "wheel", "mousewheel")
ENTROPY_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, np.inf])


def iter_windows(events, window_ms, class_name, min_events=MIN_EVENTS_PER_SUBSESSION, gap_minutes=SESSION_GAP_MINUTES):
    """Yield (session_key, names, seconds, x, y) for each window with enough events.

    Same windowing as splits.collect_windows: sessions split on a gap,
    windows aligned to the session start, the first MAX_SEQUENCE_LENGTH
    events of a window kept.
    """
    if events is None or len(events) == 0:
        return
    if "x" not in events.columns:
        events = _with_coordinates(events)
    ordered = events.sort_values(["userId", "timestamp_num"], kind="mergesort")
    user_ids = ordered["userId"].to_numpy()
    timestamps = ordered["timestamp_num"].to_numpy(dtype=np.int64)
    names = ordered["eventName"].to_numpy().astype(str)
    x_pos = ordered["x"].to_numpy(dtype=np.float64)
    y_pos = ordered["y"].to_numpy(dtype=np.float64)
    gap_ms = gap_minutes * 60 * 1000
    new_user = np.empty(len(ordered), dtype=bool)
    new_user[0] = True
    if len(ordered) > 1:
        new_user[1:] = user_ids[1:] != user_ids[:-1]
    gap = np.zeros(len(ordered), dtype=bool)
    if len(ordered) > 1:
        gap[1:] = (timestamps[1:] - timestamps[:-1]) > gap_ms
        gap[new_user] = False
    session_local = np.cumsum(new_user | gap)
    session_start = pd.Series(timestamps).groupby(session_local, sort=False).transform("min").to_numpy()
    window_id = (timestamps - session_start) // window_ms
    change = np.empty(len(ordered), dtype=bool)
    change[0] = True
    if len(ordered) > 1:
        change[1:] = (session_local[1:] != session_local[:-1]) | (window_id[1:] != window_id[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(ordered))
    for left, right in zip(starts, ends):
        length = int(right - left)
        if length < min_events:
            continue
        stop = left + min(length, MAX_SEQUENCE_LENGTH)
        key = f"{class_name}\t{user_ids[left]}\t{int(session_local[left])}"
        seconds = (timestamps[left:stop] - timestamps[left]) / 1000.0
        yield key, names[left:stop], seconds, x_pos[left:stop], y_pos[left:stop]


def _intervals_ms(seconds):
    if len(seconds) < 2:
        return np.zeros(0)
    return np.diff(seconds) * 1000.0


def _entropy(intervals):
    if len(intervals) < 2:
        return 0.0
    counts, _ = np.histogram(intervals, bins=ENTROPY_EDGES)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _std(values):
    return float(np.std(values)) if len(values) > 1 else 0.0


def _maximum(values):
    return float(np.max(values)) if len(values) else 0.0


def folch_features(names, seconds, x, y):
    """Mean and max of velocity and acceleration, overall and per axis (Folch et al.)."""
    move = names == "mousemove"
    out = np.zeros(12)
    if move.sum() < 3:
        return out
    t, px, py = seconds[move], x[move], y[move]
    dt = np.diff(t)
    ok = dt > 0
    if ok.sum() < 1:
        return out
    vx = np.diff(px)[ok] / dt[ok]
    vy = np.diff(py)[ok] / dt[ok]
    speed = np.hypot(vx, vy)
    mid = ((t[:-1] + t[1:]) / 2.0)[ok]
    out[0:6] = [_mean(speed), _maximum(speed), _mean(np.abs(vx)), _mean(np.abs(vy)), _maximum(np.abs(vx)), _maximum(np.abs(vy))]
    if len(mid) >= 2:
        dm = np.diff(mid)
        good = dm > 0
        if good.sum() >= 1:
            ax = np.diff(vx)[good] / dm[good]
            ay = np.diff(vy)[good] / dm[good]
            acc = np.hypot(ax, ay)
            out[6:12] = [_mean(acc), _maximum(acc), _mean(np.abs(ax)), _mean(np.abs(ay)), _maximum(np.abs(ax)), _maximum(np.abs(ay))]
    return out


def chu_features(names, seconds):
    """Count, timing entropy, and mean interval for mouse and keyboard streams (Chu et al.)."""
    streams = [
        np.ones(len(names), dtype=bool),
        names == "mousemove",
        np.isin(names, BUTTON),
        np.isin(names, KEY),
    ]
    out = []
    for mask in streams:
        times = seconds[mask]
        gaps = _intervals_ms(times)
        out.extend([float(mask.sum()), _entropy(gaps), _mean(gaps)])
    return np.array(out)


def uitrace_features(names, seconds, x, y):
    """Interval, click, key, scroll, and pointer-shape statistics (Lugoloobi et al., FP-Agent)."""
    gaps = _intervals_ms(seconds)
    duration = float(seconds[-1] - seconds[0]) * 1000.0 if len(seconds) > 1 else 0.0
    click = names == "click"
    keydown = names == "keydown"
    keyup = names == "keyup"
    scroll = np.isin(names, SCROLL)
    move = names == "mousemove"
    click_gaps = _intervals_ms(seconds[click])
    key_gaps = _intervals_ms(seconds[keydown])
    scroll_gaps = _intervals_ms(seconds[scroll])
    hold = []
    if keydown.any() and keyup.any():
        down_times, up_times = seconds[keydown], seconds[keyup]
        for start in down_times:
            later = up_times[up_times >= start]
            if len(later):
                hold.append((later[0] - start) * 1000.0)
    cx, cy = x[click], y[click]
    path = turn = straight = 0.0
    if move.sum() >= 3:
        px, py = x[move], y[move]
        dx, dy = np.diff(px), np.diff(py)
        step = np.hypot(dx, dy)
        path = float(step.sum())
        straight = float(np.hypot(px[-1] - px[0], py[-1] - py[0]) / path) if path > 0 else 0.0
        moving = step > 0
        if moving.sum() >= 2:
            angle = np.arctan2(dy[moving], dx[moving])
            turn_angle = np.abs(np.angle(np.exp(1j * np.diff(angle))))
            turn = float(turn_angle.mean())
    return np.array([
        float(len(names)), duration, float(len(set(names.tolist()))),
        _mean(gaps), _std(gaps), float(gaps.min()) if len(gaps) else 0.0, _maximum(gaps),
        float(move.sum()), float(np.isin(names, BUTTON).sum()), float(click.sum()),
        float(keydown.sum()), float(scroll.sum()),
        _mean(click_gaps), _std(key_gaps), _mean(hold), _mean(scroll_gaps),
        _std(cx), _std(cy),
        float(cx.max() - cx.min()) if len(cx) else 0.0, float(cy.max() - cy.min()) if len(cy) else 0.0,
        path, straight, turn,
    ])


def type_counts(names):
    counts = np.zeros(len(EVENT_TYPES))
    for name in names.tolist():
        index = TYPE_INDEX.get(name)
        if index is not None:
            counts[index] += 1
    return counts


def window_features(names, seconds, x, y):
    """One vector per window: [folch 12 | chu 12 | uitrace 23 | counts 43]."""
    return np.concatenate([
        folch_features(names, seconds, x, y),
        chu_features(names, seconds),
        uitrace_features(names, seconds, x, y),
        type_counts(names),
    ])


FEATURE_SLICES = {
    "folch": slice(0, 12),
    "chu": slice(12, 24),
    "uitrace": slice(24, 47),
    "counts": slice(47, 47 + len(EVENT_TYPES)),
    "all": slice(0, 47 + len(EVENT_TYPES)),
}
