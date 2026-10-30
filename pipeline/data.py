"""Load the local SurfSight traces with a hard cap on users and sessions."""

import csv
from pathlib import Path

import pandas as pd

from pipeline.config import (
    CLASS_FILES,
    MAX_EVENTS_PER_USER,
    MAX_SESSIONS_PER_USER,
    MAX_USERS_PER_CLASS,
    SESSION_GAP_MINUTES,
    TRACE_DIR,
)


def _read_capped_events(path, max_users, max_events_per_user):
    """Read the first users in a CSV and stop once each has enough events."""
    counts = {}
    order = []
    rows = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            user_id = row["userId"]
            if user_id not in counts:
                if len(order) >= max_users:
                    continue
                order.append(user_id)
                counts[user_id] = 0
            if counts[user_id] >= max_events_per_user:
                # Keep reading until every requested user has a full event cap.
                # A long first user must not end the file before the next user starts.
                if len(order) >= max_users and all(counts[uid] >= max_events_per_user for uid in order):
                    break
                continue
            counts[user_id] += 1
            rows.append(
                {
                    "userId": user_id,
                    "timestamp_num": int(float(row["timestamp"])),
                    "pos": row["pos"],
                    "eventName": row["eventName"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["userId", "timestamp_num", "pos", "eventName"])
    return pd.DataFrame(rows)


def _keep_first_sessions(events, max_sessions, gap_minutes):
    if events.empty:
        return events
    frames = []
    gap_ms = gap_minutes * 60 * 1000
    for _, user_events in events.groupby("userId", sort=False):
        ordered = user_events.sort_values("timestamp_num")
        gaps = ordered["timestamp_num"].diff().fillna(0) > gap_ms
        session_id = gaps.cumsum()
        frames.append(ordered[session_id < max_sessions])
    return pd.concat(frames, ignore_index=True) if frames else events.iloc[0:0]


def load_dataset(
    max_users_per_class=MAX_USERS_PER_CLASS,
    max_sessions_per_user=MAX_SESSIONS_PER_USER,
    max_events_per_user=MAX_EVENTS_PER_USER,
    trace_dir=TRACE_DIR,
    classes=None,
):
    """Return {class_name: {events_df, n_users, n_events}} for the seven classes."""
    trace_dir = Path(trace_dir)
    selected = classes or list(CLASS_FILES)
    dataset = {}
    for class_name in selected:
        frames = []
        filenames = CLASS_FILES[class_name]
        base, extra = divmod(int(max_users_per_class), len(filenames))
        leftover = 0
        for index, filename in enumerate(filenames):
            take = base + (1 if index < extra else 0) + leftover
            leftover = 0
            if take <= 0:
                continue
            path = trace_dir / filename
            if not path.exists():
                raise FileNotFoundError(path)
            frame = _read_capped_events(path, take, max_events_per_user)
            if frame.empty:
                leftover = take
                continue
            frames.append(frame)
            leftover = take - int(frame["userId"].nunique())
        if not frames:
            dataset[class_name] = {
                "events_df": pd.DataFrame(columns=["userId", "timestamp_num", "pos", "eventName"]),
                "n_users": 0,
                "n_events": 0,
            }
            continue
        events = _keep_first_sessions(
            pd.concat(frames, ignore_index=True),
            max_sessions_per_user,
            SESSION_GAP_MINUTES,
        )
        dataset[class_name] = {
            "events_df": events,
            "n_users": int(events["userId"].nunique()) if not events.empty else 0,
            "n_events": int(len(events)),
        }
    return dataset


def summarize(dataset):
    rows = []
    for class_name, payload in dataset.items():
        rows.append(
            {
                "class": class_name,
                "users": payload["n_users"],
                "events": payload["n_events"],
            }
        )
    return pd.DataFrame(rows)
