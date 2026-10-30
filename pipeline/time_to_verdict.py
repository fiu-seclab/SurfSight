"""Time from session start to the first 200 ms window with at least five events.

Uses the same user cap, five-minute session gap, and non-overlapping bins as
the closed-set pipeline. Does not train a model.
"""

import numpy as np
import pandas as pd

from pipeline.config import MIN_EVENTS_PER_SUBSESSION, SESSION_GAP_MINUTES
from pipeline.data import load_dataset

WINDOW_MS = 200


def session_times(events, window_ms=WINDOW_MS, min_events=MIN_EVENTS_PER_SUBSESSION, gap_minutes=SESSION_GAP_MINUTES):
    """Return, per session, the delay in ms to the close of the first kept window.

    A session with no kept window contributes None.
    """
    if events is None or len(events) == 0:
        return []
    ordered = events.sort_values(["userId", "timestamp_num"], kind="mergesort")
    user_ids = ordered["userId"].to_numpy()
    timestamps = ordered["timestamp_num"].to_numpy(dtype=np.int64)
    gap_ms = gap_minutes * 60 * 1000
    new_user = np.empty(len(ordered), dtype=bool)
    new_user[0] = True
    if len(ordered) > 1:
        new_user[1:] = user_ids[1:] != user_ids[:-1]
    gap = np.zeros(len(ordered), dtype=bool)
    if len(ordered) > 1:
        gap[1:] = (timestamps[1:] - timestamps[:-1]) > gap_ms
        gap[new_user] = False
    session_id = np.cumsum(new_user | gap)
    session_start = pd.Series(timestamps).groupby(session_id, sort=False).transform("min").to_numpy()
    window_id = (timestamps - session_start) // window_ms
    change = np.empty(len(ordered), dtype=bool)
    change[0] = True
    if len(ordered) > 1:
        change[1:] = (session_id[1:] != session_id[:-1]) | (window_id[1:] != window_id[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(ordered))
    first = {}
    seen = set()
    for left, right in zip(starts, ends):
        local = int(session_id[left])
        seen.add(local)
        if right - left < min_events or local in first:
            continue
        # The bin closes one window after its start, measured from the session start.
        close_ms = (int(window_id[left]) + 1) * window_ms
        first[local] = close_ms
    return [first.get(local) for local in seen]


def summarize(delays):
    present = np.array([item for item in delays if item is not None], dtype=np.float64)
    n = len(delays)
    covered = len(present)
    if covered == 0:
        return {"sessions": n, "covered": 0, "coverage": 0.0}
    seconds = present / 1000.0
    return {
        "sessions": n,
        "covered": covered,
        "coverage": covered / n,
        "median_s": float(np.median(seconds)),
        "p90_s": float(np.percentile(seconds, 90)),
        "share_at_200ms": float(np.mean(present <= WINDOW_MS)),
    }


def main():
    dataset = load_dataset()
    rows = []
    for class_name, payload in dataset.items():
        delays = session_times(payload["events_df"])
        row = {"class": class_name, **summarize(delays)}
        rows.append(row)
        print(row, flush=True)
    all_delays = []
    for class_name, payload in dataset.items():
        all_delays.extend(session_times(payload["events_df"]))
    print("ALL", summarize(all_delays), flush=True)
    human = session_times(dataset["Human"]["events_df"])
    print("HUMAN", summarize(human), flush=True)


if __name__ == "__main__":
    main()
