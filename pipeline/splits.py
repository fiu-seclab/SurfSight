"""Session-grouped splits.

Windows of one session stay in one split. Balancing is applied to the
training set only. Encoders and scaling statistics are fit on training
windows only.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from pipeline.config import MAX_SEQUENCE_LENGTH, MIN_EVENTS_PER_SUBSESSION, SESSION_GAP_MINUTES
from pipeline.models import _apply_scale, _pack, _scale_stats, _with_coordinates

ORDER = [
    "Browser-Use",
    "Gremlins",
    "HLISA",
    "Human",
    "Random Mouse",
    "Skyvern",
    "ZA Proxy",
]


def collect_windows(events, window_ms, class_name, min_events=MIN_EVENTS_PER_SUBSESSION, gap_minutes=SESSION_GAP_MINUTES):
    """Return windows with a session key, plus counts before and after the length filter.

    The session key is the class, the source user or bot run, and the
    gap-delimited session index. Splits use this key so every window of a
    session lands in the same partition.
    """
    empty = {
        "class": class_name,
        "window_ms": window_ms,
        "windows_any": 0,
        "windows_ge5": 0,
        "sessions": 0,
    }
    if events is None or len(events) == 0:
        return [], empty
    if "x" not in events.columns:
        events = _with_coordinates(events)
    ordered = events.sort_values(["userId", "timestamp_num"], kind="mergesort")
    user_ids = ordered["userId"].to_numpy()
    timestamps = ordered["timestamp_num"].to_numpy(dtype=np.int64)
    names = ordered["eventName"].to_numpy()
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
    windows = []
    sessions = set()
    kept = 0
    for left, right in zip(starts, ends):
        length = int(right - left)
        if length <= 0:
            continue
        user_id = str(user_ids[left])
        local = int(session_local[left])
        session_key = f"{class_name}\t{user_id}\t{local}"
        sessions.add(session_key)
        if length < min_events:
            continue
        kept += 1
        ts = timestamps[left:right]
        limit = min(length, MAX_SEQUENCE_LENGTH)
        start = float(ts[0])
        previous = np.empty(length, dtype=np.float64)
        previous[0] = start
        if length > 1:
            previous[1:] = ts[:-1]
        relative = (ts.astype(np.float64) - start) / 1000.0
        delta = (ts.astype(np.float64) - previous) / 1000.0
        features = [
            [str(names[left + index]), float(relative[index]), float(x_pos[left + index]), float(y_pos[left + index]), float(delta[index])]
            for index in range(limit)
        ]
        windows.append((session_key, class_name, features))
    counts = {
        "class": class_name,
        "window_ms": int(window_ms),
        "windows_any": int(len(starts)),
        "windows_ge5": int(kept),
        "sessions": int(len(sessions)),
    }
    return windows, counts


def split_sessions(windows, seed, train_frac=0.70, val_frac=0.15):
    """Assign every session to train, val, or test. Stratify by class."""
    by_class = {}
    for session_key, label, _features in windows:
        by_class.setdefault(label, set()).add(session_key)
    rng = np.random.RandomState(seed)
    assignment = {}
    for label, keys in by_class.items():
        keys = list(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(round(0.15 * n))
        n_val = int(round(val_frac * n))
        if n >= 3:
            n_test = max(1, n_test)
            n_val = max(1, n_val)
            while n_test + n_val >= n:
                if n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
        elif n == 2:
            n_test, n_val = 1, 0
        else:
            n_test, n_val = 0, 0
        for key in keys[:n_test]:
            assignment[key] = "test"
        for key in keys[n_test : n_test + n_val]:
            assignment[key] = "val"
        for key in keys[n_test + n_val :]:
            assignment[key] = "train"
    parts = {"train": [], "val": [], "test": []}
    for item in windows:
        parts[assignment[item[0]]].append(item)
    return parts


def _group_indices(labels):
    grouped = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    return grouped


def balance_train(windows, seed, mode="mean"):
    """Sample training windows to the mean class size. Replacement only when a class is short."""
    rng = np.random.RandomState(seed)
    labels = [item[1] for item in windows]
    grouped = _group_indices(labels)
    sizes = [len(indexes) for indexes in grouped.values()]
    if not sizes:
        return []
    cap = max(1, int(round(float(np.mean(sizes))))) if mode == "mean" else min(sizes)
    chosen = []
    for indexes in grouped.values():
        replace = len(indexes) < cap
        picked = rng.choice(indexes, size=cap, replace=replace)
        chosen.extend(int(item) for item in picked)
    rng.shuffle(chosen)
    return [windows[index] for index in chosen]


def balance_test(windows, seed):
    """Down-sample the test set without replacement to the smallest class."""
    rng = np.random.RandomState(seed + 10007)
    labels = [item[1] for item in windows]
    grouped = _group_indices(labels)
    if not grouped:
        return []
    cap = min(len(indexes) for indexes in grouped.values())
    if cap <= 0:
        return []
    chosen = []
    for indexes in grouped.values():
        picked = rng.choice(indexes, size=cap, replace=False)
        chosen.extend(int(item) for item in picked)
    rng.shuffle(chosen)
    return [windows[index] for index in chosen]


def encode_from_train(train_windows, other_sets):
    """Fit the event vocabulary and scaling on training windows, then encode every set."""
    train_seq = [item[2] for item in train_windows]
    event_encoder = LabelEncoder()
    event_encoder.fit([event[0] for sequence in train_seq for event in sequence])
    label_encoder = LabelEncoder()
    label_encoder.fit(ORDER)
    length = min(max(len(sequence) for sequence in train_seq), MAX_SEQUENCE_LENGTH)
    train_x = _pack(train_seq, event_encoder, length)
    stats_ = _scale_stats(train_x)
    encoded = {"train": _apply_scale(train_x, stats_)}
    for name, windows in other_sets.items():
        sequences = [item[2] for item in windows]
        packed = _pack(sequences, event_encoder, length) if sequences else np.zeros((0, length, train_x.shape[-1]), dtype=np.float32)
        encoded[name] = _apply_scale(packed, stats_) if len(sequences) else packed
    labels = {}
    labels["train"] = to_categorical(label_encoder.transform([item[1] for item in train_windows]), num_classes=len(ORDER))
    for name, windows in other_sets.items():
        if windows:
            labels[name] = to_categorical(label_encoder.transform([item[1] for item in windows]), num_classes=len(ORDER))
        else:
            labels[name] = np.zeros((0, len(ORDER)), dtype=np.float32)
    return encoded, labels, label_encoder


def evaluate_predictions(y_true, y_pred):
    true_class = np.asarray(y_true).argmax(axis=1)
    pred_class = np.asarray(y_pred).argmax(axis=1)
    names = ORDER
    report = classification_report(
        true_class,
        pred_class,
        labels=list(range(len(names))),
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_class, pred_class, labels=list(range(len(names))))
    human_index = names.index("Human")
    human_support = int(matrix[human_index].sum())
    human_correct = int(matrix[human_index, human_index])
    human_recall = float(human_correct / human_support) if human_support else float("nan")
    human_fpr = float(1.0 - human_recall) if human_support else float("nan")
    return {
        "accuracy": float((true_class == pred_class).mean()) if len(true_class) else float("nan"),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "human_recall": human_recall,
        "human_fpr": human_fpr,
        "support": int(len(true_class)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in names
        },
        "confusion": matrix.tolist(),
    }


def mean_ci(values):
    """Mean and 95% t-interval half-width. Values are on the 0-1 scale."""
    clean = [float(value) for value in values if value == value]
    n = len(clean)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(clean))
    if n < 2:
        return mean, float("nan")
    half = float(stats.t.ppf(0.975, n - 1) * np.std(clean, ddof=1) / np.sqrt(n))
    return mean, half


def fmt_percent(value):
    return f"{100.0 * value:.2f}"


def fmt_score(value):
    return f"{value:.3f}"
