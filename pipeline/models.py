"""LSTM and 1D-CNN training for the window and closed-set experiments."""

import gc

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Input,
    LSTM,
    Masking,
    MaxPooling1D,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from pipeline.config import (
    BATCH_SIZE,
    EPOCHS,
    FEATURE_EVENTS,
    FEATURE_SETS,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    MIN_EVENTS_PER_SUBSESSION,
    PATIENCE,
    SESSION_GAP_MINUTES,
    TEST_SIZE,
    TIME_WINDOWS_MS,
)


def filter_events(events, feature_set):
    allowed = FEATURE_EVENTS[feature_set]
    if allowed is None or events.empty:
        return events
    return events[events["eventName"].isin(allowed)].copy()


def _with_coordinates(events):
    """Parse pointer positions once and keep them on the frame."""
    if events.empty or "x" in events.columns:
        return events
    import json

    x_pos = np.zeros(len(events), dtype=np.float32)
    y_pos = np.zeros(len(events), dtype=np.float32)
    for index, raw in enumerate(events["pos"].to_numpy()):
        if not isinstance(raw, str) or not raw:
            continue
        try:
            pos = json.loads(raw)
            x_pos[index] = float(pos.get("x") or 0)
            y_pos[index] = float(pos.get("y") or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    parsed = events.copy()
    parsed["x"] = x_pos
    parsed["y"] = y_pos
    return parsed


def _window_slices(events, window_ms, gap_minutes, min_events):
    """Yield each session window that has enough events, as aligned numpy columns."""
    ordered = events.sort_values(["userId", "timestamp_num"], kind="mergesort")
    user_ids = ordered["userId"].to_numpy()
    timestamps = ordered["timestamp_num"].to_numpy(dtype=np.int64)
    names = ordered["eventName"].to_numpy()
    x_pos = ordered["x"].to_numpy(dtype=np.float64)
    y_pos = ordered["y"].to_numpy(dtype=np.float64)
    if len(ordered) == 0:
        return
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
    for left, right in zip(starts, ends):
        if right - left < min_events:
            continue
        yield names[left:right], timestamps[left:right], x_pos[left:right], y_pos[left:right]


def sequences_from_dataset(
    dataset,
    window_ms,
    feature_set,
    min_events=MIN_EVENTS_PER_SUBSESSION,
    balance="min",
):
    sequences = []
    labels = []
    per_class = {}
    for class_name, payload in dataset.items():
        events = payload["events_df"]
        if "x" not in events.columns:
            events = _with_coordinates(events)
            payload["events_df"] = events
        events = filter_events(events, feature_set)
        if events.empty:
            per_class[class_name] = 0
            continue
        class_count = 0
        for names, timestamps, x_pos, y_pos in _window_slices(
            events, window_ms, SESSION_GAP_MINUTES, min_events
        ):
            start = float(timestamps[0])
            previous = np.empty(len(timestamps), dtype=np.float64)
            previous[0] = start
            previous[1:] = timestamps[:-1]
            relative = (timestamps.astype(np.float64) - start) / 1000.0
            delta = (timestamps.astype(np.float64) - previous) / 1000.0
            limit = min(len(timestamps), MAX_SEQUENCE_LENGTH)
            features = [
                [names[index], float(relative[index]), float(x_pos[index]), float(y_pos[index]), float(delta[index])]
                for index in range(limit)
            ]
            sequences.append(features)
            labels.append(class_name)
            class_count += 1
        per_class[class_name] = class_count
    if balance and labels:
        sequences, labels = _balance(sequences, labels, mode="min" if balance is True else balance)
        per_class = {}
        for label in labels:
            per_class[label] = per_class.get(label, 0) + 1
    return sequences, labels, per_class


def _balance(sequences, labels, mode="min", seed=42):
    """Equalize class counts.

    ``min`` downsamples to the smallest class (window comparison).
    ``mean`` matches the large-scale protocol: every class is sampled with
    replacement up or down to the mean class size.
    """
    rng = np.random.RandomState(seed)
    grouped = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    sizes = [len(indexes) for indexes in grouped.values()]
    if mode == "mean":
        cap = max(1, int(round(float(np.mean(sizes)))))
    elif mode == "min":
        cap = min(sizes)
    else:
        raise ValueError(mode)
    chosen = []
    for indexes in grouped.values():
        picked = rng.choice(indexes, size=cap, replace=len(indexes) < cap)
        chosen.extend(int(item) for item in picked)
    rng.shuffle(chosen)
    return [sequences[index] for index in chosen], [labels[index] for index in chosen]


def _pack(sequences, event_encoder, length):
    """One-hot event type, then the four continuous features.

    Index 0 is reserved for padding. A real event occupies its own column,
    so a padded step is all zeros and is not confused with an event id.
    """
    names = list(event_encoder.classes_)
    index_of = {name: index for index, name in enumerate(names)}
    width = len(names) + 4
    encoded = np.zeros((len(sequences), length, width), dtype=np.float32)
    for row, sequence in enumerate(sequences):
        for col, event in enumerate(sequence[:length]):
            event_index = index_of.get(event[0])
            if event_index is not None:
                encoded[row, col, event_index] = 1.0
            encoded[row, col, -4:] = event[1:]
    return encoded


def _scale_stats(encoded):
    stats = []
    for feature_index in range(encoded.shape[-1] - 4, encoded.shape[-1]):
        observed = encoded[:, :, feature_index]
        observed = observed[observed != 0]
        if len(observed) == 0:
            stats.append(None)
            continue
        low, high = np.percentile(observed, [5, 95])
        stats.append((float(low), float(high)) if high > low else None)
    return stats


def _apply_scale(encoded, stats):
    """Scale the four continuous columns. Leave padded steps at zero."""
    scaled = encoded.copy()
    start = encoded.shape[-1] - 4
    padding = encoded[:, :, :start].sum(axis=-1) == 0
    for offset, bounds in enumerate(stats):
        if bounds is None:
            continue
        low, high = bounds
        feature_index = start + offset
        column = np.clip(encoded[:, :, feature_index], low, high)
        column = (column - low) / (high - low)
        column[padding] = 0.0
        scaled[:, :, feature_index] = column
    return scaled


def _encode(sequences, labels):
    event_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    event_encoder.fit([event[0] for sequence in sequences for event in sequence])
    label_encoder.fit(labels)
    length = min(max(len(sequence) for sequence in sequences), MAX_SEQUENCE_LENGTH)
    encoded = _pack(sequences, event_encoder, length)
    stats = _scale_stats(encoded)
    encoded = _apply_scale(encoded, stats)
    y_index = label_encoder.transform(labels)
    y = to_categorical(y_index, num_classes=len(label_encoder.classes_))
    return encoded, y, label_encoder


def encode_known_and_unknown(known_sequences, known_labels, unknown_sequences):
    """Fit the event vocabulary on known sequences and encode both sets the same way."""
    event_encoder = LabelEncoder()
    label_encoder = LabelEncoder()
    event_encoder.fit([event[0] for sequence in known_sequences for event in sequence])
    label_encoder.fit(known_labels)
    length = min(max(len(sequence) for sequence in known_sequences), MAX_SEQUENCE_LENGTH)
    known = _pack(known_sequences, event_encoder, length)
    unknown = _pack(unknown_sequences, event_encoder, length)
    stats = _scale_stats(known)
    known = _apply_scale(known, stats)
    unknown = _apply_scale(unknown, stats)
    y_index = label_encoder.transform(known_labels)
    y = to_categorical(y_index, num_classes=len(label_encoder.classes_))
    return known, y, unknown, label_encoder


def _build(architecture, input_shape, num_classes):
    """Architectures used in the evaluation.

    LSTM: recurrent layers of 128 and 64 units, batch normalization, and dropout.
    1D-CNN: convolutions of 64, 128, and 256 filters with kernels 3, 5, and 7,
    then dense layers of 128 and 64. The layer named "features" is the
    penultimate representation used by the Mahalanobis open-set score.
    """
    inputs = Input(shape=input_shape)
    if architecture == "lstm":
        hidden = Masking(mask_value=0.0)(inputs)
        hidden = LSTM(128, return_sequences=True)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(0.2)(hidden)
        features = LSTM(64, name="features")(hidden)
        features = BatchNormalization()(features)
    elif architecture == "cnn":
        hidden = inputs
        for filters, kernel in ((64, 3), (128, 5), (256, 7)):
            hidden = Conv1D(filters, kernel, padding="same", activation="relu")(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = MaxPooling1D(2)(hidden)
            hidden = Dropout(0.2)(hidden)
        hidden = GlobalMaxPooling1D()(hidden)
        hidden = Dense(128, activation="relu")(hidden)
        hidden = Dropout(0.2)(hidden)
        features = Dense(64, activation="relu", name="features")(hidden)
    else:
        raise ValueError(architecture)
    logits = Dense(num_classes, name="logits")(Dropout(0.2)(features))
    model = Model(inputs, logits)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def _split(x_data, y_data):
    class_counts = y_data.argmax(axis=1)
    can_stratify = len(x_data) >= 8 and np.min(np.bincount(class_counts)) >= 2
    return train_test_split(
        x_data,
        y_data,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=class_counts if can_stratify else None,
    )


def train_and_evaluate(
    dataset,
    architecture="lstm",
    window_ms=200,
    feature_set="all_events",
    epochs=EPOCHS,
    bot_only=False,
    balance="min",
):
    """Train one model and return accuracy, macro F1, and per-class scores."""
    data = dataset
    if bot_only:
        data = {name: payload for name, payload in dataset.items() if name != "Human"}
    sequences, labels, per_class = sequences_from_dataset(
        data, window_ms, feature_set, balance=balance
    )
    result = {
        "architecture": architecture,
        "window_ms": window_ms,
        "feature_set": feature_set,
        "n_sequences": len(sequences),
        "sequences_per_class": per_class,
        "accuracy": None,
        "macro_f1": None,
        "per_class": {},
        "note": "",
    }
    smallest = min(per_class.values()) if per_class else 0
    if len(sequences) < 4 or len(set(labels)) < 2 or smallest < 30:
        result["note"] = "Not enough sequences for this window and feature set."
        return result

    x_data, y_data, label_encoder = _encode(sequences, labels)
    x_train, x_test, y_train, y_test = _split(x_data, y_data)
    if len(np.unique(y_train.argmax(axis=1))) < 2:
        result["note"] = "Training split collapsed to a single class."
        return result

    model = _build(architecture, x_data.shape[1:], y_data.shape[1])
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=min(BATCH_SIZE, len(x_train)),
        callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")],
        verbose=2,
    )
    probabilities = model.predict(x_test, verbose=0)
    true_class = y_test.argmax(axis=1)
    pred_class = probabilities.argmax(axis=1)
    present = sorted(set(true_class) | set(pred_class))
    names = [label_encoder.classes_[index] for index in present]
    report = classification_report(
        true_class,
        pred_class,
        labels=present,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    result["accuracy"] = float((true_class == pred_class).mean())
    result["macro_f1"] = float(report["macro avg"]["f1-score"])
    result["per_class"] = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": int(report[name]["support"]),
        }
        for name in names
    }
    result["model"] = model
    result["label_encoder"] = label_encoder
    return result


def sweep(
    dataset,
    architecture,
    windows=TIME_WINDOWS_MS,
    feature_sets=FEATURE_SETS,
    epochs=EPOCHS,
    bot_only=False,
    balance="min",
    checkpoint=None,
):
    """Train each requested window and feature set. Returns a flat results table."""
    rows = []
    details = []
    for feature_set in feature_sets:
        for window_ms in windows:
            print(f"{architecture} | {feature_set} | {window_ms} ms", flush=True)
            outcome = train_and_evaluate(
                dataset,
                architecture=architecture,
                window_ms=window_ms,
                feature_set=feature_set,
                epochs=epochs,
                bot_only=bot_only,
                balance=balance,
            )
            outcome.pop("model", None)
            outcome.pop("label_encoder", None)
            keras_backend.clear_session()
            gc.collect()
            details.append(outcome)
            row = {
                "architecture": architecture,
                "feature_set": feature_set,
                "window_ms": window_ms,
                "n_sequences": outcome["n_sequences"],
                "accuracy": outcome["accuracy"],
                "macro_f1": outcome["macro_f1"],
                "note": outcome["note"],
            }
            rows.append(row)
            if checkpoint is not None:
                pd.DataFrame(rows).to_csv(checkpoint, index=False)
            accuracy = outcome["accuracy"]
            shown = "n/a" if accuracy is None else f"{accuracy:.3f}"
            print(f"  sequences={outcome['n_sequences']} accuracy={shown} {outcome['note']}", flush=True)
    return pd.DataFrame(rows), details


def per_class_frame(details):
    """Per-class recall for each completed training run."""
    rows = []
    for outcome in details:
        for class_name, scores in outcome["per_class"].items():
            rows.append(
                {
                    "architecture": outcome["architecture"],
                    "feature_set": outcome["feature_set"],
                    "window_ms": outcome["window_ms"],
                    "class": class_name,
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                    "support": scores["support"],
                }
            )
    return pd.DataFrame(rows)
