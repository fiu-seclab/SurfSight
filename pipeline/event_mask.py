"""Retrain the closed-set LSTM after masking event types inside fixed windows.

Windows are the same 200 ms bins as the large-scale evaluation: built on the
full event stream, kept only when they contain at least five events, and
split by session with seed 0. The mask then drops events whose type is
outside the allowed set. A window that becomes empty is kept and padded, so
every condition is scored on the same windows.
"""

import gc
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping

from pipeline.config import BATCH_SIZE, EPOCHS, LARGE_SCALE_WINDOW_MS, PATIENCE
from pipeline.data import load_dataset
from pipeline.models import _build
from pipeline.splits import (
    ORDER,
    balance_test,
    balance_train,
    collect_windows,
    encode_from_train,
    evaluate_predictions,
)


def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)

POINTER = {
    "mousedown",
    "mouseup",
    "mousemove",
    "click",
    "dblclick",
    "wheel",
    "mousewheel",
    "touchstart",
    "touchend",
    "touchmove",
}
KEYBOARD = {"keydown", "keyup", "keypress"}
MASKS = (
    ("All events", None),
    ("Pointer and keyboard", POINTER | KEYBOARD),
    ("Pointer", POINTER),
    ("Keyboard", KEYBOARD),
)


def build_windows(dataset):
    windows = []
    for class_name in ORDER:
        class_windows, counts = collect_windows(
            dataset[class_name]["events_df"], LARGE_SCALE_WINDOW_MS, class_name
        )
        windows.extend(class_windows)
        print(
            f"windows {class_name}: kept={counts['windows_ge5']} sessions={counts['sessions']}",
            flush=True,
        )
    return windows


def split_sessions_stable(windows, seed):
    """Same session split as splits.split_sessions, with sorted keys.

    A set of session ids has no stable order across processes. Sorting
    before the shuffle makes seed 0 select the same sessions every run.
    """
    by_class = {}
    for session_key, label, _features in windows:
        by_class.setdefault(label, set()).add(session_key)
    return _split_sorted(windows, seed, by_class)


def _split_sorted(windows, seed, by_class):
    rng = np.random.RandomState(seed)
    assignment = {}
    for label in ORDER:
        keys = sorted(by_class.get(label, ()))
        rng.shuffle(keys)
        n = len(keys)
        n_test = int(round(0.15 * n))
        n_val = int(round(0.15 * n))
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


def apply_mask(windows, allowed):
    masked = []
    empty = 0
    lengths = []
    for session_key, label, features in windows:
        if allowed is None:
            kept = list(features)
        else:
            kept = [event for event in features if event[0] in allowed]
        if not kept:
            empty += 1
        lengths.append(len(kept))
        masked.append((session_key, label, kept))
    return masked, empty, lengths


def _train(encoded, labels, seed):
    _set_seed(seed)
    model = _build("lstm", encoded["train"].shape[1:], labels["train"].shape[1])
    history = model.fit(
        encoded["train"],
        labels["train"],
        validation_data=(encoded["val"], labels["val"]),
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, len(encoded["train"])),
        callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")],
        verbose=2,
    )
    ran = len(history.history.get("loss", []))
    return model, ran


def run_mask(windows, name, allowed, seed=0):
    parts = split_sessions_stable(windows, seed)
    train = balance_train(parts["train"], seed, mode="mean")
    test_balanced = balance_test(parts["test"], seed)
    train_m, train_empty, _train_lengths = apply_mask(train, allowed)
    val_m, _val_empty, _val_lengths = apply_mask(parts["val"], allowed)
    test_m, test_empty, test_lengths = apply_mask(test_balanced, allowed)
    natural_m, _natural_empty, _natural_lengths = apply_mask(parts["test"], allowed)
    n_events = sum(len(item[2]) for item in train_m)
    print(
        f"{name}: train_windows={len(train_m)} train_events={n_events} "
        f"empty_balanced_test={test_empty}/{len(test_m)}",
        flush=True,
    )
    if n_events == 0:
        raise RuntimeError(f"{name} left the training set with no events")
    encoded, labels, _encoder = encode_from_train(
        train_m,
        {"val": val_m, "test": natural_m, "test_balanced": test_m},
    )
    print(
        f"{name}: encoded train {encoded['train'].shape} balanced test {encoded['test_balanced'].shape}",
        flush=True,
    )
    started = time.time()
    model, epochs_ran = _train(encoded, labels, seed)
    pred = model.predict(encoded["test_balanced"], verbose=0)
    balanced = evaluate_predictions(labels["test_balanced"], pred)
    row = {
        "mask": name,
        "accuracy": balanced["accuracy"],
        "macro_f1": balanced["macro_f1"],
        "human_recall": balanced["human_recall"],
        "n_test": int(balanced["support"]),
        "empty_test": int(test_empty),
        "empty_train": int(train_empty),
        "median_events": float(np.median(test_lengths)),
        "epochs_ran": int(epochs_ran),
        "seconds": round(time.time() - started, 1),
        "per_class": balanced["per_class"],
    }
    print(
        f"{name}: acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} "
        f"epochs={epochs_ran} seconds={row['seconds']}",
        flush=True,
    )
    del model, encoded, labels, pred
    keras_backend.clear_session()
    gc.collect()
    return row


def main():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"gpus={gpus}", flush=True)
    dataset = load_dataset()
    windows = build_windows(dataset)
    del dataset
    gc.collect()
    rows = []
    out_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.environ.get("MASK_OUT", "event_mask.json"))
    selected = {item.strip() for item in os.environ.get("MASK_ONLY", "").split(",") if item.strip()}
    masks = tuple(item for item in MASKS if not selected or item[0] in selected)
    print(f"masks={[name for name, _allowed in masks]} out={out_path}", flush=True)
    for name, allowed in masks:
        row = run_mask(windows, name, allowed, seed=0)
        rows.append(row)
        payload = {
            "seed": 0,
            "window_ms": LARGE_SCALE_WINDOW_MS,
            "architecture": "lstm",
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "rows": rows,
        }
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {out_path}", flush=True)
    return rows


if __name__ == "__main__":
    main()
