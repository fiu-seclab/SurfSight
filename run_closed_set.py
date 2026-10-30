"""Closed-set LSTM at 200 ms, with session-grouped splits.

Writes results/closed_set.json as soon as the five seeds finish. A window
grid at other lengths is not part of this run.
"""

import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping

from pipeline.config import (
    BATCH_SIZE,
    EPOCHS,
    LARGE_SCALE_WINDOW_MS,
    MAX_USERS_PER_CLASS,
    PATIENCE,
)
from pipeline.data import load_dataset
from pipeline.models import _build
from pipeline.splits import (
    ORDER,
    balance_test,
    balance_train,
    collect_windows,
    encode_from_train,
    evaluate_predictions,
    fmt_percent,
    fmt_score,
    mean_ci,
    split_sessions,
)

OUT = Path(__file__).resolve().parent / "results"
SEEDS = [0, 1, 2, 3, 4]


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _train(architecture, encoded, labels, seed):
    _set_seed(seed)
    model = _build(architecture, encoded["train"].shape[1:], labels["train"].shape[1])
    model.fit(
        encoded["train"],
        labels["train"],
        validation_data=(encoded["val"], labels["val"]),
        epochs=EPOCHS,
        batch_size=min(BATCH_SIZE, len(encoded["train"])),
        callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")],
        verbose=2,
    )
    return model


def _predict_labels(model, x_data):
    logits = model.predict(x_data, verbose=0)
    return logits


def build_all_windows(dataset, window_ms):
    windows = []
    counts = []
    for class_name in ORDER:
        class_windows, class_counts = collect_windows(dataset[class_name]["events_df"], window_ms, class_name)
        windows.extend(class_windows)
        counts.append(class_counts)
        print(
            f"windows {window_ms} ms {class_name}: any={class_counts['windows_any']} kept={class_counts['windows_ge5']} sessions={class_counts['sessions']}",
            flush=True,
        )
    return windows, counts


def run_one(windows, architecture, seed):
    parts = split_sessions(windows, seed)
    train = balance_train(parts["train"], seed, mode="mean")
    test_balanced = balance_test(parts["test"], seed)
    if len(train) < 4 or len(parts["val"]) < 1 or len(test_balanced) < 1:
        raise RuntimeError(f"split too small arch={architecture} seed={seed}")
    encoded, labels, _encoder = encode_from_train(
        train,
        {"val": parts["val"], "test": parts["test"], "test_balanced": test_balanced},
    )
    model = _train(architecture, encoded, labels, seed)
    pred_balanced = _predict_labels(model, encoded["test_balanced"])
    pred_natural = _predict_labels(model, encoded["test"])
    balanced = evaluate_predictions(labels["test_balanced"], pred_balanced)
    natural = evaluate_predictions(labels["test"], pred_natural)
    row = {
        "model": architecture,
        "window_ms": None,
        "seed": seed,
        "accuracy": balanced["accuracy"],
        "accuracy_natural": natural["accuracy"],
        "macro_f1": balanced["macro_f1"],
        "macro_f1_natural": natural["macro_f1"],
        "human_recall": balanced["human_recall"],
        "human_fpr": balanced["human_fpr"],
        "n_train": int(len(train)),
        "n_val": int(len(parts["val"])),
        "n_test": int(balanced["support"]),
        "n_test_natural": int(natural["support"]),
    }
    return row, balanced, natural, model


def _write_closed_set(seed_rows, balanced_rows, natural_rows):
    acc = [row["accuracy"] for row in seed_rows]
    f1 = [row["macro_f1"] for row in seed_rows]
    recall = [row["human_recall"] for row in seed_rows]
    fpr = [row["human_fpr"] for row in seed_rows]
    acc_mean, acc_half = mean_ci(acc)
    f1_mean, _f1_half = mean_ci(f1)
    recall_mean, _recall_half = mean_ci(recall)
    fpr_mean, _fpr_half = mean_ci(fpr)
    confusion = np.zeros((len(ORDER), len(ORDER)), dtype=np.int64)
    for metrics in balanced_rows:
        confusion += np.asarray(metrics["confusion"], dtype=np.int64)
    per_class = []
    for metrics in balanced_rows:
        per_class.append(metrics["per_class"])
    payload = {
        "n_windows": int(round(float(np.mean([row["n_test"] for row in seed_rows])))),
        "accuracy_mean": fmt_percent(acc_mean),
        "accuracy_ci": fmt_percent(acc_half / 100.0) if acc_half == acc_half else "",
        "macro_f1_mean": fmt_score(f1_mean),
        "human_recall": fmt_score(recall_mean),
        "human_fpr_percent": fmt_percent(fpr_mean),
        "accuracy_natural_mean": fmt_percent(mean_ci([row["accuracy_natural"] for row in seed_rows])[0]),
        "macro_f1_natural_mean": fmt_score(mean_ci([row["macro_f1_natural"] for row in seed_rows])[0]),
        "n_train_mean": int(round(float(np.mean([row["n_train"] for row in seed_rows])))),
        "n_test_natural_mean": int(round(float(np.mean([row["n_test_natural"] for row in seed_rows])))),
        "seeds": seed_rows,
        "per_class_by_seed": per_class,
        "confusion_matrix_sum": confusion.tolist(),
        "labels": ORDER,
        "natural_by_seed": [
            {"accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "per_class": row["per_class"]}
            for row in natural_rows
        ],
        "protocol": {
            "split": "session-grouped 70/15/15 before balancing",
            "balance": "training set only, sample to mean class size",
            "test_balanced": "down-sample without replacement",
            "window_ms": LARGE_SCALE_WINDOW_MS,
            "model": "lstm",
            "seeds": SEEDS,
            "max_users_per_class": MAX_USERS_PER_CLASS,
            "min_events": 5,
        },
        "git": _git_hash(),
    }
    # accuracy_ci is a half-width in percentage points. mean_ci returns the
    # half-width on the 0-1 scale, and fmt_percent multiplies by 100.
    payload["accuracy_ci"] = f"{100.0 * acc_half:.2f}" if acc_half == acc_half else ""
    path = OUT / "closed_set.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {path}", flush=True)
    return payload


def main():
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"gpus={gpus}", flush=True)
    print("Loading traces", flush=True)
    dataset = load_dataset()
    for class_name in ORDER:
        payload = dataset[class_name]
        print(f"loaded {class_name}: users={payload['n_users']} events={payload['n_events']}", flush=True)

    window_csv = OUT / "window_study.csv"
    filter_csv = OUT / "window_counts.csv"
    window_rows = []
    filter_rows = []
    if window_csv.exists():
        window_rows = pd.read_csv(window_csv).to_dict("records")
    done = {(row["model"], int(row["window_ms"]), int(row["seed"])) for row in window_rows}

    closed_rows = []
    closed_balanced = []
    closed_natural = []

    # The reported closed-set result is the 200 ms LSTM over five seeds.
    order = [LARGE_SCALE_WINDOW_MS]
    for window_ms in order:
        windows, counts = build_all_windows(dataset, window_ms)
        for count in counts:
            if not any(row.get("window_ms") == window_ms and row.get("class") == count["class"] for row in filter_rows):
                filter_rows.append(count)
        pd.DataFrame(filter_rows).to_csv(filter_csv, index=False)
        for architecture in ("lstm",):
            for seed in SEEDS:
                key = (architecture, int(window_ms), int(seed))
                if key in done:
                    print(f"skip existing {key}", flush=True)
                    continue
                print(f"train {architecture} {window_ms} ms seed {seed}", flush=True)
                row, balanced, natural, model = run_one(windows, architecture, seed)
                row["window_ms"] = int(window_ms)
                window_rows.append(row)
                pd.DataFrame(window_rows).to_csv(window_csv, index=False)
                print(
                    f"done {architecture} {window_ms} seed {seed} acc={row['accuracy']:.4f} natural={row['accuracy_natural']:.4f} n_test={row['n_test']}",
                    flush=True,
                )
                if architecture == "lstm" and window_ms == LARGE_SCALE_WINDOW_MS:
                    closed_rows.append(row)
                    closed_balanced.append(balanced)
                    closed_natural.append(natural)
                    _write_closed_set(closed_rows, closed_balanced, closed_natural)
                del model
                keras_backend.clear_session()
                gc.collect()
        del windows
        gc.collect()

    if not closed_rows:
        lstm_rows = [
            row
            for row in window_rows
            if row["model"] == "lstm" and int(row["window_ms"]) == LARGE_SCALE_WINDOW_MS
        ]
        if len(lstm_rows) == len(SEEDS) and not (OUT / "closed_set.json").exists():
            print("closed-set rows were already in the CSV; per-class JSON needs a fresh 200 ms run", flush=True)
    elapsed = time.time() - started
    print(f"closed set finished in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
