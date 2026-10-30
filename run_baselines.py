"""Tree baselines on the same 200 ms windows and session splits as the LSTM.

Windows are featurized once and cached. Each seed uses the same session-grouped
70/15/15 split, training-set balancing, and balanced test set as the LSTM
runs. Models are scikit-learn only, so no GPU is needed.
"""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from pipeline.baselines import FEATURE_SLICES, iter_windows, window_features
from pipeline.config import LARGE_SCALE_WINDOW_MS
from pipeline.data import load_dataset
from pipeline.event_mask import split_sessions_stable
from pipeline.splits import ORDER, balance_test, balance_train, evaluate_predictions

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data" / "features_200ms.npz"
OUT = ROOT / "results"
SEEDS = [0, 1, 2, 3, 4]
N_JOBS = int(os.environ.get("N_JOBS", "4"))

# name -> (feature block, model factory). Each entry follows one cited study.
MODELS = {
    "Folch-style (GBDT, pointer kinematics)": ("folch", lambda s: HistGradientBoostingClassifier(max_iter=200, random_state=s)),
    "Chu-style (decision tree, timing entropy)": ("chu", lambda s: DecisionTreeClassifier(criterion="entropy", min_samples_leaf=10, random_state=s)),
    "UI-trace-style (GBDT, interval and click features)": ("uitrace", lambda s: HistGradientBoostingClassifier(max_iter=200, random_state=s)),
    "Random forest (depth 10)": ("all", lambda s: RandomForestClassifier(n_estimators=200, criterion="gini", max_features="sqrt", max_depth=10, n_jobs=N_JOBS, random_state=s)),
    "Random forest (no depth cap)": ("all", lambda s: RandomForestClassifier(n_estimators=200, criterion="gini", max_features="sqrt", n_jobs=N_JOBS, random_state=s)),
}


def build_features():
    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        return data["keys"].tolist(), data["labels"].tolist(), data["features"]
    dataset = load_dataset()
    keys, labels, rows = [], [], []
    for class_name in ORDER:
        count = 0
        for key, names, seconds, x, y in iter_windows(dataset[class_name]["events_df"], LARGE_SCALE_WINDOW_MS, class_name):
            keys.append(key)
            labels.append(class_name)
            rows.append(window_features(names, seconds, x, y))
            count += 1
        print(f"featurized {class_name}: {count} windows", flush=True)
        dataset[class_name]["events_df"] = None
    features = np.asarray(rows, dtype=np.float32)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, keys=np.array(keys, dtype=object), labels=np.array(labels, dtype=object), features=features)
    return keys, labels, features


def one_hot(index, n=len(ORDER)):
    out = np.zeros((len(index), n), dtype=np.float32)
    out[np.arange(len(index)), index] = 1.0
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keys, labels, features = build_features()
    print(f"windows={len(keys)} features={features.shape[1]}", flush=True)
    class_index = {name: i for i, name in enumerate(ORDER)}
    windows = [(keys[i], labels[i], i) for i in range(len(keys))]
    only = {item.strip() for item in os.environ.get("BASELINES", "").split("|") if item.strip()}
    rows = []
    for seed in SEEDS:
        parts = split_sessions_stable(windows, seed)
        train = balance_train(parts["train"], seed, mode="mean")
        test_balanced = balance_test(parts["test"], seed)
        natural = parts["test"]
        sets = {"train": train, "balanced": test_balanced, "natural": natural}
        index = {name: np.array([item[2] for item in items]) for name, items in sets.items()}
        y = {name: np.array([class_index[item[1]] for item in items]) for name, items in sets.items()}
        for name, (block, factory) in MODELS.items():
            if only and name not in only:
                continue
            cols = FEATURE_SLICES[block]
            started = time.time()
            model = factory(seed)
            model.fit(features[index["train"]][:, cols], y["train"])
            result = {}
            for split in ("balanced", "natural"):
                proba = model.predict_proba(features[index[split]][:, cols])
                result[split] = evaluate_predictions(one_hot(y[split]), proba)
            row = {
                "model": name,
                "seed": seed,
                "accuracy": result["balanced"]["accuracy"],
                "macro_f1": result["balanced"]["macro_f1"],
                "human_recall": result["balanced"]["human_recall"],
                "accuracy_natural": result["natural"]["accuracy"],
                "macro_f1_natural": result["natural"]["macro_f1"],
                "n_train": int(len(train)),
                "n_test": int(result["balanced"]["support"]),
                "seconds": round(time.time() - started, 1),
                "per_class": result["balanced"]["per_class"],
                "confusion": result["balanced"]["confusion"],
            }
            rows.append(row)
            print(f"seed {seed} {name}: acc={row['accuracy']:.4f} f1={row['macro_f1']:.4f} natural={row['accuracy_natural']:.4f} ({row['seconds']}s)", flush=True)
            (OUT / "baselines.json").write_text(json.dumps({"window_ms": LARGE_SCALE_WINDOW_MS, "seeds": SEEDS, "rows": rows}, indent=1))
    return rows


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
