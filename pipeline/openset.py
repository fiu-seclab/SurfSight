"""Leave-one-class-out open-set scores.

Scores are Maximum Softmax Probability, temperature-scaled MSP (T=2),
energy (T=1), and Mahalanobis distance in the penultimate layer.
A larger score means the subsession looks more like a known class.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.metrics import f1_score, roc_auc_score

import gc

from tensorflow.keras import backend as keras_backend
from tensorflow.keras.callbacks import EarlyStopping

from pipeline.config import BATCH_SIZE, EPOCHS, LARGE_SCALE_WINDOW_MS, PATIENCE
from pipeline.models import _build, _split, encode_known_and_unknown, sequences_from_dataset


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _mahalanobis(features, means, precision):
    delta = features[:, None, :] - means[None, :, :]
    distances = np.einsum("nkd,de,nke->nk", delta, precision, delta)
    return -distances.min(axis=1)


def _best_unknown_f1(is_known, score):
    unknown = (~is_known).astype(int)
    low, high = np.percentile(score, [1, 99])
    if high <= low:
        thresholds = np.unique(score)
    else:
        thresholds = np.linspace(low, high, 100)
    best = 0.0
    for threshold in thresholds:
        predicted_unknown = (score < threshold).astype(int)
        best = max(best, f1_score(unknown, predicted_unknown, zero_division=0))
    return float(best)


def _fit_known(x_known, y_known, epochs):
    x_train, x_test, y_train, y_test = _split(x_known, y_known)
    if len(np.unique(y_train.argmax(axis=1))) < 2 or len(x_test) == 0:
        return None
    model = _build("lstm", x_known.shape[1:], y_known.shape[1])
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=min(BATCH_SIZE, len(x_train)),
        callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")],
        verbose=2,
    )
    feature_model = None
    for layer in model.layers:
        if layer.name == "features":
            from tensorflow.keras.models import Model

            feature_model = Model(model.input, layer.output)
            break
    train_prob = model.predict(x_train, verbose=0)
    train_features = feature_model.predict(x_train, verbose=0)
    train_labels = y_train.argmax(axis=1)
    means = []
    for class_index in range(y_known.shape[1]):
        members = train_features[train_labels == class_index]
        if len(members) == 0:
            means.append(np.zeros(train_features.shape[1]))
        else:
            means.append(members.mean(axis=0))
    means = np.vstack(means)
    centered = train_features - means[train_labels]
    covariance = np.cov(centered, rowvar=False)
    if np.ndim(covariance) == 0:
        covariance = np.array([[float(covariance)]])
    covariance = covariance + 1e-3 * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(covariance)
    return {
        "model": model,
        "feature_model": feature_model,
        "x_test": x_test,
        "means": means,
        "precision": precision,
        "train_prob": train_prob,
    }


def _score_batch(fitted, x_data):
    logits = fitted["model"].predict(x_data, verbose=0)
    probabilities = _softmax(logits)
    temperature = 2.0
    softened = _softmax(logits / temperature)
    features = fitted["feature_model"].predict(x_data, verbose=0)
    return {
        "MSP": probabilities.max(axis=1),
        "TempScale": softened.max(axis=1),
        "Energy": logsumexp(logits, axis=1),
        "Mahalanobis": _mahalanobis(features, fitted["means"], fitted["precision"]),
    }


def leave_one_class_out(
    dataset,
    window_ms=LARGE_SCALE_WINDOW_MS,
    feature_set="all_events",
    epochs=EPOCHS,
    checkpoint=None,
):
    """Hold out each class in turn and score known-versus-unknown separation."""
    sequences, labels, per_class = sequences_from_dataset(dataset, window_ms, feature_set)
    print("Sequences per class:", per_class)
    rows = []
    classes = [name for name, count in per_class.items() if count > 0]
    for held_out in classes:
        known_seq = [seq for seq, label in zip(sequences, labels) if label != held_out]
        known_y = [label for label in labels if label != held_out]
        unknown_seq = [seq for seq, label in zip(sequences, labels) if label == held_out]
        row = {"unknown": held_out, "known_sequences": len(known_seq), "unknown_sequences": len(unknown_seq)}
        if len(unknown_seq) == 0 or len(set(known_y)) < 2 or len(known_seq) < 4:
            row["note"] = "Skipped: not enough known or unknown sequences."
            rows.append(row)
            print(held_out, row["note"])
            continue
        x_known, y_known, x_unknown, _encoder = encode_known_and_unknown(known_seq, known_y, unknown_seq)
        fitted = _fit_known(x_known, y_known, epochs)
        if fitted is None:
            row["note"] = "Skipped: known-class split was too small."
            rows.append(row)
            print(held_out, row["note"])
            continue
        known_scores = _score_batch(fitted, fitted["x_test"])
        unknown_scores = _score_batch(fitted, x_unknown)
        for method in ("MSP", "TempScale", "Energy", "Mahalanobis"):
            scores = np.concatenate([known_scores[method], unknown_scores[method]])
            is_known = np.concatenate(
                [
                    np.ones(len(known_scores[method]), dtype=bool),
                    np.zeros(len(unknown_scores[method]), dtype=bool),
                ]
            )
            if is_known.all() or (~is_known).all() or np.allclose(scores, scores[0]):
                auroc, f1 = float("nan"), float("nan")
            else:
                auroc = float(roc_auc_score(is_known.astype(int), scores))
                f1 = _best_unknown_f1(is_known, scores)
            row[f"{method}_auroc"] = auroc
            row[f"{method}_f1"] = f1
        row["note"] = ""
        rows.append(row)
        print(
            f"{held_out}: "
            + " ".join(
                f"{method} AUROC={row.get(f'{method}_auroc', float('nan')):.3f}"
                for method in ("MSP", "TempScale", "Energy", "Mahalanobis")
            ),
            flush=True,
        )
        del fitted
        keras_backend.clear_session()
        gc.collect()
        if checkpoint is not None:
            pd.DataFrame(rows).to_csv(checkpoint, index=False)
    table = pd.DataFrame(rows)
    if checkpoint is not None:
        table.to_csv(checkpoint, index=False)
    return table
