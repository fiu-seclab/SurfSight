"""Figures for the evaluation.

Reads the tables in results/ and writes PDFs to results/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results"
OUT_DIRS = [
    ROOT / "results" / "figures",
]

# Okabe–Ito, distinguishable in print and for the common color-vision deficiencies.
COLORS = {
    "Browser-Use": "#0072B2",
    "Gremlins": "#E69F00",
    "HLISA": "#009E73",
    "Random Mouse": "#D55E00",
    "Skyvern": "#CC79A7",
    "ZA Proxy": "#56B4E9",
    "Human": "#000000",
}
MARKERS = {
    "Browser-Use": "o",
    "Gremlins": "s",
    "HLISA": "^",
    "Random Mouse": "D",
    "Skyvern": "v",
    "ZA Proxy": "P",
}
LINESTYLES = {
    "Browser-Use": "-",
    "Gremlins": "--",
    "HLISA": "-.",
    "Random Mouse": ":",
    "Skyvern": (0, (3, 1, 1, 1)),
    "ZA Proxy": (0, (5, 1)),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def _save(figure, name):
    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)


def bot_recall():
    """Per-class recall of the bot-only LSTM by window.

    Same canvas, fonts, line weights, and markers as window_model_figure, so the
    two line figures read as a pair when both are included at 0.85 of the text width.
    """
    import csv
    from collections import OrderedDict

    series = OrderedDict()
    with open(RUN / "bots_per_class_recall.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            series.setdefault(row["class"], []).append((int(row["window_ms"]), float(row["recall"]) * 100))
    figure, axis = plt.subplots(figsize=(4.6, 2.5), constrained_layout=True)
    windows = [50, 100, 200, 300, 500, 800, 1000]
    positions = list(range(len(windows)))
    for class_name, points in series.items():
        ordered = [recall for _, recall in sorted(points)]
        axis.plot(
            positions,
            ordered,
            color=COLORS[class_name],
            marker=MARKERS[class_name],
            linestyle=LINESTYLES[class_name],
            linewidth=1.3,
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=class_name,
        )
    axis.set_xlabel("Subsession window (ms)")
    axis.set_ylabel("Recall (%)")
    axis.set_xticks(positions)
    axis.set_xticklabels([str(window) for window in windows])
    axis.set_ylim(92, 100.6)
    axis.set_yticks([92, 94, 96, 98, 100])
    axis.yaxis.grid(True, linewidth=0.4, color="#d0d0d0")
    axis.set_axisbelow(True)
    axis.legend(loc="lower right", frameon=False, ncol=3, handlelength=2.2, columnspacing=0.9)
    _save(figure, "bots_per_class_accuracy_vs_time_window.pdf")


CLASS_ORDER = ["Browser-Use", "Gremlins", "HLISA", "Human", "Random Mouse", "Skyvern", "ZA Proxy"]


def event_type_bars():
    """Per-class F1 of the 200 ms LSTM by event types kept (zoomed dot plot).

    Pointer, keyboard, and pointer-plus-keyboard share a zoomed F1 axis, joined by
    a line per class. Keyboard alone sits in its own panel on the full 0 to 1 scale.
    Those three conditions are one session split (results/event_mask.json).
    All events is the mean F1 over the five closed-set seeds (results/closed_set.json).
    """
    import json

    import numpy as np
    from matplotlib.gridspec import GridSpec

    mask = {row["mask"]: row for row in json.load(open(ROOT / "results" / "event_mask.json"))["rows"]}
    closed = json.load(open(ROOT / "results" / "closed_set.json"))

    def per_class(name):
        return np.array([mask[name]["per_class"][c]["f1"] for c in CLASS_ORDER])

    pointer = per_class("Pointer")
    both = per_class("Pointer and keyboard")
    keyboard = per_class("Keyboard")
    full = np.array([np.mean([seed[c]["f1"] for seed in closed["per_class_by_seed"]]) for c in CLASS_ORDER])
    accuracy = {
        "Pointer": mask["Pointer"]["accuracy"] * 100,
        "Pointer + keyboard": mask["Pointer and keyboard"]["accuracy"] * 100,
        "All events": float(closed["accuracy_mean"]),
        "Keyboard": mask["Keyboard"]["accuracy"] * 100,
    }

    rows = len(CLASS_ORDER)
    positions = np.arange(rows)
    figure = plt.figure(figsize=(5.4, 2.9), constrained_layout=True)
    grid = GridSpec(1, 2, figure=figure, width_ratios=[5.2, 1.7], wspace=0.04)
    main = figure.add_subplot(grid[0, 0])
    keys = figure.add_subplot(grid[0, 1], sharey=main)

    # Main panel: one line per class from the lowest to the highest condition.
    low = np.minimum.reduce([pointer, both, full])
    high = np.maximum.reduce([pointer, both, full])
    main.hlines(positions, low, high, color="#9a9a9a", linewidth=1.4, zorder=2)
    styles = [
        ("Pointer", pointer, "o", "#9ECAE1"),
        ("Pointer + keyboard", both, "s", "#4292C6"),
        ("All events", full, "D", "#08519C"),
    ]
    for name, values, marker, color in styles:
        main.scatter(
            values,
            positions,
            s=26,
            marker=marker,
            color=color,
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
            label=f"{name} ({accuracy[name]:.1f}%)",
        )
    main.set_xlim(0.87, 1.005)
    main.set_xticks([0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    main.set_xlabel("F1")
    main.set_ylim(rows - 0.5, -0.5)
    main.set_yticks(positions)
    main.set_yticklabels(CLASS_ORDER)
    main.xaxis.grid(True, linewidth=0.4, color="#d0d0d0", zorder=0)
    main.set_axisbelow(True)
    main.tick_params(axis="y", length=0)
    main.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=3,
        handletextpad=0.2,
        columnspacing=0.8,
        borderaxespad=0.2,
        fontsize=7,
    )

    # Keyboard only, on the full scale.
    keys.barh(positions, keyboard, height=0.5, color="#BDBDBD", zorder=3)
    for row, value in zip(positions, keyboard):
        keys.text(value + 0.04, row, f"{value:.2f}", va="center", fontsize=7)
    keys.set_xlim(0, 1.3)
    keys.set_xticks([0, 0.5, 1.0])
    keys.set_xlabel("F1")
    keys.tick_params(axis="y", length=0, labelleft=False)
    keys.xaxis.grid(True, linewidth=0.4, color="#d0d0d0", zorder=0)
    keys.set_axisbelow(True)
    keys.spines["left"].set_visible(False)
    keys.set_title(f"Keyboard only ({accuracy['Keyboard']:.1f}%)", fontsize=7, pad=4)
    _save(figure, "event_type_f1_bars.pdf")
# Geometry of the closed-set matrix, in inches. The LaTeX table beside it in
# content/03-Methodology.tex uses CM_PITCH and CM_TOP for its row heights, so
# change them together. The figure is saved at its exact size (no tight crop).
CM_WIDTH = 4.04
CM_PITCH = 0.34
CM_TOP = 0.72
CM_LEFT = 1.0
CM_BOTTOM = 0.08


def closed_set_matrix():
    """Row-normalized confusion matrix of the full-stream LSTM, pooled over five seeds.

    Per-label precision, recall, and F1 sit beside this figure in the paper,
    with rows aligned to the matrix rows. Source: results/closed_set.json.
    """
    import json

    import numpy as np
    from matplotlib.colors import PowerNorm

    closed = json.load(open(ROOT / "results" / "closed_set.json"))
    labels = closed["labels"]
    counts = np.array(closed["confusion_matrix_sum"], dtype=float)
    percent = counts / counts.sum(axis=1, keepdims=True) * 100
    wrap = {"Browser-Use": "Browser-\nUse", "Random Mouse": "Random\nMouse", "ZA Proxy": "ZA\nProxy"}
    columns = [wrap.get(name, name) for name in labels]

    rows = len(labels)
    height = CM_TOP + rows * CM_PITCH + CM_BOTTOM
    figure = plt.figure(figsize=(CM_WIDTH, height))
    axis = figure.add_axes(
        [CM_LEFT / CM_WIDTH, CM_BOTTOM / height, (CM_WIDTH - CM_LEFT - 0.03) / CM_WIDTH, rows * CM_PITCH / height]
    )
    cmap = plt.get_cmap("Blues")
    norm = PowerNorm(gamma=0.45, vmin=0, vmax=100)
    for row in range(rows):
        for col in range(rows):
            level = norm(percent[row, col])
            axis.add_patch(
                plt.Rectangle((col - 0.5, row - 0.5), 1, 1, facecolor=cmap(level), edgecolor="white", linewidth=0.8)
            )
            axis.text(
                col,
                row,
                f"{percent[row, col]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if level >= 0.62 else "#1a1a1a",
            )
    axis.set_xlim(-0.5, rows - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)
    axis.xaxis.tick_top()
    axis.xaxis.set_label_position("top")
    axis.set_xticks(range(rows))
    axis.set_xticklabels(columns)
    axis.set_yticks(range(rows))
    axis.set_yticklabels(labels)
    axis.set_xlabel("Predicted label (% of true class)", labelpad=9)
    axis.set_ylabel("True label")
    axis.tick_params(length=0, pad=2)
    axis.tick_params(axis="x", labelsize=7)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for directory in OUT_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
        figure.savefig(directory / "lstm_large_scale.pdf")
    plt.close(figure)


# Known-versus-Unknown AUROC for leave-one-class-out (Table "Open-set recognition"
# in content/03-Methodology.tex). Columns: MSP, TempScale, Energy, Mahalanobis.
OSR_AUROC = {
    "Browser-Use": [0.853, 0.865, 0.854, 0.843],
    "Gremlins": [0.690, 0.700, 0.650, 0.953],
    "HLISA": [0.635, 0.653, 0.719, 0.656],
    "Human": [0.751, 0.761, 0.768, 0.909],
    "Random Mouse": [0.717, 0.723, 0.624, 0.800],
    "Skyvern": [0.794, 0.810, 0.892, 0.743],
    "ZA Proxy": [0.812, 0.823, 0.875, 0.983],
    "Average": [0.750, 0.762, 0.769, 0.841],
}


def osr_auroc_dots():
    """Known-versus-Unknown AUROC per held-out class, one marker per scorer.

    Rows are sorted by the Mahalanobis AUROC; the average is its own row below a
    rule. The logit-based scorers (MSP, TempScale, Energy) use light markers and
    Mahalanobis is the dark diamond. Values are the ones in the appendix table.
    """
    import numpy as np

    rows = sorted((n for n in OSR_AUROC if n != "Average"), key=lambda n: -OSR_AUROC[n][3])
    rows.append("Average")
    positions = np.arange(len(rows))
    positions = np.where(positions == len(rows) - 1, positions + 0.45, positions)
    values = np.array([OSR_AUROC[name] for name in rows])

    figure, axis = plt.subplots(figsize=(5.4, 2.9), constrained_layout=True)
    axis.hlines(positions, values.min(axis=1), values.max(axis=1), color="#9a9a9a", linewidth=1.4, zorder=2)
    styles = [
        ("MSP", 0, "o", "#C6DBEF", 26),
        ("TempScale", 1, "s", "#9ECAE1", 26),
        ("Energy", 2, "^", "#6BAED6", 30),
        ("Mahalanobis", 3, "D", "#08306B", 38),
    ]
    for name, column, marker, color, size in styles:
        axis.scatter(
            values[:, column],
            positions,
            s=size,
            marker=marker,
            color=color,
            edgecolor="#222222",
            linewidth=0.5,
            zorder=3,
            label=name,
        )
    axis.axhline((positions[-2] + positions[-1]) / 2, color="#1a1a1a", linewidth=0.8)
    axis.set_xlim(0.55, 1.0)
    axis.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
    axis.set_xlabel("AUROC (Known versus Unknown)")
    axis.set_ylim(positions[-1] + 0.5, -0.5)
    axis.set_yticks(positions)
    axis.set_yticklabels(rows)
    axis.get_yticklabels()[-1].set_fontweight("bold")
    axis.xaxis.grid(True, linewidth=0.4, color="#d0d0d0", zorder=0)
    axis.set_axisbelow(True)
    axis.tick_params(axis="y", length=0)
    axis.set_ylabel("Held-out (unknown) class")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        ncol=4,
        handletextpad=0.2,
        columnspacing=0.9,
        borderaxespad=0.2,
        fontsize=7,
    )
    _save(figure, "osr_auroc_dots.pdf")


# All-events accuracy (%) by window for the window-level split used to pick the
# operating point (Table "All-events classification accuracy" in the appendix).
WINDOWS_MS = [50, 100, 200, 300, 500, 800, 1000]
WINDOW_SEQUENCES = [18662, 33859, 44163, 44296, 35441, 27055, 23317]
WINDOW_ACC = {
    "LSTM": [96.25, 97.98, 98.61, 98.39, 98.41, 98.78, 97.98],
    "1D-CNN": [94.83, 97.18, 98.11, 98.30, 98.53, 98.63, 98.71],
}


def window_model_figure():
    figure, axis = plt.subplots(figsize=(4.6, 2.5), constrained_layout=True)
    bars = axis.twinx()
    positions = list(range(len(WINDOWS_MS)))
    bars.bar(positions, [n / 1000 for n in WINDOW_SEQUENCES], width=0.55, color="#e3e3e3", zorder=0)
    bars.set_ylim(0, 110)
    bars.set_ylabel("Balanced sequences (thousands)", color="#707070")
    bars.tick_params(axis="y", colors="#707070")
    bars.spines["right"].set_visible(True)
    bars.spines["right"].set_color("#b0b0b0")
    # The accuracy axis is drawn above the bars.
    axis.set_zorder(bars.get_zorder() + 1)
    axis.patch.set_visible(False)
    styles = {"LSTM": ("#0072B2", "o", "-"), "1D-CNN": ("#D55E00", "s", "--")}
    for name, values in WINDOW_ACC.items():
        color, marker, linestyle = styles[name]
        axis.plot(
            positions,
            values,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=1.3,
            markersize=4,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=name,
        )
    selected = WINDOWS_MS.index(200)
    axis.text(selected, WINDOW_ACC["LSTM"][selected] + 0.2, "selected", ha="center", va="bottom", fontsize=7)
    axis.set_xticks(positions)
    axis.set_xticklabels([str(w) for w in WINDOWS_MS])
    axis.set_xlabel("Subsession window (ms)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(94, 99.4)
    axis.yaxis.grid(True, linewidth=0.4, color="#d0d0d0")
    axis.legend(loc="lower right", frameon=False, ncol=2, handlelength=2.2)
    _save(figure, "window_model_comparison.pdf")


def main():
    bot_recall()
    event_type_bars()
    closed_set_matrix()
    osr_auroc_dots()
    window_model_figure()
    print("Wrote figures")


if __name__ == "__main__":
    main()
