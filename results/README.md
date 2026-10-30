# Reported results

These tables are the session-grouped evaluation. Every window from one session stays in one split. The operating window is 200 ms, and a window is kept when it has at least five events. They were computed on the full corpus, not on the samples in `data/corpus/`.

| File | What it is |
| --- | --- |
| `closed_set.json` | 200 ms LSTM over five seeds: accuracy, per-class scores, and the pooled confusion counts. |
| `window_study.csv` | The same splits, one row per seed. LSTM seeds 0–4 and 1D-CNN seeds 0–3. |
| `window_counts.csv` | Sessions and kept 200 ms windows per class. |
| `event_mask.json` | One session split. Accuracy after keeping pointer events, keyboard events, both, or all events. |
| `baselines.json` | Five session splits. Decision tree, gradient boosting, and random forest. |
| `time_to_verdict.json` | Delay from the first event of a session to the close of the first kept 200 ms window. |
| `corpus_counts.csv` | Users, sessions, and kept windows by class and collection page. |
| `bots_per_class_recall.csv` | Bot-only LSTM recall at each window length. No Human class. |

`figures/` holds the plots used in the paper: the window comparison, the bot-only recall curves, the event-type comparison, the closed-set confusion matrix, and the open-set AUROC chart.
