# SurfSight: Online Attribution of Web Automation and Autonomous Browser Agents via Behavioral Signals

SurfSight decides, from a short stretch of in-page activity, whether a visitor is a person or one of six automation families. Browser-Use and Skyvern are LLM browser agents. Gremlins, HLISA, Random Mouse, and ZA Proxy are scripted tools. The decision uses only what a page can observe: the event type, the time since the window opened, the pointer position, and the gap since the previous event. It does not read the key that was pressed, the page URL, or a browser fingerprint.

A visit is split into sessions wherever the visitor is idle for five minutes. Each session is cut into non-overlapping 200 ms windows, and a window is kept when it holds at least five events. An LSTM reads that sequence and names the family. The same windows, summarized as fixed features, are the input to a decision tree, gradient boosting, and a random forest. A score on the LSTM representation can also flag a family that was held out of training. Every window from one session stays in a single split, so a model cannot train and test on two pieces of the same visit.

## Repository

```text
pipeline/          windowing, splits, models, baselines, figures
run_closed_set.py  LSTM evaluation, five seeds
run_baselines.py   tree baselines on the same windows
data/corpus/       sample traces (see data/README.md)
```

`pipeline/config.py` holds the protocol: the 200 ms window, the five-event minimum, the five-minute session gap, and the cap of 200 users per class. `pipeline/event_mask.py` retrains the LSTM on pointer events, keyboard events, or both. `pipeline/time_to_verdict.py` measures how long a session runs before the first kept window closes. `pipeline/make_figures.py` redraws the figures.

## Running it

Python 3.8 and the packages in `requirements.txt`. TensorFlow 2.10 does not install on newer Python. From this directory:

```bash
python run_closed_set.py
python -m pipeline.event_mask
python run_baselines.py
python -m pipeline.time_to_verdict
python -m pipeline.make_figures
```

The CSVs in `data/corpus/` are samples: two visitors per source, at most 1,500 events each, with timestamps counted from that visitor's first event. Human traces were collected under an IRB waiver and are included here only as a format sample.
