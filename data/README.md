# Traces

Each class file is a CSV with four columns:

```text
userId,timestamp,pos,eventName
```

`timestamp` is milliseconds since that user's first event in the file, so the calendar date is not stored. `pos` is a JSON object with `x` and `y`. `eventName` is the DOM event type (`mousemove`, `keydown`, `click`, and the rest of the monitored set). The collector records event type, time, and position. It does not record which key was pressed.

The files in `corpus/` are short samples: the first two user ids in each source file, and at most 1,500 events per user. `corpus/sample_manifest.csv` is the count of what was kept. `corpus/full_corpus_manifest.csv` is the count of the corpus those samples were cut from (about 3.1 million events across the eleven files). The tables in `results/` were computed on that full corpus, with the user cap in `pipeline/config.py` (`MAX_USERS_PER_CLASS = 200`), not on these samples. `results/corpus_counts.csv` is the user, session, and window count by source.

Human rows are a format sample only. Those logs were collected under an IRB waiver, and the full human files are not in this repository.
