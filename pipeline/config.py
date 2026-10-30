"""Evaluation protocol.

Compares an LSTM and a 1D-CNN at 50, 100, 200, 300, 500, 800, and 1000 ms.
Traces are the CSVs in data/corpus. The reported runs use up to 200 users
per class.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = PACKAGE_DIR.parent
TRACE_DIR = ANALYSIS_ROOT / "data" / "corpus"

# Up to 200 users per class, all of their sessions and events.
MAX_USERS_PER_CLASS = 200
MAX_SESSIONS_PER_USER = 10**9
MAX_EVENTS_PER_USER = 10**9

EPOCHS = 30
PATIENCE = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
TEST_SIZE = 0.2
MIN_EVENTS_PER_SUBSESSION = 5
MAX_SEQUENCE_LENGTH = 100

TIME_WINDOWS_MS = [50, 100, 200, 300, 500, 800, 1000]
# 200 ms is the operating point: LSTM accuracy is 98.61% on 44,163
# balanced sequences, against 98.39% on 44,296 sequences at 300 ms.
LARGE_SCALE_WINDOW_MS = 200

FEATURE_SETS = ("all_events", "mousemove_keypress", "mousemove_only")
FEATURE_EVENTS = {
    "all_events": None,
    "mousemove_keypress": {"mousemove", "keypress", "keydown", "keyup"},
    "mousemove_only": {"mousemove"},
}

# Paper classes. Human sources and the two random-mouse tools are merged.
CLASS_FILES = {
    "Browser-Use": ["browser_use_agents.csv"],
    "Gremlins": ["gremlins.csv"],
    "HLISA": ["hlisa_traces.csv"],
    "Human": ["tax1_human.csv", "tax2_human.csv", "survey1_human.csv", "survey2_human.csv"],
    "Random Mouse": ["random_mouse_bot.csv", "random_mouse_with_sleep_bot.csv"],
    "Skyvern": ["skyvern_agents.csv"],
    "ZA Proxy": ["za_proxy.csv"],
}

SESSION_GAP_MINUTES = 5
