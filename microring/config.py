"""Project paths and acquisition defaults."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LEGACY_RECORD_CSV = PROJECT_ROOT / "legacy" / "record_old.csv"
RECORDS_CSV = DATA_DIR / "records.csv"
ANALYSIS_RUNS_CSV = DATA_DIR / "analysis_runs.csv"
ALLOW_LOCAL_FILE_IMPORT = os.environ.get("MICRORING_ALLOW_LOCAL_FILE_IMPORT", "0") == "1"

START_NM = 1260.0
END_NM = 1360.0
SWEEP_RANGE_NM = END_NM - START_NM
NUM_ROWS = 2_000_000
TIME_AMOUNT = 64
SWEEP_RATE = 2.0
DATA_LENGTH = int(NUM_ROWS / TIME_AMOUNT * SWEEP_RANGE_NM / SWEEP_RATE)
