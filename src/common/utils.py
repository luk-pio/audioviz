from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "interim")
DATA_INTERIM_DIR = os.path.join(DATA_DIR, "interim")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

DATASET_CONFIGS_DIR = os.path.join(PROJECT_DIR, "dataset_configs")
