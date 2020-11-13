import pickle
from pathlib import Path
import os

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "interim")
DATA_INTERIM_DIR = os.path.join(DATA_DIR, "interim")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

DATASET_CONFIGS_DIR = os.path.join(PROJECT_DIR, "dataset_configs")


def write_ndarray(path: str, array: np.ndarray, overwrite=False) -> bool:
    """
    Wrapper for writing writing numpy array to file
    """
    if os.path.isfile(path):
        if overwrite or str.lower(
            input(f"There is already a file at {path}. Overwrite? [y/n]")
        ) in ("yes", "y", "t"):
            np.save(path, array)
            return True
    return False


def stringify_funcall(func, *args, **kwargs):
    # Use protocol = 0 for ascii encoded bytes object:
    # https://stackoverflow.com/questions/30469575/how-to-pickle-and-unpickle-to-portable-string-in-python-3
    return pickle.dumps((func, args, kwargs), protocol=0).decode("ASCII")


def unpickle_funcall(string: str):
    return pickle.loads(bytes(string, "ASCII"))
