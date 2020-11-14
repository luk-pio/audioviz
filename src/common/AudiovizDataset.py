import logging
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from src.common.dataset_config import dataset_config_factory
from src.common.utils import DATA_INTERIM_DIR

# noinspection PyUnresolvedReferences
import src.common.log


class AudiovizDataset(Bunch):
    """
    Represents a dataset. In essence a dictionary, which can also be accessed through attributes.
    Follows sckikit-learn conventions:

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : dataframe
            The data matrix.
        target: Series
            The classification target.
        target_names: list
            The names of target classes.
        frame: DataFrame
            DataFrame with `data` and `target`.
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        metadata_filename: str
            The path to the location of the data.
        metadata: dict
            pd.Dataframe of any remaining metadata
    """

    def __init__(
        self,
        data: np.ndarray,
        target: pd.DataFrame,
        name: str,
        target_names: List[str] = None,
        data_filename: str = None,
        metadata_filename: str = None,
        metadata: pd.DataFrame = None,
    ):
        if len(data.shape) < 2 or data.shape[0] < 1 or data.shape[1] < 1:
            raise ValueError("AudiovizDataset data must be a non-empty 2d ndarray.")
        if not name:
            raise ValueError("AudiovizDataset must have a non-empty name.")
        if target is None or target.size != data.shape[0]:
            raise ValueError(
                "AudiovizDataset must have a target dataframe with length equal to number datapoints."
            )

        super().__init__(
            data=data,
            target=target,
            name=name,
            target_names=target_names,
            data_filename=data_filename,
            metadata_filename=metadata_filename,
            metadata=metadata,
        )


def load_audioviz_dataset(name):
    config = dataset_config_factory(name)
    file_prefix = config.get_name()
    data_path = os.path.join(DATA_INTERIM_DIR, file_prefix + "_data.npy")
    metadata_path = os.path.join(DATA_INTERIM_DIR, file_prefix + "_metadata.csv")
    try:
        data = np.load(data_path)
    except IOError:
        logging.error(
            f"Could not read file at {data_path}. Have you run preprocessing for this dataset?"
        )
        raise
    try:
        metadata = pd.read_csv(metadata_path)
    except IOError:
        logging.error(
            f"Could not read file at {metadata_path}. Have you run preprocessing for this dataset?"
        )
        raise
    return AudiovizDataset(
        data=data,
        target=metadata["class"],
        name=name,
        target_names=config.get_classes(),
        data_filename=data_path,
        metadata_filename=metadata_path,
        metadata=metadata,
    )


def load_medley_solos_db():
    load_audioviz_dataset("medley_solos_db")
