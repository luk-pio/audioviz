import gc
from collections import defaultdict
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

# noinspection PyUnresolvedReferences
import src.common.log
from src.common.audioviz_datastore import (
    AbstractAudiovizDataStore,
    AudiovizDataStoreFactory,
)


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

    # @staticmethod
    # def load(name: str, store: AbstractAudiovizDataStore = None):
    #     config = dataset_config_factory(name)
    #     if store is None:
    #         file_prefix = config.get_name()
    #         file_path = os.path.join(DATA_INTERIM_DIR, file_prefix + "_data.h5")
    #         store = Hdf5AudiovizDataStore(file_path)
    #     data = store[]
    #     return AudiovizDataset(
    #         data=data,
    #         target=metadata["class"],
    #         name=name,
    #         target_names=config.get_classes(),
    #         data_filename=file_path,
    #         metadata_filename=metadata_path,
    #         metadata=metadata,
    #     )

    # @staticmethod
    # def dump(directory, fieldnames, files_metadata, root_dir, samples):
    #     data_path = os.path.join(root_dir, directory + "_data.h5")
    #     np.asarray(samples):
    #         logging.info(f"Saved {len(samples)} samples of {directory} to {data_path}")
    #     else:
    #         logging.error(
    #             f"Failed to save {len(samples)} samples of {directory} to {data_path}"
    #         )
    #     metadata_path = os.path.join(root_dir, directory + "_metadata.csv")
    #     try:
    #         write_dict_as_csv(metadata_path, files_metadata, fieldnames)
    #         logging.info(f"Wrote metadata to file {metadata_path}")
    #     except OSError:
    #         logging.exception(f"Could not write metadata file to {metadata_path}.")


def save_medley_solos_db(
    name: str,
    dataset: Dict[str, Any],
    path: str,
    metadata_keys: List[str],
    store: AbstractAudiovizDataStore = None,
):
    store = (
        AudiovizDataStoreFactory.get_instance(path, "h5") if store is None else store
    )

    subsets_mapping = {"training": 0, "test": 1, "validation": 2}
    subsets_list = ["training", "test", "validation"]
    classes = [
        "clarinet",
        "distorted electric guitar",
        "female singer",
        "flute",
        "piano",
        "tenor saxophone",
        "trumpet",
        "violin",
    ]
    samplerate = next(iter(dataset.items()))[1].audio[1]
    dataset_metadata = {
        "subsets_list": subsets_list,
        "classes": classes,
        "samplerate": samplerate,
    }

    samples = []
    files_metadata = defaultdict(list)
    sample_shape = (
        next(iter(dataset.items()))[1].audio[0].shape
    )  # Gets the length of the sample

    shape = (len(dataset), *sample_shape)

    with store:
        store._file.create_dataset(name, shape=shape)
    i = 0
    chunksize = 200
    for _, sample in dataset.items():
        samples.append(sample.audio[0])
        # Turn list of metadata dictionaries for each file into one dict with a list of vals
        for key in ["instrument_id", "subset", "song_id"]:
            value = getattr(sample, key)
            # Convert subset to nominal
            if key == "subset":
                value = subsets_mapping[value]
            files_metadata[key].append(value)
        i += 1
        if i % 100 == 0:
            print(f"Processed {i} samples")
        if i % chunksize == 0:
            with store:
                ds = store[name]
                ds[i - chunksize : i] = np.asarray(samples)
            print(f"Wrote chunk: [{i-chunksize}:{i}]")
        del samples
        gc.collect()
        samples = []
    with store:
        ds = store[name]
        remainder = i % chunksize
        ds[i - remainder : i] = np.asarray(samples)
        for k, v in files_metadata.items():
            store._file.create_dataset(k, data=v)
        for k, v in dataset_metadata.items():
            ds.attrs[k] = v

    del samples

    store.close()


# class AudiovizDatasetFactory:
#     implemented = {"medley-solos-db": create_medley_solos_ds}
#
#     @classmethod
#     def get_instance(cls, dataset: str):
#         try:
#             return cls.implemented[dataset]
#         except KeyError as err:
#             msg = f"The dataset {dataset} has not been implemented"
#             logging.exception(msg)
#             raise NotImplementedError(msg) from err
#
# @staticmethod
# def load(name):
#     config = dataset_config_factory(name)
#     file_prefix = config.get_name()
#     data_path = os.path.join(DATA_INTERIM_DIR, file_prefix + "_data.npy")
#     metadata_path = os.path.join(DATA_INTERIM_DIR, file_prefix + "_metadata.csv")
#     try:
#         data = np.load(data_path)
#     except IOError:
#         logging.error(
#             f"Could not read file at {data_path}. Have you run preprocessing for this dataset?"
#         )
#         raise
#     try:
#         metadata = pd.read_csv(metadata_path)
#     except IOError:
#         logging.error(
#             f"Could not read file at {metadata_path}. Have you run preprocessing for this dataset?"
#         )
#         raise
#     return AudiovizDataset(
#         data=data,
#         target=metadata["class"],
#         name=name,
#         target_names=config.get_classes(),
#         data_filename=data_path,
#         metadata_filename=metadata_path,
#         metadata=metadata,
#     )
#
# @staticmethod
# def dump(directory, fieldnames, files_metadata, root_dir, samples):
#     data_path = os.path.join(root_dir, directory + "_data.npy")
#     if write_ndarray(data_path, np.asarray(samples)):
#         logging.info(f"Saved {len(samples)} samples of {directory} to {data_path}")
#     else:
#         logging.error(
#             f"Failed to save {len(samples)} samples of {directory} to {data_path}"
#         )
#     metadata_path = os.path.join(root_dir, directory + "_metadata.csv")
#     try:
#         write_dict_as_csv(metadata_path, files_metadata, fieldnames)
#         logging.info(f"Wrote metadata to file {metadata_path}")
#     except OSError:
#         logging.exception(f"Could not write metadata file to {metadata_path}.")
