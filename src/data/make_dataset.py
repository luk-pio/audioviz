# -*- coding: utf-8 -*-
import csv
import importlib
import logging
import os
import sys
from collections import defaultdict
from functools import partial
from typing import Tuple, List, Dict, Optional, Any, Callable

import click
import librosa
import mirdata
import numpy as np

# noinspection PyUnresolvedReferences
import src.common.log

from src.common.audioviz_dataset import save_medley_solos_db

from src.common.utils import (
    DATA_INTERIM_DIR,
    DATA_RAW_DIR,
    DATASET_CONFIGS_DIR,
)
from src.common.dataset_config import dataset_config_factory


def find_files_recursive(
    root: str, file_types: List[str] = None
) -> Dict[str, List[str]]:
    """
    Recursively walk through subdirectories of root and construct a dictionary of directory_name: List[filenames] pairs
    If file_types is given, filter files by extensions in file_types.

    :param root: The root directory
    :param file_types: List of extensions to find
    :return:
    """
    if file_types is None:
        file_types = [""]
    # Collect all of the sample file paths under root as strings and group them by directory
    directory_file_list_dict = defaultdict(list)
    total_files = 0
    for dirpath, _, filenames in os.walk(root):
        dirname = os.path.split(dirpath)[1]
        for filename in filenames:
            # Ignore the file if it doesn't have the required extension
            if any([filename.endswith(file_type) for file_type in file_types]):
                total_files += 1
                directory_file_list_dict[dirname].append(
                    os.path.join(dirpath, filename)
                )
    logging.info(f"Found {total_files} files under {root}.")
    return directory_file_list_dict


def load_sample(
    fp: str,
    metadata_extractor: Callable[[str], Dict[str, str]] = None,
    sr: int = None,
    duration: float = None,
    normalize: bool = True,
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Load a sample as a one-dimensional np.ndarray. If sample has more than one track, it will be converted to mono.
    Returns None if file could not be read.

    :param metadata_extractor: function used for extracting metadata from the filename
    :param fp: Path of the file.
    :param sr: Samplerate to load the sample at. If left as None, the files original samplerate is used.
    :param duration: Max sample duration in seconds. If None, array will equal original file length.
    If sample is longer, it will be cropped. If shorter, the array will be padded with zeros.
    :param normalize: If True, magnitude will be normalized to a value between 0 and 1.
    :return: a tuple containing (filename, np.array, metadata_dict)
    """

    if fp == "":  # ignore empty filenames
        return fp, None, None
    try:
        audio, sr = librosa.load(fp, sr, mono=True, duration=duration)
    except OSError as e:
        logging.error(f"File {fp} could not be read: {e}")
        return fp, None, None
    file_length = len(audio)
    if file_length == 0:  # ignore zero-length samples
        logging.info(f"File {fp} is of length 0")
        return fp, None, None
    if duration:
        audio.resize(int(duration * sr))
    max_val = np.abs(audio).max()
    if max_val == 0:  # ignore completely silent sounds
        logging.info(f"File {fp} is silent")
        return fp, None, None
    if normalize:
        audio /= max_val
    return (
        fp,
        audio,
        dict(
            duration=duration, samplerate=sr, **metadata_extractor(os.path.basename(fp))
        ),
    )


def load_samples(
    file_list: List[str],
    metadata_extractor: Callable[[str], Dict[str, str]],
    sr: int = None,
    duration: float = None,
    normalize: bool = True,
    processes: int = None,
    limit: int = None,
) -> List[Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]]:
    """
    Load (filename, np.array, metadata_dict) tuples for all files in file_list.

    :param file_list: the list of file paths
    :param metadata_extractor: function for extracting metadata form the filename
    :param sr: Samplerate to load the sample at. If left as None, the file samplerate is used.
    :param duration: Max sample duration in seconds. If None, array will equal original file length.
    If sample is longer, it will be cropped. If shorter, the array will be padded with zeros.
    :param normalize: If true magnitude will be normalized to a value between 0 and 1.
    :param processes: The number of worker processes to use. If None use os.cpu_count() cores.
    :param limit: The limit of files to load in each directory.
    :return: List of (filename, np.array, metadata_dict)
    """

    # TODO multiprocessing

    job = partial(
        load_sample,
        metadata_extractor=metadata_extractor,
        sr=sr,
        duration=duration,
        normalize=normalize,
    )
    # pool = Pool(processes)
    # processed_files = pool.map(
    #     job,
    #     ( file_list[:limit] )
    # )
    # logging.info(f'Processed {len(file_list)}')
    # return processed_files
    return [job(fn) for fn in file_list[:limit]]


def load_samples_recursive(
    root: str,
    metadata_extractor: Callable[[str], Dict[str, str]],
    file_types: List[str] = None,
    sr: int = None,
    duration: float = None,
    normalize: bool = True,
    processes: int = None,
    limit: int = None,
) -> Dict[str, List[Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]]]:
    """
    Recursively load (filename, np.array, metadata_dict) tuples for all files in directory tree starting at root.

    :param root: root of the directory tree
    :param metadata_extractor: function for extracting metadata form the filename
    :param file_types: list of suffixes (file extensions) to load
    :param sr: Samplerate to load the sample at. If left as None, the file samplerate is used.
    :param duration: Max sample duration in seconds. If None, array will equal original file length.
    If sample is longer, it will be cropped. If shorter, the array will be padded with zeros.
    :param normalize: If true magnitude will be normalized to a value between 0 and 1.
    :param processes: The number of worker processes to use. If None use os.cpu_count() cores.
    :param limit: The limit of files to load in each directory.
    :return: Dictionary of directories to lists of (filename, np.array, metadata_dict)
    """
    logging.info(f"Attempting to load files from {root}")
    directory_file_list_dict = find_files_recursive(root, file_types=file_types)
    return {
        directory: load_samples(
            files, metadata_extractor, sr, duration, normalize, processes, limit
        )
        for directory, files in directory_file_list_dict.items()
    }


@click.command()
@click.option("--download", default=False)
@click.option("--verify", default=False)
@click.option("-input_dir", default=DATA_RAW_DIR, type=click.Path(exists=True))
@click.option("-output_dir", default=DATA_INTERIM_DIR, type=click.Path())
@click.option("-dataset", default="medley_solos_db")
@click.option("-processes", default=None, type=click.INT)
@click.option("-file_limit", default=None, type=click.INT)
@click.option(
    "-dataset_configs_directory",
    default=DATASET_CONFIGS_DIR,
    type=click.Path(exists=True),
)
def main(
    download,
    verify,
    input_dir,
    output_dir,
    dataset,
    processes,
    file_limit,
    dataset_configs_directory,
):
    """
    Runs data processing scripts to turn raw data from (../raw) into
    preprocessed data ready for feature extraction (saved in ../features).

    Parameters
    ----------
    input_dir: directory of the raw dataset
    output_dir: directory in which to save output files
    dataset: string id of the dataset. As specified in dataset config
    processes: number of processes to utilize for loading samples
    file_limit: The maximum number of files to load from each directory
    dataset_configs_directory: the path to the dataset configs directory

    Returns
    -------
    None

    """

    datasets_savers = {"medley_solos_db": save_medley_solos_db}
    mirdata_module = importlib.import_module(f"mirdata.{dataset}")
    if download:
        logging.info("Downloading dataset...")
        mirdata_module.download(data_home=input_dir)
    if verify:
        logging.info("Validating dataset...")
        mirdata_module.validate(data_home=input_dir)
    loaded = mirdata_module.load(data_home=input_dir)
    logging.info("Loaded dataset metadata.")

    output_path = os.path.join(output_dir, f"{ dataset }_data.h5")
    logging.info("Saving dataset to " + output_path)

    datasets_savers[dataset](
        dataset, loaded, output_path, ["instrument_id", "subset", "song_id"]
    )


if __name__ == "__main__":
    main()
