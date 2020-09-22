# -*- coding: utf-8 -*-
import csv
import logging
import os
from collections import defaultdict
from functools import partial
from typing import Tuple, List, Dict, Optional, Any, Callable

import click
import librosa
import numpy as np

from src.common.utils import PROJECT_DIR
from src.data.dataset_config import dataset_config_factory


def find_files(root: str, file_types: List[str] = None) -> Dict[str, List[str]]:
    """
    Recursively find all files in a directory and organize them into a dictionary where keys are names of parent
    directories and values are filenames. If file_types is given, filter files by extensions in file_types.

    :param root: The root directory
    :param file_types: List of extensions to find
    :return:
    """
    if file_types is None:
        file_types = ['']
    # Collect all of the sample file paths as strings
    directory_file_list_dict = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        dirname = os.path.split(dirpath)[1]
        for filename in filenames:
            if any([filename.endswith(file_type) for file_type in file_types]):
                directory_file_list_dict[dirname].append(os.path.join(dirpath, filename))
    return directory_file_list_dict


def load_sample(fp: str, metadata_extractor: Callable[[str], Dict[str, str]] = None, sr: int = None,
                duration: float = None, normalize: bool = True) \
        -> Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Load a sample as a one-dimensional np.ArrayLike. If sample has more than one track, it will be converted to mono.
    Returns None as the np.array if file could not be read.

    :param metadata_extractor:
    :param fp: Path of the file.
    :param sr: Samplerate to load the sample at. If left as None, the file samplerate is used.
    :param duration: Max sample duration in seconds. If None, array will equal original file length.
    If sample is longer, it will be cropped. If shorter, the array will be padded with zeros.
    :param normalize: If true magnitude will be normalized to a value between 0 and 1.
    :return: a tuple containing (filename, np.array, metadata_dict)
    """

    if fp == '':  # ignore empty filenames
        return fp, None, None
    try:
        audio, sr = librosa.load(fp, sr, mono=True, duration=duration)
    except OSError as e:
        logging.error(f'File {fp} could not be read: {e}')
        return fp, None, None
    file_length = len(audio)
    if file_length == 0:  # ignore zero-length samples
        return fp, None, None
    if duration:
        audio.resize(int(duration * sr))
    max_val = np.abs(audio).max()
    if max_val == 0:  # ignore completely silent sounds
        return fp, None, None
    if normalize:
        audio /= max_val
    return fp, audio, dict(duration=duration, samplerate=sr, **metadata_extractor(os.path.basename(fp)))


def load_samples(
        file_list: List[str],
        metadata_extractor: Callable[[str], Dict[str, str]],
        sr: int = None,
        duration: float = None,
        normalize: bool = True,
        processes: int = None,
        limit: int = None
) -> List[
    Tuple[
        str,
        Optional[np.ndarray],
        Optional[Dict[str, Any]]
    ]
]:
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
    :return: Dictionary of directories to list of (filename, np.array, metadata_dict)
    """

    #TODO multiprocessing

    # job = partial(load_sample())
    #
    # pool = Pool(processes)
    # processed_files = pool.map(
    #     job,
    #     ( file_list[:limit] )
    # )
    # logging.info(f'Processed {len(file_list)}')
    # return processed_files
    job = partial(load_sample, metadata_extractor=metadata_extractor, sr=sr, duration=duration, normalize=normalize)
    return [job(fn) for fn in file_list[:limit]]


def load_samples_in_dir_tree(
        root: str,
        metadata_extractor: Callable[[str], Dict[str, str]],
        file_types: List[str] = None,
        sr: int = None,
        duration: float = None,
        normalize: bool = True,
        processes: int = None,
        limit: int = None
) -> Dict[str,
          List[
              Tuple[
                  str,
                  Optional[np.ndarray],
                  Optional[Dict[str, Any]]
              ]
          ]
]:
    directory_file_list_dict = find_files(root, file_types=file_types)
    return {directory: load_samples(files, metadata_extractor, sr, duration, normalize, processes, limit) for
            directory, files in directory_file_list_dict.items()}


def write_numpy(path: str, samples: np.ndarray) -> None:
    """
    Write audio data to file.
    """
    np.save(path, samples)


def write_metadata(path: str, metadata: List[Dict[str, str]], fieldnames: List[str]) -> None:
    """
    Write audio metadata to file
    """
    with open(path, 'w+') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    logging.info(f"wrote metadata to file {path}")


def write_audio_data(
        root_dir: str,
        directory_to_loaded_samples_dict: Dict[
            str, List[Tuple[str, Optional[np.ndarray], Optional[Dict[str, Any]]]]],
        fieldnames: List[str] = None) -> None:
    for directory, files in directory_to_loaded_samples_dict.items():
        samples = []
        files_metadata = []
        for fn, audio, metadata in files:
            if fieldnames is None:
                fieldnames = metadata.keys()
            samples.append(audio)
            files_metadata.append(metadata)
        write_numpy(os.path.join(root_dir, directory + '_samples.npy'), np.asarray(samples))
        write_metadata(os.path.join(root_dir, directory + '_metadata.txt'), files_metadata, fieldnames)

        logging.info('Saved', len(samples), 'samples of ' + directory)


@click.command()
@click.option('-input_dir', default=None, type=click.Path(exists=True))
@click.option('-output_dir', default=None, type=click.Path())
@click.option('-dataset', default='medley-solos-db')
@click.option('-processes', default=None, type=click.INT)
@click.option('-file_limit', default=None, type=click.INT)
@click.option('-dataset_configs_directory', default=None, type=click.Path(exists=True))
def main(input_dir, output_dir, dataset, processes, file_limit, dataset_configs_directory):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    input_dir = os.path.join(PROJECT_DIR, 'data', 'raw') if input_dir is None else input_dir
    output_dir = os.path.join(PROJECT_DIR, 'data', 'processed') if output_dir is None else output_dir
    dataset_configs_directory = os.path.join(PROJECT_DIR,
                                             'dataset_configs') if dataset_configs_directory is None else dataset_configs_directory

    dataset_config = dataset_config_factory(dataset, dataset_configs_directory)
    logger.info(f"Loaded config for the {dataset} dataset")
    directory_to_loaded_samples_dict = load_samples_in_dir_tree(
        input_dir,
        metadata_extractor=dataset_config.get_extractor(),
        file_types=dataset_config.get_audio_file_types(),
        sr=dataset_config.get_samplerate(),
        duration=dataset_config.get_duration(),
        processes=processes,
        limit=file_limit
    )
    write_audio_data(os.path.join(PROJECT_DIR, 'data', 'interim'), directory_to_loaded_samples_dict)


if __name__ == '__main__':
    main()
