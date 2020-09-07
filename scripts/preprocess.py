import csv
import os
from collections import defaultdict
from multiprocessing import Pool
from typing import Tuple, List, Dict, Optional, Any

import librosa
import numpy as np


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
    for directory in os.walk(root):
        dirpath, dirnames, filenames = directory
        # get the dirname from path
        dirname = os.path.split(dirpath)[1]
        for filename in filenames:
            if any([filename.endswith(file_type) for file_type in file_types]):
                directory_file_list_dict[dirname].append(filename)
    return directory_file_list_dict


def load_sample(
        fn: str,
        sr: int = None,
        duration: float = None,
        normalize: bool = True
) -> Tuple[str, Optional[np.typing.ArrayLike], Optional[Dict[Any]]]:
    """
    Load a sample as a one-dimensional np.ArrayLike. If sample has more than one track, it will be converted to mono.
    Returns None as the np.array if file could not be read.

    :param fn: Path of the file.
    :param sr: Samplerate to load the sample at. If left as None, the file samplerate is used.
    :param duration: Max sample duration in seconds. If None, array will equal original file length.
    If sample is longer, it will be cropped. If shorter, the array will be padded with zeros.
    :param normalize: If true magnitude will be normalized to a value between 0 and 1.
    :return: a tuple containing (filename, np.array, metadata_dict)
    """

    if fn == '':  # ignore empty filenames
        return fn, None, None
    try:
        audio, sr = librosa.load(fn, sr, mono=True, duration=duration)
    except OSError as e:
        print(f'File {fn} could not be read: {e}')
        return fn, None, None
    file_length = len(audio)
    if file_length == 0:  # ignore zero-length samples
        return fn, None, None
    if duration:
        audio.resize(duration * sr)
    max_val = np.abs(audio).max()
    if max_val == 0:  # ignore completely silent sounds
        return fn, None, None
    if normalize:
        audio /= max_val
    return fn, audio, dict(duration=duration, samplerate=sr)


def load_samples(
        directory_file_list_dict: Dict[str, List[str]],
        sr=None,
        duration=None,
        normalize=True,
        processes: int = None,
        limit=None
) -> Dict[str,
          List[
              Tuple[str, Optional[np.typing.ArrayLike], Optional[Dict[Any]]]
          ]
]:
    """
    Load a (filename, np.array, duration) tuple for all files in directory_file_list_dict.
    
    :param directory_file_list_dict: Mapping between parent directories and lists of files
    :param sr: Samplerate to load the sample at. If left as None, the file samplerate is used.
    :param duration: Max sample duration in seconds. If None, array will equal original file length.
    If sample is longer, it will be cropped. If shorter, the array will be padded with zeros.
    :param normalize: If true magnitude will be normalized to a value between 0 and 1.
    :param processes: The number of worker processes to use. If None use os.cpu_count() cores.
    :param limit: The limit of files to load in each directory.
    :return: Dictionary of directories to list of (filename, np.array, metadata_dict)
    """
    directory_loaded_samples_dict = {}
    pool = Pool(processes)
    for instrument, files in directory_file_list_dict.items():
        directory_loaded_samples_dict[instrument] = pool.map(
            lambda fn: load_sample(fn, sr=sr, duration=duration, normalize=normalize),
            files[:limit]
        )
        print(f'Processed {len(directory_loaded_samples_dict[instrument])} samples for {instrument}')
    return directory_loaded_samples_dict


def write_numpy(path: str, samples: np.typing.ArrayLike) -> None:
    """
    Write audio data to file.
    """
    with open(path, 'w+') as f:
        np.save(f, samples)


def write_metadata(path: str, metadata: List[Dict[str, str]], fieldnames: List[str]) -> None:
    """
    Write audio metadata to file
    """
    with open(path, 'w+') as f:
        writer = csv.DictWriter(f, fieldnames)
        writer.writeheader()
        writer.writerows(metadata)


def write_audio_data(
        root_dir: str,
        directory_loaded_samples_dict: Dict[str, List[Tuple[str, Optional[np.typing.ArrayLike], Optional[Dict[Any]]]]],
        fieldnames: List[str] = None) -> None:
    if fieldnames is None:
        fieldnames = ['duration']
    for directory, files in directory_loaded_samples_dict.items():
        samples = []
        files_metadata = []
        for fn, audio, metadata in files:
            if not audio:
                continue
            samples.append(audio)
            files_metadata.append(metadata)
        write_numpy(os.path.join(root_dir, directory, '_samples.npy'), np.asarray(samples))
        write_metadata(os.path.join(root_dir, directory, '_metadata.txt'), files_metadata, fieldnames)

        print('Saved', len(samples), 'samples of ' + directory)


def main():
    from scripts.utils import DATA_DIR
    directory_file_list_dict = find_files(DATA_DIR)
    directory_loaded_samples_dict = load_samples(directory_file_list_dict, duration=2)
    write_audio_data(os.path.join(DATA_DIR, 'artifacts'), directory_loaded_samples_dict)


if __name__ == '__main__':
    main()
# pickle.dump(CLASSNAMES, open(data_root+"/CLASSNAMES.pickle", "w"))
# pickle.dump(lengths, open(data_root+"/lengths.pickle", "w"))

# Regex matching for extracting secondary and tertiary attributes from filenames

# From the drum class names, generate the regular expression used to match against sample file paths
# regex = r"\d{3}__\[(\w{3})\]\[(\w{3})\]\[(\w+)\]\d+__\d+.wav"

# filter filenames into sets by matching vs regex
# instrument_sets = {}
# for i in range(len(CLASSNAMES)):
#     instrument_sets[CLASSNAMES[i]] = {fileName for fileName in filenames if re.match(drumRegex[i], fileName)}
