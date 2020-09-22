import logging
import os
import re
from typing import Callable, Dict, List, Any

import yaml

from src.common.utils import PROJECT_DIR


class DatasetConfigParser:
    """
    Provides functionality for extracting dataset configs from yaml files. An example yaml
    file can be seen in /dataset_configs/example.yaml.
    """

    def __init__(self, dataset_config: Dict[str, Any]):
        self.dataset_config = dataset_config

    def get_extractor(self) -> Callable[[str], Dict[str, str]]:
        """Return a function for extracting metadata from file paths"""
        try:
            regex = self.dataset_config['metadata_regex']
        except KeyError as e:
            logging.error(
                "Didn't find a 'metadata_regex' entry in dataset configuration. Please specify a regex for "
                "extracting metadata from dataset filenames or empty string if not needed.")
            raise e
        if not regex:
            return lambda: {}

        pattern = re.compile(regex)

        def lb(s):
            fullmatch = pattern.fullmatch(s)
            if fullmatch:
                return fullmatch.groupdict()
            else:
                logging.error(f"Could not match metadata_regex to {s}")
                return {}

        return lb

    def get_classes(self) -> List[str]:
        """Return a list of classes where index in the list corresponds to integer mapping"""
        try:
            classes = self.dataset_config['classes']
        except KeyError as e:
            logging.error('Please specify a list of class names under the "classes" key in the dataset config file. '
                          'Class name index in the list should correspond to the ordinal encoding.')
            raise e
        return classes

    def get_class_color_mapping(self) -> Dict[str, str]:
        """Return a mapping from class to hex color code to use in visualization"""
        try:
            colors = self.dataset_config['colors']
        except KeyError as e:
            logging.error('Please specify a class-to-color mapping for the dataset under the "colors" key in the '
                          'dataset config file')
            raise e
        return colors

    def get_audio_file_types(self) -> List[str]:
        """Return the audio file types this dataset contains"""
        return self.dataset_config.get('file_types')

    def get_samplerate(self) -> int:
        """Return the audio samplerate for the files in the dataset"""
        return self.dataset_config.get('samplerate')

    def get_duration(self) -> float:
        """Return the max audio duration to load for the files in the dataset"""
        return self.dataset_config.get('duration')


def get_dataset_configs(dataset_configs_dir: str = 'dataset_configs'):
    if dataset_configs_dir is None:
        dataset_configs_dir = os.path.join(PROJECT_DIR, 'dataset_configs')
    return [f for f in os.listdir(dataset_configs_dir) if f.endswith('.yaml')]


def dataset_config_factory(dataset, dataset_configs_dir) -> DatasetConfigParser:
    with open(os.path.join(dataset_configs_dir, dataset + '.yaml')) as f:
        dataset_config = yaml.safe_load(f)
    return DatasetConfigParser(dataset_config)
