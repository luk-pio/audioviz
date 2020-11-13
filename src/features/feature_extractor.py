import json
import logging
import pickle
import sys
from functools import partial
from typing import List, Any

import numpy as np
from librosa import stft

# noinspection PyUnresolvedReferences
import src.common.log


class FeatureExtractor:
    def __init__(self, name, func, args):
        self.name = name
        self.func = func
        self.args = args

    def __call__(self, arr: np.ndarray):
        return partial(self.func, **self.args)(arr)

    def __repr__(self):
        return (
            f"FeatureExtractor(name={self.name}, "
            f"func={pickle.dumps(self.func, protocol=0).decode('ascii')}, "
            f"args={self.args})"
        )


class FeatureExtractorFactory:
    implemented_extractors = {"stft": stft}

    @classmethod
    def get_instance(cls, name: str, args: List[Any]):
        try:
            return FeatureExtractor(name, cls.implemented_extractors[name], args)
        except KeyError as err:
            logging.exception("This feature has not been implemented.")
            raise NotImplementedError from err


def parse_feature_extractors(features):
    feature_extractors = []
    for feature in features:
        logging.info(f"Attempting to decode feature {feature}")
        try:
            fe = FeatureExtractorFactory.get_instance(
                **json.loads(feature, encoding=str(sys.stdin.encoding))
            )
        except Exception:
            logging.exception(f"Could not decode feature {feature}. Skipping...")
            continue
        logging.info(f"Successfully Decoded feature {feature}.")
        feature_extractors.append(fe)
    return feature_extractors
