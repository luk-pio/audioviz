import unittest
import numpy as np
import pandas as pd

from src.features.build_features import FeatureSet
from tests.common import dummy_func


def generate_AudiovizDataset(
    rows=100, cols=1000, classes=5, metadata=None, name="test_dataset"
):
    metadata = {} if metadata is None else metadata
    data = np.random.random((rows, cols))

    if "class" not in metadata:
        metadata["class"] = np.random.randint(0, classes, rows)

    metadata = pd.DataFrame(metadata)

    from src.common.AudiovizDataset import AudiovizDataset

    return AudiovizDataset(data, metadata, name)


class TestFeatureSet(unittest.TestCase):
    def setUp(self) -> None:

        self._name1 = "testname1"
        self.fe_args1 = (1, 2)
        self.fe_kwargs1 = {"d": "", "e": ""}
        self._feature_extractor1 = dummy_func
        self._feature_directory = ""
        self._dataset1 = generate_AudiovizDataset()
        self._fs1_args = (
            self._name1,
            self._dataset1,
            self._feature_extractor1,
            self.fe_args1,
            self.fe_kwargs1,
        )

    def test_init(self):
        fs = FeatureSet(*self._fs1_args, features_directory=self._feature_directory)
        print(fs.__repr__())
        self.assertEqual(fs._name, self._name1)
        self.assertEqual(fs._dataset, self._dataset1)
        self.assertEqual(fs._feature_extractor, self._feature_extractor1)
        self.assertEqual(fs.fe_args, self.fe_args1)
        self.assertEqual(fs._features_directory, self._feature_directory)
        self.assertEqual(fs.fe_kwargs, self.fe_kwargs1)
        import pickle

        filename = (
            str(
                pickle.dumps(
                    (
                        self._name1,
                        self._feature_extractor1,
                        self.fe_args1,
                        self.fe_kwargs1,
                    )
                )
            )
            + FeatureSet.FEATURE_FILE_SUFFIX
        )
        self.assertEqual(fs._filename, filename)

    # def test_init_standard(self):
    #     fs = FeatureSet(*self._fs1_args, features_directory=self._feature_directory)
    #     self.assertEqual(fs._name, self._name1)
    #     self.assertEqual(fs._dataset, self._dataset1)
    #     self.assertEqual(fs._feature_extractor, self._feature_extractor1)
    #     self.assertEqual(fs.fe_args, self.fe_args1)
    #     self.assertEqual(fs._features_directory, self._feature_directory)
    #     self.assertEqual(fs.fe_kwargs, self.fe_kwargs1)
    #     import pickle
    #
    #     filename = (
    #         str(pickle.dumps((self._name1, self._feature_extractor1, self.fe_args1, self.fe_kwargs1)))
    #         + FeatureSet.FEATURE_FILE_SUFFIX
    #     )
    #     self.assertEqual(fs._filename, filename)
    #


if __name__ == "__main__":
    unittest.main()
