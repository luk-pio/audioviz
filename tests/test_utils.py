import unittest

from src.common.utils import stringify_funcall, unpickle_funcall
from tests.common import dummy_func


class TestUtils(unittest.TestCase):
    def test_unpickle_funcall(self):
        args = (1, 2, 3)
        kwargs = dict(d="d", e="e")
        pickled = stringify_funcall(dummy_func, *args, **kwargs)
        unpickled = unpickle_funcall(pickled)
        self.assertEqual(unpickled[0], dummy_func)
        self.assertEqual(unpickled[1], args)
        self.assertEqual(unpickled[2], kwargs)
