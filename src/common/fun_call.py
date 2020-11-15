import json
import logging
import pickle
import sys
from abc import ABC
from functools import partial
from typing import List, Any


class FunCall:
    """
    Represents a function call which can be pickled
    """

    def __init__(self, name, func, args):
        self.name = name
        self.func = func
        self.args = args

    def __call__(self, *args, **kwargs):
        return partial(self.func, **self.args)(*args, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"func={pickle.dumps(self.func, protocol=0).decode('ascii')}, "
            f"args={self.args})"
        )


class FunCallFactory(ABC):
    @classmethod
    def implemented(cls):
        try:
            return cls._implemented.copy()
        except AttributeError:
            raise NotImplementedError(
                "Please define a dictionary of implemented functions for the factory under "
                "the __implemented class attribute "
            )

    @classmethod
    def get_instance(cls, name: str, args: List[Any]):
        try:
            return FunCall(name, cls.implemented()[name], args)
        except KeyError as err:
            logging.exception("This feature has not been implemented.")
            raise NotImplementedError from err


def parse_funcall(funcall, factory):
    logging.info(f"Attempting to decode function call {funcall}")
    try:
        fc = factory.get_instance(
            **json.loads(funcall, encoding=str(sys.stdin.encoding))
        )
    except Exception:
        logging.exception(f"Could not decode function call {funcall}.")
        raise
    logging.info(f"Successfully Decoded function call {funcall}.")
    return fc
