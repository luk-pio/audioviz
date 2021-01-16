import json
import logging
import pickle
import sys
from abc import ABC
from functools import partial
from typing import List, Any, Dict


class FunCall:
    """
    Represents a function call which can be pickled
    """

    def __init__(self, name, func, args):
        self.name = name
        self.func = func
        self.args = args
        self._suffix = []

    def __call__(self, *args, **kwargs):
        return partial(self.func, **self.args)(*args, **kwargs)

    def add_suffix(self, suffix):
        self._suffix.append(suffix)

    def __repr__(self):
        suffix = "".join([s.__repr__() for s in self._suffix])
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"func={pickle.dumps(self.func, protocol=0).decode('ascii')}, "
            f"args={self.args})" + suffix
        )

    def __str__(self):
        args_str = " ".join([f"{k}={v}" for k, v in self.args.items()])
        suffix = (
            f" <- ({''.join([str(s) for s in self._suffix])})" if self._suffix else ""
        )
        return f"{self.name}({args_str})" + suffix


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
    def get_instance(cls, name: str, args: Dict[str, Any]):
        try:
            return FunCall(name, cls.implemented()[name], args)
        except KeyError as err:
            logging.exception("This feature has not been implemented.")
            raise NotImplementedError from err


def parse_funcall(funcall, factory):
    try:
        fc = factory.get_instance(
            **json.loads(funcall, encoding=str(sys.stdin.encoding))
        )
    except Exception:
        logging.exception(f"Could not decode function call {funcall}.")
        raise
    logging.info(f"Successfully Decoded function call {str(fc)}.")
    return fc


def parse_funcalls(funcalls, funcallFactory):
    parsed = []
    for funcall in funcalls:
        try:
            fe = parse_funcall(funcall, funcallFactory)
        except Exception:
            logging.exception(f"Skipping funcall {funcall} ...")
            continue
        parsed.append(fe)
    return parsed
