# -------------------------------------------
# Project root pointer
# -------------------------------------------
import os
from abc import ABCMeta
from pathlib import Path
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.absolute()
UNITDATA_DIR = os.path.join(ROOT_DIR, 'data', 'unittest_data')


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonABCMeta(ABCMeta, Singleton):
    pass
