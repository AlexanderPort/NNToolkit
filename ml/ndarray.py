from functools import reduce
from math import exp, tanh
import random


def init_ndarray_class(cls=None):
    try:
        from .c_ndarray import ndarray
        return ndarray

    except ImportError:
        from .py_ndarray import ndarray
        return ndarray


ndarray = init_ndarray_class()