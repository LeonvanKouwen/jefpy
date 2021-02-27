
""" Collection of convenience functions."""

from functools import wraps
from typing import Iterable
import numpy as np


class JProperty:
    # TODO add docstrings

    def __init__(self, default=None):
        self.default = default

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, type=None):
        return obj.__dict__[self.name]

    def __set__(self, obj, property):
        if callable(property):
            property_new = property
        elif property is None:
            def property_new(t=None):
                return self.default
        else:
            # Assume it is a number or a collection.
            def property_new(t=None):
                return np.array(property, dtype=np.float)
        obj.__dict__[self.name] = property_new


def broadcast_spacetime(method):
    """"
    Decorator to allow polymorphic behaviour of spacetime (r, t) coordinates.
    r can be any shape array with the last dimension of length 3:
    (..., 3). Time can be a float or a one dimensional array. If r or t is
    not a numpy ndarray, it is being converted to a numpy array.
    This broadcasting decorator makes vectorized computation of the fields
    possible, so it beneficial for convenience and performance.

    If t is an array, a new first dimension is added to r, matching the length
    of t. The t array gets singleton dimensions added to match the number of
    dimensions of r. The result is that operations involving r and t are
    safely broadcasted.
    """
    @wraps(method)
    def broadcasted(self, r, t=.0, **k):
        r = np.array(r)
        # t_is_1D = np.array(t).ndim == 1
        # t_is_singleton = np.array(t).size == 1
        if np.array(t).ndim == 1:
            # If t is an array (not singleton), we need to broadcast
            # but if t is higher dimensional, it is already broadcasted.
            new_shape = [len(t)] + [1] * r.ndim
            t = np.reshape(np.array(t), new_shape)
            r = r[np.newaxis, ...]
            r = np.repeat(r, len(t), axis=0)
        return method(self, r, t, **k)
    return broadcasted


def match_shape(x, to_match):
    """
    Gives x the shape of to_match by creating equal dimensions and equal size of
    the dimensions.
    :param x:
    :param to_match:
    :return:
    """
    return x + 0 * to_match
    return x * np.ones_like(to_match) #TODO, check if this is equal


def is_iter(arg):
    # TODO DEPRECATED?
    """ Shorthand function to check if arg is an iterable."""
    return isinstance(arg, Iterable)


def default(arg, default):
    """
    Returns default if arg is None, else returns arg.
    Solution to assign defaults to mutable key-word arguments
    """
    return default if arg is None else arg


def check_spatial_vec(vector):
    # TODO DEPRECATED?
    """ Raises value error if vector does not have the last dim of length 3."""
    vector = np.array(vector)
    if vector.shape[-1] != 3:
        raise ValueError("Spatial array requires last dim of length 3. ")
    return vector







