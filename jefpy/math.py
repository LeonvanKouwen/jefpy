"""
Set of math functions that operate explicitly on the last dimension. Assumes
the last dimension has length 3 (x, y, z). The _keepdim functions put a
singleton dimension at the end such that the number of dimensions doesn't
change. This is useful for broadcasting in vector operations.
"""

import numpy as np

def vec_split(x):
    x_norm = norm_keepdim(x)
    x_unit = x / x_norm
    return x_norm, x_unit


def norm(a):
    """ 
    Returns the norm array along the last dimension. 
    The last dimension should be length 3. 
    The returned norm has one dimension less that the input. 
    """
    return np.sqrt(a[..., 0]**2 + a[..., 1]**2 + a[..., 2]**2)


def norm_keepdim(a):
    """ 
    Returns the norm of an array along the last dimension.
    The returned norm has a singleton last dimension. 
    """
    return norm(a)[..., np.newaxis]


def norm_keepshape(a):
    """
    Returns the norm of an array along the last dimension.
    The last dimension of a should be length 3 and stays 3 (copies of the norm).
    """
    n = norm(a)
    return np.stack((n, n, n), axis=-1)


def cross(a, b):
    """ 
    Returns the cross product of two numpy arrays along 
    the last dimension. The last dimension should have length 3. 
    """

    c = np.empty_like(a + b)
    c[..., 0] = a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1]
    c[..., 1] = a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2]
    c[..., 2] = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    return c


def dot(a, b):
    """ 
    Returns the inner product of two arrays along the last dimension. 
    The last dimension should be length 3. 
    The returned norm has one dimension less that the input. 
    """
    return (a[..., 0] * b[..., 0] +
            a[..., 1] * b[..., 1] +
            a[..., 2] * b[..., 2])


def dot_keepdim(a, b):
    """ 
    Returns the inner product of two arrays along the last dimension. 
    The last dimension should be length 3. 
    The returned norm has a singleton last dimension. 
    """
    return dot(a, b)[..., np.newaxis]


class HarmonicOscillator:
    """
    Holds parameterized functions of a harmonic oscillator up to the second
    derivative (explicitly).
    """

    def __init__(self, amplitude=1.0, freq=1e6, phase=0.0):
        """
        :param amplitude: float,
        :param freq: float, frequency in Hz.
        :param phase: float, phase shift in radians.
        """
        self.ang_freq = freq * 2 * np.pi
        self.amp = amplitude
        self.phase = phase

    def f(self, t):
        """
        :param t: float, array, time.
        :return: float, array, harmonic oscillator function value.
        """
        return self.amp * np.sin(self.ang_freq * t + self.phase)

    def df_dt(self, t):
        """
        :param t: float, array, time.
        :return: float, array, harmonic oscillator first derivative.
        """
        return self.ang_freq * self.amp * np.cos(self.ang_freq * t
                                                 + self.phase)

    def d2f_dt2(self, t):
        """
        :param t: float, array, time.
        :return: float, array, harmonic oscillator second derivative.
        """
        return -self.ang_freq ** 2 * self.amp * np.sin(self.ang_freq * t
                                                       + self.phase)