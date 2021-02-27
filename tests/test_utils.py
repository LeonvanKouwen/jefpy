

import sys
sys.path.insert(0, '..')

import jefpy as jp
import numpy as np



def test_Jproperty():

    class Test():
        a = jp.JProperty()
        b = jp.JProperty(30)
        c = jp.JProperty()
        d = jp.JProperty()

        def __init__(self, c=30, **kwargs):
            self.c = c


    test1 = Test(c=123)
    test1.c(1231.123)
    assert hasattr(test1, 'c')

def test_is_iter():
    assert jp.is_iter(np.array(1))
    assert jp.is_iter((1, 2, 3))
    assert not jp.is_iter(1.0)


def test_mutkwarg():
    assert jp.default(None, 1) == 1
    assert jp.default(2, 1) == 2


# def test_make_callable_return_array():
#     x = (1, 2, 3)
#     assert jp.make_callable_return_array(x)('arg')[1] == 2
#
#     fun = jp.make_callable_return_array(lambda x: x*3)
#     assert fun(1) == 3


# def test_make_iterable():
#     assert jp.make_iterable(1.0).pop() == 1.0
#     assert jp.make_iterable([1.0])[0] == 1.0


def test_broadcast_spacetime():

    @jp.broadcast_spacetime
    def fun(self, r, t):
        return r * t + r + t
    _ = None
    assert fun(_, [1, 2, 3], 1).shape == (3,)
    assert fun(_, [1, 2, 3], [1, 2]).shape == (2, 3)
    assert fun(_, [[1, 2, 3], [4, 5, 6]], 1).shape == (2, 3)
    assert fun(_, [[1, 2, 3], [4, 5, 6]], [1, 3, 4, 5]).shape == (4, 2, 3)

def test_match_shape():
    to_match = np.zeros((10, 13, 11, 3))
    matched = jp.match_shape(np.ones(3), to_match)
    some_array = matched * to_match
    assert matched.shape == to_match.shape
    assert some_array.shape == to_match.shape

    to_match = np.zeros((3))
    matched = jp.match_shape(np.ones(3), to_match)
    some_array = matched * to_match
    assert matched.shape == to_match.shape
    assert some_array.shape == to_match.shape

    to_match = np.zeros((3))
    matched = jp.match_shape(1.0, to_match)
    some_array = matched * to_match
    assert matched.shape == to_match.shape
    assert some_array.shape == to_match.shape

    to_match = np.zeros((3))
    matched = jp.match_shape([1.0], to_match)
    some_array = matched * to_match
    assert matched.shape == to_match.shape
    assert some_array.shape == to_match.shape



# def test_loop_arg():
#
#     class Test:
#
#         @jp.loop_arg
#         def f(self, x):
#             return type(x)
#
#     assert all(Test().f((1, 2.0, 3)) == (int, float, int))