from numba import *
import numba as nb

@autojit
def test_count(type, value):
    """
    >>> test_count(int_, [1,2,3,4,5,1,2])
    (0L, 1L, 2L)
    >>> test_count(float_, [1,2,3,4,5,1,2])
    (0L, 1L, 2L)
    """
    ttuple = nb.typedtuple(type, value)
    return ttuple.count(0), ttuple.count(3), ttuple.count(1)

@autojit
def test_count_complex(type, value):
    """
    >>> test_count_complex(complex128, [1+1j, 1+2j, 2+1j, 2+2j, 1+1j, 2+2j, 2+2j])
    (1L, 2L, 3L)
    """
    ttuple = nb.typedtuple(type, value)
    return ttuple.count(1+2j), ttuple.count(1+1j), ttuple.count(2+2j)
 
@autojit   
def test_index(type):
    """
    >>> test_index(int_)
    (0L, 2L, 4L)
    >>> test_index(float_)
    (0L, 2L, 4L)
    """
    ttuple = nb.typedtuple(type, [5,4,3,2,1])
    return ttuple.index(5), ttuple.index(3), ttuple.index(1)

@autojit
def test_index_error(type):
    """
    >>> test_index_error(int_)
    Traceback (most recent call last):
        ...
    ValueError: 10L is not in list

    >>> test_index_error(float_)
    Traceback (most recent call last):
        ...
    ValueError: 10.0 is not in list
    """
    ttuple = nb.typedtuple(type, [1,2,3,4,5])
    return ttuple.index(10)

def test(module):
    nb.testing.testmod(module)

if __name__ == "__main__":
    import __main__ as module
else:
    import test_typed_tuple as module

test(module)
__test__ = {}

