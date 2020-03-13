import typing as t
from itertools import zip_longest
import random


def is_one_of_types(val: t.Any, dtypes: t.Sequence[t.Any]) -> bool:
    return any(isinstance(val, dtype) for dtype in dtypes)


def is_two_tuple(val) -> bool:
    return is_one_of_types(val, [list, tuple]) and len(val) == 2


def grouper(iterable: t.Iterable, n: int, fillvalue=None):
    """
    Let's you iterate over `iterable` `n` elements at a time, padding
    the end with `fillvalue` if the number of elements in `iterable` is
    not evenly divisible by `n`. Source:
    https://stackoverflow.com/questions/434287/what-is-the-most-pythonic-way-to-iterate-over-a-list-in-chunks
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def shuffle(iterable: t.Sequence) -> t.Sequence:
    """
    Randomly shuffles `iterable`. Does not perform the shuffle in-place
    like `random.shuffle` but returns a shuffled copy.
    """
    return random.sample(iterable, len(iterable))


def inverted(f: t.Callable) -> t.Callable:
    """
    Creates an inverted version of `f`. Useful if your objective
    is to minimize `f` but you want to use `ga`, since `ga` tries
    to maximize the objective.
    """

    def inverted_f(*args, **kwargs) -> float:
        return -f(*args, **kwargs)

    return inverted_f
