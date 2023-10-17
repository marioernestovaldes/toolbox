from time import time, perf_counter
from contextlib import contextmanager
import numpy as np

@contextmanager
def timer(name):
    """
    A context manager to measure and print the time taken by a code block.

    Args:
        name (str): A descriptive name for the measured block of code.

    Usage:
    ```python
    with timer("My Code Block"):
        # Code to be timed
    ```

    This will print "[My Code Block] done in x.xxx s".
    """
    t0 = perf_counter()
    yield
    t1 = perf_counter()
    print("[{}] done in {:.3f} s".format(name, t1 - t0))

def intersects(group_a, group_b):
    """
    Find the intersection and unique elements between two lists or arrays.

    Args:
        group_a (list or numpy array): The first group of elements.
        group_b (list or numpy array): The second group of elements.

    Returns:
        dict: A dictionary containing three lists - 'intersect' for common elements,
              'only_a' for elements unique to group_a, and 'only_b' for elements unique to group_b.

    Example:
    ```python
    result = intersects([1, 2, 3], [2, 3, 4])
    print(result)
    ```
    Output: {'intersect': [2, 3], 'only_a': [1], 'only_b': [4]}
    """
    intersect = list(np.intersect1d(group_a, group_b))
    only_a = list(group_a)
    only_b = list(group_b)
    for el in intersect:
        only_a.remove(el)
        only_b.remove(el)
    output = dict(intersect=intersect, only_a=only_a, only_b=only_b)
    return output
