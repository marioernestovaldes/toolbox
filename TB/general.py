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


def time_format(seconds: int) -> str:
    """
    Format a duration in seconds into a human-readable string representation.

    Args:
        seconds (int): The duration in seconds.

    Returns:
        str: A formatted string representing the duration in days, hours, minutes, and seconds.
    """
    if seconds is not None:
        # Convert seconds to an integer to ensure it's a whole number.
        seconds = int(seconds)

        # Calculate the number of days, hours, minutes, and remaining seconds.
        d = seconds // (3600 * 24)  # Calculate days
        h = seconds // 3600 % 24  # Calculate hours
        m = seconds % 3600 // 60  # Calculate minutes
        s = seconds % 3600 % 60  # Calculate remaining seconds

        # Check if there are days, hours, minutes, or seconds and format accordingly.
        if d > 0:
            return '{:02d}D {:02d}H {:02d}m {:02d}s'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}H {:02d}m {:02d}s'.format(h, m, s)
        elif m > 0:
            return '{:02d}m {:02d}s'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)

    # Return a '-' if the input is None or if the time is 0 seconds.
    return '-'


def intersects(group_a, group_b):
    """
    Find the intersection and unique elements between two lists or arrays.

    Args:
        group_a (list or numpy array): The first group of elements.
        group_b (list or numpy array): The second group of elements.

    Returns:
        dict: A dictionary containing three sets - 'intersect' for common elements,
              'only_a' for elements unique to group_a, and 'only_b' for elements unique to group_b.

    Example:
    ```python
    result = intersects([1, 2, 3], [2, 3, 4])
    print(result)
    ```
    Output: {'intersect': {2, 3}, 'only_a': {1}, 'only_b': {4}}
    """
    set_a = set(group_a)
    set_b = set(group_b)

    intersect = set_a & set_b
    only_a = set_a - intersect
    only_b = set_b - intersect

    return {'intersect': list(intersect), 'only_a': list(only_a), 'only_b': list(only_b)}
