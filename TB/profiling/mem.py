from collections import Counter
import linecache
import os
import tracemalloc
from datetime import datetime
from queue import Queue, Empty
from resource import getrusage, RUSAGE_SELF
from threading import Thread
from time import sleep

"""

"""


def memory_monitor(command_queue: Queue, poll_interval=1):
    """
    Monitor memory usage and display top memory-consuming lines.

    Parameters:
    - command_queue: A Queue used to control the monitor thread.
    - poll_interval: Polling interval for memory usage (default is 1 second).

    Example:

    You can use the memory_monitor function during the data processing phase. For instance, you are reading a
    large CSV file and performing various data transformations. You suspect that some operations might cause
    memory spikes. Here's how you can use the function:

    # Import the necessary modules
    from queue import Queue
    from your_memory_profile_module import memory_monitor

    # Create a command queue
    command_queue = Queue()

    # Start the memory monitor in a separate thread
    monitor_thread = Thread(target=memory_monitor, args=(command_queue,))
    monitor_thread.start()

    # Perform your data processing operations here
    process_large_data()

    # Stop the memory monitor after data processing
    command_queue.put("stop")
    monitor_thread.join()

    The memory_monitor function will periodically check and display memory usage. If it detects memory spikes during
    your data processing operations, it will display the top memory-consuming lines. This helps you pinpoint areas
    in your code that may need optimization.

    """
    tracemalloc.start()
    old_max = 0
    snapshot = None
    while True:
        try:
            command_queue.get(timeout=poll_interval)
            if snapshot is not None:
                print(datetime.now())
                display_top(snapshot)

            return
        except Empty:
            max_rss = getrusage(RUSAGE_SELF).ru_maxrss
            if max_rss > old_max:
                old_max = max_rss
                snapshot = tracemalloc.take_snapshot()
                print(datetime.now(), "max RSS", max_rss)


def display_top(snapshot, key_type="lineno", limit=3):
    """
    Display the top memory-consuming lines.

    Parameters:
    - snapshot: Memory snapshot to analyze.
    - key_type: The type used for statistics (default is "lineno").
    - limit: The number of top lines to display (default is 3).
    """
    snapshot = snapshot.filter_traces(
        (
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        )
    )
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # Replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print(
            "#%s: %s:%s: %.1f KiB" % (index, filename, frame.lineno, stat.size / 1024)
        )
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print("    %s" % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def memory_profile_function(func, *args, **kwargs):
    """
    Profile memory usage of a function.

    Parameters:
    - func: The function to profile.
    - *args: Positional arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.

    Example

    # Import the necessary modules
    from your_memory_profile_module import memory_profile_function

    # Define a complex data transformation function
    def complex_data_transformation(data):
        # Your complex data processing code here
        return processed_data

    # Profile the memory usage of the complex_data_transformation function
    memory_profile_function(complex_data_transformation, data)

    The function will profile the memory usage of complex_data_transformation and print the top memory-consuming lines
    within that function. This allows you to identify specific parts of the function that may need optimization.

    These functions are especially useful when working with large datasets or complex data processing tasks, where memory
    efficiency is critical to prevent crashes and slowdowns due to excessive memory consumption.

    """
    queue = Queue()
    poll_interval = 0.1
    monitor_thread = Thread(target=memory_monitor, args=(queue, poll_interval))
    monitor_thread.start()
    try:
        most_common = func(*args, **kwargs)
        print("Top prefixes:", most_common)
    finally:
        queue.put("stop")
        monitor_thread.join()
