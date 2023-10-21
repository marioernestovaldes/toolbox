# Import necessary libraries and modules
import os
import re
import shutil
import logging

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Importing and configuring libraries for data visualization
import matplotlib
from matplotlib import pylab, mlab
from matplotlib import pyplot as plt
import matplotlib as mpl

# Importing utilities for data visualization
from IPython.core.pylabtools import figsize, getfigs

# Importing date and file-related modules
from datetime import date
from os.path import isdir, isfile, basename, dirname, join
from time import sleep
from glob import glob
from pathlib import Path as P
from tqdm.notebook import tqdm

from .general import time_format

# Importing more numerical and visualization modules
from pylab import *
from numpy import *

import seaborn as sns
import plotly.express as px

# Import custom plotting functions
from .plotting import savefig as sf
from .plotting import (
    plot_roc,
    plot_random_roc,
    plot_diagonal,
    plot_hlines,
    plot_vlines,
    heatmap,
    legend_outside,
    plot_dendrogram,
)

# Configure Matplotlib settings for figures
plt.rcParams["figure.facecolor"] = "w"
# plt.rcParams["figure.dpi"] = 100

# Configure Pandas options for displaying dataframes
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

# Set Seaborn plotting context and style
# sns.set_context("paper")

from .color_and_style import set_sns_style, lighten_color
set_sns_style(style='ticks')

# Import custom functions for LSARP data
from .lsarp import get_plate

# Define a function to get the current date in a specific format
def today():
    return date.today().strftime("%y%m%d")

# Define utility functions for logarithmic transformations
def log2p1(x):
    try:
        return np.log2(x + 1)
    except:
        return x

def log10p1(x):
    try:
        return np.log10(x + 1)
    except:
        return x

# Define a function to remove digits from a string
remove_digits = lambda x: "".join([i for i in x if not i.isdigit()])

# STOP

# with open(__file__, "r") as this_file:
#     """
#     Read the content of the current Python script (the module it's placed in) and prints the lines until it
#     encounters a line that contains the string "STOP". It also prints the current working directory and the
#     current date.
#     """
#     for line in this_file.readlines():
#         if re.search("STOP", line):
#             break
#         print(line, end="")
#     print(f"# Current working directory: {os.getcwd()}")
#     print(f"# Current date: {today()}")
