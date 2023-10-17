# Import necessary libraries and modules
import os
import re
import shutil
import logging

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
plt.rcParams["figure.dpi"] = 150

# Configure Pandas options for displaying dataframes
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

# Set Seaborn plotting context and style
sns.set_context("paper")
sns.set_style("white")

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

# Print information about the current working directory and date
with open(__file__, "r") as this_file:
    for line in this_file.readlines():
        if re.search("STOP", line):
            break
        print(line, end="")
    print(f"# Current working directory: {os.getcwd()}")
    print(f"# Current date: {today()}")
