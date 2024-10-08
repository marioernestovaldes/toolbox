# /usr/bin/env python
import io, os
import versioneer

from setuptools import setup
from os.path import dirname, realpath

with io.open("README.md", encoding="utf8") as readme:
    long_description = readme.read()

ROOT = dirname(realpath(__file__))

# -----------------------------------------------------------------------------
# Versioneer
# -----------------------------------------------------------------------------

versioneer.versionfile_source = "TB/_version.py"
versioneer.versionfile_build = "TB/_version.py"
versioneer.tag_prefix = "0.1"  # tags are like 1.1.0
versioneer.parentdir_prefix = "TB-"  # dirname like 'myproject-1.1.0'
_version = versioneer.get_version()


def package_tree(pkgroot):
    """Get list of packages by walking the directory structure and
    including all dirs that have an __init__.py or are named test.
    """
    subdirs = [
        os.path.relpath(i[0], ROOT).replace(os.path.sep, ".")
        for i in os.walk(os.path.join(ROOT, pkgroot))
        if "__init__.py" in i[2]
    ]
    return subdirs


REQUIRES = [
    "numpy==1.26.3",
    "rpy2",
    "statsmodels",
    "polars==0.20.31",
    "distinctipy",
    "tqdm",
    "seaborn",
    "missingno",
    "pyteomics",
    "lxml",
    "ipywidgets",
    "scikit-learn",
    "optuna",
    "matplotlib-venn",
    "openpyxl",
    "tabulate",
    "plotly",
    #"psycopg2-binary",
    "plotly",
    "black",
    "pytest",
    "plotnine",
    "scikit-misc",
    "adjustText",
    "jedi==0.17.2",
    "aiohttp",
    "requests",
    "beautifulsoup4",
    "xgboost",
    "shap",
    "nbformat",
    "SciencePlots"
]

config = {
    "description": "My useful Python tools.",
    "author": "Soren Wacker",
    "url": "https://github.com/sorenwacker/toolbox/",
    "download_url": "github.org",
    "author_email": "swacker@ucalgary.ca",
    "install_requires": REQUIRES,
    "packages": package_tree("TB"),
    "package_data": {},
    "name": "TB",
    "license": "",
    "version": _version,
}

setup(**config)
