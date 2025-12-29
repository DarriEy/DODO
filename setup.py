#!/usr/bin/env python3
"""Setup for DODO package."""

from setuptools import setup, find_packages


setup(
    name="dodo-hydro",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
)
