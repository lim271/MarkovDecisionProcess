#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

requirements = [
    "deprecated",
    "numpy",
    "scipy",
    "setuptools>=41.0.0",  # to satisfy dependency constraints
]


def read_file(filename: str) -> str:
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()


version = read_file("VERSION")
readme_text = read_file("README.md")

packages = find_packages(".", exclude=["example.py"])

setup(
    name="MarkovDecisionProcess",
    version=version,
    author="Jaehyun Lim",
    author_email="jaehyunlim@yonsei.ac.kr",
    description="Markov decision process",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    license=None,
    keywords="stochastic-optimal-control",
    url="https://github.com/lim271/MarkovDecisionProcess",
    project_urls={
        "Source on GitHub": "https://github.com/lim271/MarkovDecisionProcess",
        "Documentation": "https://github.com/lim271/MarkovDecisionProcess/README.md",
    },
    packages=packages,
    include_package_data=False,
    install_requires=requirements,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Free For Educational Use",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Typing :: Typed",
    ],
)