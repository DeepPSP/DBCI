"""
"""

from pathlib import Path

import setuptools

from diff_binom_confint import __version__

cwd = Path(__file__).resolve().parent

long_description = (cwd / "README.md").read_text(encoding="utf-8")

extras = {}
extras["test"] = [
    "pandas",
    "pre-commit",
    "tqdm",
    "pytest",
    "pytest-xdist",
    "pytest-cov",
]
extras["acc"] = ["numba"]
extras["dev"] = extras["test"] + extras["acc"]


setuptools.setup(
    name="diff_binom_confint",
    version=__version__,
    author="DeepPSP",
    author_email="wenh06@gmail.com",
    license="MIT",
    description="Computation of confidence intervals for binomial proportions and for difference of binomial proportions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DeepPSP/DBCI",
    # project_urls={},
    packages=setuptools.find_packages(
        exclude=[
            "docs*",
            "test*",
        ]
    ),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=(cwd / "requirements.txt").read_text().splitlines(),
    extras_require=extras,
)
