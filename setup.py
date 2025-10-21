#!/usr/bin/env python
import codecs
import os
import re

from setuptools import find_packages, setup

###############################################################################

NAME = "fer"
PACKAGES = find_packages(where="src")
HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_PATH = os.path.join("src", NAME, "__init__.py")
META_FILE = read(META_PATH)
KEYWORDS = ["facial expressions", "emotion detection", "faces", "images"]


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        rf"^__{meta}__ = ['\"]([^'\"]*)['\"]", META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError(f"Unable to find __{meta}__ string.")


URL = find_meta("url")
PROJECT_URLS = {
    "Documentation": URL,
    "Bug Tracker": "https://github.com/justinshenk/fer/issues",
    "Source Code": "https://github.com/justinshenk/fer",
}
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Topic :: Software Development :: Libraries",
]

PYTHON_REQUIRES = ">= 3.8"

INSTALL_REQUIRES = [
    "matplotlib",
    "opencv-contrib-python",
    "keras>=2.0.0",
    "pandas",
    "requests",
    "facenet-pytorch",
    "tqdm>=4.62.1",
    "moviepy",
    "ffmpeg-python>=0.2.0",
    "Pillow",
]

EXTRAS_REQUIRE = {"docs": ["sphinx"], "tests": ["coverage", "pytest"]}
EXTRAS_REQUIRE["dev"] = (
    EXTRAS_REQUIRE["tests"] + EXTRAS_REQUIRE["docs"] + ["wheel", "pre-commit"]
)

VERSION = find_meta("version")

# README.md
LONG = open("README.md").read()

setup(
    name=NAME,
    version=find_meta("version"),
    author=find_meta("author"),
    author_email=find_meta("email"),
    maintainer=find_meta("author"),
    maintainer_email=find_meta("email"),
    description=find_meta("description"),
    license=find_meta("license"),
    keywords=KEYWORDS,
    url=URL,
    project_urls=PROJECT_URLS,
    packages=PACKAGES,
    long_description=LONG,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    python_requires=PYTHON_REQUIRES,
    include_package_data=True,
    package_dir={"": "src"},
    zip_safe=False,
)
