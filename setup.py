import setuptools
import sys

if sys.version_info.major != 3:
    raise TypeError(
        'This Python is only compatible with Python 3, but you are running '
        'Python {}. The installation will likely fail.'.format(
            sys.version_info.major))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="safe_rl",  # this is the name displayed in 'pip list'
    version="0.1",
    author="",
    author_email="",
    description="Safe RL",
    install_requires=['numpy'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)