import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def install_requires():
    required = None
    current_dir = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(os.path.join(current_dir, "requirements.txt"), encoding="utf-8") as f:
            required = f.read().split("\n")
    except FileNotFoundError:
        required = []
    return required


setup(
    name="cassava_classifier",
    version="0.0.1",
    author="Prasanna Kumar, PS Vishnu",
    author_email="vpkpypi@gmail.com, psvpypi@gmail.com",
    description=("Casssava leaf disease classification using Deep neural network in Pytorch"),
    long_description_content_type="text/markdown",
    long_description=read("README.md"),
    license="MIT",
    keywords=["image classification", "leaf disease classifier", "pytorch"],
    url="https://github.com/p-s-vishnu/cassava-leaf-disease-classification",
    packages=find_packages(exclude=["tests", "docs", "images"]),
    install_requires=install_requires(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
