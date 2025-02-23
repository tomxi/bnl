from setuptools import setup, find_packages

setup(
    name="bnl",
    version="0.1.0",
    author="tomxi",
    author_email="tom.xi@nyu.edu",
    description="A package for handling hierarchical segmentation annotations.",
    packages=find_packages(include=["bnl", "bnl.*"]),
    install_requires=[
        "librosa",
        "jams",
        "scikit-learn",
        "tqdm",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
