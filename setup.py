from setuptools import find_packages, setup


setup(
    name="portafolios-inversion",
    version="0.1.0",
    description="Utilities for portfolio research and asset allocation.",
    packages=find_packages(include=["src", "src.*"]),
)
