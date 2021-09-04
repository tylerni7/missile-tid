from distutils.core import setup, Extension
import os

setup(
    name="tid",
    version="0.1",
    author="Tyler Nighswander, Michael Nute",
    install_requires=[
        "laika @ git+https://github.com/commaai/laika#egg=laika",
        "matplotlib",
        "numpy",
        "scipy",
    ],
    packages=["tid"],
    description="Ionospheric measurements from GPS",
)
