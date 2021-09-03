from distutils.core import setup, Extension
import os

setup(
    name="tusa",
    version="0.1",
    author="Tyler Nighswander, Michael Nute",
    install_requires=["laika", "matplotlib", "numpy", "scipy"],
    packages=["tusa"]
    description="Ionospheric measurements from GPS",
)
