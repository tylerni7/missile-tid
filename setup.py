from distutils.core import setup, Extension
import os

setup(
    name="gnss",
    version="0.1",
    author="Tyler Nighswander",
    install_requires=["laika", "matplotlib", "numpy", "scipy"],
    packages=[os.path.join("pytid", "gnss")],
    ext_package=os.path.join("pytid", "gnss"),
    ext_modules=[
        Extension('brute', [os.path.join("pytid", "gnss", "minimize.c")], extra_compile_args=["-O3", "-march=native"])
    ],
    description="Good gnss data"
)
