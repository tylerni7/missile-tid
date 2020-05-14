from distutils.core import setup, Extension

setup(
    name="gnss",
    version="0.1",
    author="Tyler Nighswander",
    install_requires=["laika", "numpy", "scipy","matplotlib"],
    packages=["gnss"],
    ext_package="gnss",
    ext_modules=[
        Extension('brute', ["gnss/minimize.c"], extra_compile_args=["-O3", "-march=native"])
    ],
    description="Good gnss data"
)

