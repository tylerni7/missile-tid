"""
Common datatypes, so we can have functions be more clear about
return values than saying "numpy.array"
"""

from abc import ABC
from typing import List

import numpy


class ECEF_XYZ(numpy.ndarray):
    """
    numpy array of an ECEF XYZ coordinate in meters
    shape: (3,)
    """


# ECEF_XYZ.register(numpy.ndarray)


class ECEF_XYZ_LIST(numpy.ndarray):
    """
    numpy array of a list of ECEF XYZ coordinate in meters
    shape: (1,3)
    """


# ECEF_XYZ_LIST.register(numpy.ndarray)


class DenseDataType(numpy.ndarray):
    """
    numpy array of type dense_data.DENSE_TYPE
    """


# DenseDataType.register(numpy.ndarray)
