"""
Common datatypes, so we can have functions be more clear about
return values than saying "numpy.array"
"""
from typing import Any, Callable, Dict, TypeVar

import numpy

# placeholder type generic name
# pylint: disable=invalid-name
T = TypeVar("T")

# placeholder type for generic functions
FuncT = TypeVar("FuncT", bound=Callable[..., Any])


class StationPrnMap(Dict[str, Dict[str, T]]):
    """
    Map of Station to a map of PRNs to whatever

    This is a very common type so let's wrap it for convenience
    """


# pylint: disable=invalid-name
class ECEF_XYZ(numpy.ndarray):
    """
    numpy array of an ECEF XYZ coordinate in meters
    shape: (3,)
    """


# pylint: disable=invalid-name
class ECEF_XYZ_LIST(numpy.ndarray):
    """
    numpy array of a list of ECEF XYZ coordinate in meters
    shape: (n,3)
    """


class Observations(numpy.ndarray):
    """
    numpy array of type get_data.DENSE_TYPE
    """


class DenseMeasurements(Dict[str, Observations]):
    """
    Dictionary of PRN -> array of dense data
    """
