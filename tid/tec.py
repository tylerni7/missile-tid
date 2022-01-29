"""
Functions related to Total Electron Content calculations.
Basically all the weird combinations of signals used in
all the GNSS textbooks and ionospheric calculations
belong in here
"""
from __future__ import annotations  # defer type annotations due to circular stuff

from typing import cast, TYPE_CHECKING, Optional, Tuple
import numpy

from laika import constants

from tid import types


# deal with circular type definitions
if TYPE_CHECKING:
    from tid.connections import Connection

K = 40.308e16
M_TO_TEC = 6.158  # meters of L1 error to TEC
# set ionosphere puncture to 350km
IONOSPHERE_H = constants.EARTH_RADIUS + 350000
# maximum density of electrons for slant calculation
IONOSPHERE_MAX_D = constants.EARTH_RADIUS + 350000
C = constants.SPEED_OF_LIGHT


def melbourne_wubbena(
    frequencies: Optional[Tuple[float, float]],
    observations: types.Observations,
) -> Optional[numpy.ndarray]:
    """
    Calculate the Melbourne Wubbena signal combination for these observations.
    This relies on being able to get the frequencies (which can sometimes fail for
    GLONASS which uses FDMA)

    Args:
        frequencies: chan1 and chan2 frequencies
        observations: our dense data format

    Returns:
        numpy array of MW values or None if the calculation couldn't be completed
    """
    # calculate Melbourne Wubbena, this should be relatively constant
    # during a single connection

    # if we can't get this, we won't be able to do our other calculations anyway
    if frequencies is None:
        return None
    f1, f2 = frequencies

    phase = C / (f1 - f2) * (observations["L1C"] - observations["L2C"])
    pseudorange = 1 / (f1 + f2) * (f1 * observations["C1C"] + f2 * observations["C2C"])
    # wavelength = C/(f0 - f2)
    return phase - pseudorange


def calc_delay_factor(connection: Connection) -> float:
    """
    Calculate the delay factor: the strength of the ionospheric delay that the
    signal experiences, due to the frequency

    Args:
        connection: the connection of interest

    Returns:
        delay factor, in units of seconds^2
    """
    f1, f2 = connection.frequencies
    return ((f1**2) * (f2**2)) / ((f1**2) - (f2**2))


def calc_carrier_delays(connection: Connection, delay_factor: float) -> numpy.ndarray:
    """
    Calculate delay differences between L1C and L2C signals
    Normalized to meters

    Args:
        connection: the connection for which we want to calculate the carrier delays

    Returns:
        numpy array of the differences, in units of meters
    """

    sat_bias = connection.scenario.sat_biases.get(connection.prn, 0)
    station_bias_vector = connection.scenario.rcvr_biases.get(
        connection.station, (0, 0, 0)
    )
    f1, f2 = connection.frequencies

    # glonass station bias has a channel dependence
    if connection.is_glonass:
        station_bias = (
            station_bias_vector[1] + connection.glonass_chan * station_bias_vector[2]
        )
    else:
        station_bias = station_bias_vector[0]

    raw_phase_difference_meters = C * (
        connection.observations["L1C"] / f1 - connection.observations["L2C"] / f2
    )
    return (
        raw_phase_difference_meters
        + connection.carrier_correction_meters
        + (sat_bias - station_bias) * K / delay_factor
    )


def s_to_v_factor(elevations: numpy.ndarray, ionh: float = IONOSPHERE_H):
    """
    Calculate the unitless scaling factor to translate the slant ionospheric measurement
    to the vertical value.

    Args:
        elevations: a list of elevations (in units of radians)
        ionh: optional radius of the ionosphere in meters where the pierce point occurs

    Returns:
        numpy array of the unitless scaling factors
    """
    return numpy.sqrt(1 - (numpy.cos(elevations) * constants.EARTH_RADIUS / ionh) ** 2)


def calculate_vtecs(connection: Connection) -> numpy.ndarray:
    """
    For a given connection object, calculate the Vertical TEC values
    associated with each observation. That is: the total electron number
    the signal experienced through the atmosphere, normalized
    for path length due to the angle

    Why does this return the slant factors? because many of the
    calculations using vtec values still need it. Biases and corrections
    need the slant factor associated with the measurement as
    the slant factor also multiplies all the biases.

    Args:
        connection: the connection of interest

    Returns:
        numpy array of (
            TEC counts in TECu (1e16 electrons/m^2),
            unitless slant factors
        )
    """
    delay_factor = calc_delay_factor(connection)
    delays = calc_carrier_delays(connection, delay_factor)
    elevations = connection.elevation(
        cast(types.ECEF_XYZ, connection.observations["sat_pos"])
    )

    # total electron count integrated across the whole ionosphere
    slant_tec = delays * delay_factor / K
    # correction factor due to angle
    s_to_v_factors = s_to_v_factor(elevations)

    return numpy.array([slant_tec * s_to_v_factors, s_to_v_factors])


def ion_locs(
    rec_pos: types.ECEF_XYZ, sat_pos: types.ECEF_XYZ_LIST, ionh: float = IONOSPHERE_H
) -> types.ECEF_XYZ_LIST:
    """
    Given a receiver and a satellite, where does the line between them intersect
    with the ionosphere?

    Based on:
    http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm

    All positions are XYZ ECEF values in meters

    Args:
        rec_pos: the receiver position(s), a numpy array of shape (3,)
        sat_pos: the satellite position(s), a numpy array of shape (?,3)
        ionh: optional radius of the ionosphere in meters

    Returns:
        numpy array of positions of ionospheric pierce points
    """
    # Names are from quadratic formula, so a bit opaque
    # pylint: disable=invalid-name
    a = numpy.sum((sat_pos - rec_pos) ** 2, axis=1)
    b = 2 * numpy.sum((sat_pos - rec_pos) * rec_pos, axis=1)
    c = numpy.sum(rec_pos**2) - ionh**2

    common = numpy.sqrt(b**2 - (4 * a * c)) / (2 * a)
    b_scaled = -b / (2 * a)
    # TODO, I think the there is a clever way to vectorize the loop below
    # solutions = numpy.stack((b_scaled + common, b_scaled - common), axis=1)

    # for each solution, use the one with the smallest absolute value
    # (that is the closest intersection, the other is the further intersection)
    scale = numpy.zeros(sat_pos.shape)
    for i, (x, y) in enumerate(zip(b_scaled + common, b_scaled - common)):
        if abs(x) < abs(y):
            smallest = x
        else:
            smallest = y
        scale[i][0] = smallest
        scale[i][1] = smallest
        scale[i][2] = smallest

    return rec_pos + (sat_pos - rec_pos) * scale
