
# get estimate of total electron content for a given location for a time range

# TEC = (p_f1 - p_f2) / (k * (1 / f1**2 - 1 / f2**2))

import numpy
import math

from laika import constants, helpers
from laika.lib import coordinates

K = 40.308e16
m_to_TEC = 6.158  # meters of L1 error to TEC
# set ionosphere puncture to 350km
IONOSPHERE_H = constants.EARTH_RADIUS + 350000
# maximum density of electrons for slant calculation
IONOSPHERE_MAX_D = constants.EARTH_RADIUS + 350000
C = constants.SPEED_OF_LIGHT

# TODO: glonass if FDMA, so this doesn't work :/
#  and galileo is more complicated and needs l5a and l5b
#  so ignore this for anything but GPS...
F_lookup = {
    'R':(constants.GLONASS_L1, constants.GLONASS_L2, constants.GLONASS_L5),
    'G':(constants.GPS_L1, constants.GPS_L2, constants.GPS_L5),
    'E':(constants.l1, constants.l5, math.nan)
}

def correct_tec(tec_entry, rcvr_bias=0, sat_bias=0):
    # NOTE: the biases now have units of TECu, not meters !
    location, vtecish, s_to_v = tec_entry
    stec = vtecish / s_to_v  + (sat_bias - rcvr_bias)
    return location, stec * s_to_v, s_to_v

def calc_vtec(scenario, station, prn, tick, ionh=IONOSPHERE_H, el_cut=0.3, n1=0, n2=0, offset=0, rcvr_bias=0, sat_bias=0):
    """
    Given a receiver position and measurement from a specific sv this calculates the vertical TEC, and the point
    at which that TEC applies (between the sat and the rec). Checks to see whether the satellite is too low in
    the sky and returns None if so.

    Here the STEC estimate is pretty simple:

        STEC = [f_1**2 X f_2**2/(f_1**2 - f_2**2)] X [(Phi_1 - n1)/f_1 - (Phi_2 - n2)/f_2]*C
                |----  ^Delay Factor      -----|     |-----   Carrier-Phase Delay    ------|

    If we have receiver or satellite clock biases, those are included.
    Returns: ( VTEC (i.e. STEC * Slant_to_Vertical , Ionosphere-piercing loc , S-to-V factor )
    """
    measurement = scenario.station_data[station][prn][tick]
    if measurement is None:
        return None

    rec_pos = scenario.station_locs[station]

    n1 = n1 or 0
    n2 = n2 or 0
    offset = offset or 0

    # Calculate the slant TEC.
    res = calc_carrier_delay(
        scenario.dog,
        measurement,
        n1=n1, n2=n2, offset=offset,
        rcvr_bias=rcvr_bias,
        sat_bias=sat_bias,
    )
    if res is None:
        return None

    phase_diff, delay_factor = res
    stec = phase_diff * delay_factor / K

    if math.isnan(stec):
        return None

    if not measurement.processed:
        measurement.process(scenario.dog)
    el = scenario.station_el(station, measurement.sat_pos)
    # ignore things too low in the sky
    if el < el_cut or math.isnan(el):
        return None
    s_to_v = s_to_v_factor(el, ionh=ionh)
    return stec * s_to_v, ion_loc(rec_pos, measurement.sat_pos), s_to_v

def s_to_v_factor(el, ionh=IONOSPHERE_H):
    '''
    Converts between S and V. (Slant and Vertical?). See the Springer Handbook, equation (6.99) for this
    formula. Note that this is raised to the 1/2 power instead of -1/2 because we are going from STEC
    to VTEC but (6.99) goes the other way.
    :param el:  Elevation angle
    :param ionh: Height of the ionosphere in the zenith direction. Note that this shoould
                    already have Earth's Radius added in (as IONOSPHERE_H does above).
    :return:
    '''
    return math.sqrt(1 - (math.cos(el) * constants.EARTH_RADIUS / ionh) ** 2)

def ion_loc(rec_pos, sat_pos):
    """
    Given a receiver position and sat position, figure out
    where the ionosphere boundary between the two lines
    http://www.ambrsoft.com/TrigoCalc/Sphere/SpherLineIntersection_.htm
    """
    a = sum( (sat_pos - rec_pos)**2 )
    b = 2 * sum(( sat_pos - rec_pos) * rec_pos )
    c = sum(rec_pos**2) - IONOSPHERE_H**2

    t0 = (-b + math.sqrt(b ** 2 - (4 * a * c))) / (2 * a)
    t1 = (-b - math.sqrt(b ** 2 - (4 * a * c))) / (2 * a)

    res0 = numpy.array([(rec_pos[d] + (sat_pos[d] - rec_pos[d]) * t0) for d in range(3)])
    res1 = numpy.array([(rec_pos[d] + (sat_pos[d] - rec_pos[d]) * t1) for d in range(3)])

    # one of those intersections is the other side of the world...
    if (sum((rec_pos - res1)**2)) > (sum((rec_pos - res0)**2)):
        return res0
    else:
        return res1

def calc_carrier_delay(dog, measurement, n1=0, n2=0, offset=0, rcvr_bias=0, sat_bias=0):
    """
    calculates the carrier phase delay associated with a measurement
    """
    if measurement is None:
        return None
    observable = measurement.observables

    band_1 = 'L1C'
    if measurement.prn[0] == 'E':
        band_2 = 'L5C'
    else:
        band_2 = 'L2C'

    freqs = [dog.get_frequency(measurement.prn, measurement.recv_time, band) for band in [band_1, band_2]]

    if (
        math.isnan(observable.get(band_1, math.nan))
        or math.isnan(observable.get(band_2, math.nan))
    ):
        # missing info: can't do the calculation
        return None

    delay_factor = freqs[0]**2 * freqs[1]**2/(freqs[0]**2 - freqs[1]**2)

    phase_diff_meters = C * (
        (observable[band_1] - n1)/freqs[0] - (observable[band_2] - n2)/freqs[1]
    )
    return phase_diff_meters + offset + sat_bias - rcvr_bias, delay_factor


def melbourne_wubbena(dog, measurement):
    """
    Melbourne-Wubbena Combination:  (Carrier-Phase Widelane - Code-Phase Narrowlane)
                                    (\phi_WL - R_NL)
    Wide lane measurement, should not change "much" for GPS, except
    due to cycle slips.
    GLONASS is different though... and this doesn't work so well
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'

    freqs = [dog.get_frequency(measurement.prn, measurement.recv_time, band) for band in ['C1C', 'C2C']]
    phase = C/(freqs[0] - freqs[1])*(observable['L1C'] - observable['L2C'])
    pseudorange = 1/(freqs[0] + freqs[1])*(freqs[0]*observable['C1C'] + freqs[1]*observable[chan2])
    wavelength = C/(freqs[0] - freqs[1])
    return phase - pseudorange, wavelength

def wide_lane(measurement):
    """
    For Carrier-Phase (cycles): \phi_WL = (\phi_1 * f_1 - \phi_2 * f_2) / (f_1 - f_2)
           Code-Phase:
    Wide lane measurement: acts as measurement with wider wavelength
        used for ambiguity correction
    Note: This function returns carrier-phase Widelane denominated in meters, not cycles
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C/(freqs[0] - freqs[1])*(observable['L1C'] - observable['L2C'])
    pseudorange = 1/(freqs[0] - freqs[1])*(freqs[0]*observable['C1C'] - freqs[1]*observable[chan2])
    wavelength = C/(freqs[0] - freqs[1])
    return phase, pseudorange, wavelength

def narrow_lane(measurement):
    """
    Narrow lane measurement: acts as measurement with narrower wavelength
    used for ambiguity correction
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C/(freqs[0] + freqs[1])*(observable['L1C'] + observable['L2C'])
    pseudorange = 1/(freqs[0] + freqs[1])*(freqs[0]*observable['C1C'] + freqs[1]*observable[chan2])
    wavelength = C/(freqs[0] + freqs[1])
    return phase, pseudorange, wavelength

def ionosphere_free(measurement):
    """
    ionosphere free signal combination
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C/(freqs[0]**2 + freqs[1]**2)*(observable['L1C']*freqs[0] + observable['L2C']*freqs[1])
    pseudorange = 1/(freqs[0]**2 + freqs[1]**2)*(freqs[0]**2*observable['C1C'] + freqs[1]**2*observable[chan2])
    wavelength = C/(freqs[0] + freqs[1])  # same as narrow lane combination
    return phase, pseudorange, wavelength

def geometry_free(measurement):
    """
    geometry free (ionosphere) signal combination
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    phase = C*(observable['L1C']/freqs[0] - observable['L2C']/freqs[1])
    # yes, pseudorange is flipped intentionally
    pseudorange = observable[chan2] - observable['C1C']
    return phase, pseudorange

def ionosphere_combination(measurement, n1=0, n2=0):
    """
    combining the narrowland and widelane combinations to get
    a signal where the ionosphere is counted twice
    eq 8.28 in Petrovski and Tsujii Digital Satellite Navigation and Geophysics
    kind of like an anti-Melbourne Wubbena
    """

    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]

    N_w = n1 - n2
    N_n = n1 + n2

    lambda_w = C/(freqs[0] - freqs[1])
    phi_w = lambda_w * (observable['L1C'] - observable['L2C'])

    lambda_n = C/(freqs[0] + freqs[1])
    phi_n = lambda_n * (observable['L1C'] + observable['L2C'])

    # phi_w - phi_n = 2*I - lambda_n * N_n + lambda_w * N_w + (errors)
    ion = phi_w - phi_n + lambda_n * N_n - lambda_w * N_w

    return ion/2