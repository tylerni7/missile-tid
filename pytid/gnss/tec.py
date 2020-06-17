
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
    location, vtecish, s_to_v = tec_entry
    stec = vtecish / s_to_v  + 9.517753907876292 * (sat_bias - rcvr_bias)
    return location, stec * s_to_v, s_to_v

def calc_vtec(dog, rec_pos, measurement, ionh=IONOSPHERE_H, el_cut=0.30, n1=0, n2=0, rcvr_bias=0, sat_bias=0):
    """
    Given a receiver position and measurement from a specific sv
    this calculates the vertical TEC, and the point at which
    that TEC applies (between the sat and the rec)
    """
    if measurement is None:
        return None

    # the velocity of the satellite in the direction of the receiver
    # positive is approaching, negative is receding
    # this can probably safely be ignored for just about everything...
    doppler = (
        numpy.linalg.norm(rec_pos - measurement.sat_pos_final + measurement.sat_vel)
        - numpy.linalg.norm(rec_pos - measurement.sat_pos_final)
    )

    stec = calc_tec(
        measurement,
        n1=n1, n2=n2,
        rcvr_bias=rcvr_bias,
        sat_bias=sat_bias,
        doppler=doppler,
    )
    if stec is None:
        return None
    if math.isnan(stec):
        return None

    if not measurement.processed:
        measurement.process(dog)
    el, az = helpers.get_el_az(rec_pos, measurement.sat_pos)
    # ignore things too low in the sky
    if el < el_cut or math.isnan(el):
        return None
    s_to_v = s_to_v_factor(el, ionh=ionh)
    return stec * s_to_v, ion_loc(rec_pos, measurement.sat_pos), s_to_v

def s_to_v_factor(el, ionh=IONOSPHERE_H):
    return math.sqrt(1 - (math.cos(el) * constants.EARTH_RADIUS / ionh) ** 2)


def ion_loc2(rec_pos, sat_pos, ionh=IONOSPHERE_H, el=None):
    """
    Get ionospheric pierce point for receiver - satellite connection
    Taken from laika.iono.get_delay
    """
    rec = coordinates.LocalCoord.from_ecef(rec_pos)
    alt = numpy.linalg.norm(rec_pos)
    if el is None:
        el, _ = helers.get_el_az(rec_pos, sat_pos)
    alpha = numpy.pi/2 + el
    beta = numpy.arcsin(alt * numpy.sin(alpha) / (ionh + constants.EARTH_RADIUS))
    gamma = numpy.pi - alpha - beta
    ipp_dist = alt * numpy.sin(gamma) / numpy.sin(beta)
    ipp_ned = rec.ecef2ned(sat_pos) * ipp_dist / numpy.linalg.norm(sat_pos)
    return rec.ned2ecef(ipp_ned)

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

def calc_carrier_delay(measurement, n1=0, n2=0, rcvr_bias=0, sat_bias=0, doppler=0):
    """
    calculates the carrier phase delay associated with a measurement
    """
    if measurement is None:
        return None
    observable = measurement.observables
    band = measurement.prn[0]  # GPS/GLONASS/GALILEO
    freqs = F_lookup[measurement.prn[0]]

    if doppler:
        # doppler adjust the frequencies
        gamma = math.sqrt((C - doppler)/(C + doppler))
        freqs = [f * gamma for f in freqs]

    band_1 = 'L1C'
    if measurement.prn[0] == 'E':
        band_2 = 'L5C'
    else:
        band_2 = 'L2C'


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
    return phase_diff_meters + sat_bias - rcvr_bias, delay_factor


def calc_tec(measurement, n1=0, n2=0, rcvr_bias=0, sat_bias=0, doppler=0):
    """
    Calculates the slant TEC for a set of observable
    or None if we are missing required observable
    """
    if measurement is None:
        return None

    res = calc_carrier_delay(
        measurement,
        n1=n1, n2=n2,
        rcvr_bias=rcvr_bias,
        sat_bias=sat_bias,
        doppler=doppler,
    )
    if res is None:
        return None

    phase_diff, delay_factor = res
    return phase_diff * delay_factor / K

def melbourne_wubbena(measurement):
    """
    Wide lane measurement, should not change "much" for GPS, except
    due to cycle slips.
    GLONASS is different though... and this doesn't work so well
    """
    if measurement is None:
        return None
    observable = measurement.observables
    chan2 = 'C2C' if not math.isnan(observable.get('C2C', math.nan)) else 'C2P'
    freqs = F_lookup[measurement.prn[0]]
    phase = C/(freqs[0] - freqs[1])*(observable['L1C'] - observable['L2C'])
    pseudorange = 1/(freqs[0] + freqs[1])*(freqs[0]*observable['C1C'] + freqs[1]*observable[chan2])
    wavelength = C/(freqs[0] - freqs[1])
    return phase - pseudorange, wavelength

def wide_lane(measurement):
    """
    Wide lane measurement: acts as measurement with wider wavelength
    used for ambiguity correction
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