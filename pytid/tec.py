
# get estimate of total electron content for a given location for a time range

# TEC = (p_f1 - p_f2) / (k * (1 / f1**2 - 1 / f2**2))

import numpy
import math

from laika import constants, helpers

K = 40.308e16
m_to_TEC = 6.158
# set ionosphere puncture to 350km
IONOSPHERE_H = constants.EARTH_RADIUS + 350000
# maximum density of electrons for slant calculation
IONOSPHERE_MAX_D = constants.EARTH_RADIUS + 350000
C = constants.SPEED_OF_LIGHT

F_lookup = {
    'R':(constants.GLONASS_L1, constants.GLONASS_L2),
    'G':(constants.GPS_L1, constants.GPS_L2),
    'E':(constants.l1, constants.l5)
}

def calc_vtec(dog, rec_pos, measurement, ionh=IONOSPHERE_H, el_cut=0.40):
    """
    Given a receiver position and measurement from a specific sv
    this calculates the vertical TEC, and the point at which
    that TEC applies (between the sat and the rec)
    """
    if measurement is None:
        return None
    stec = calc_tec(measurement)
    if stec is None:
        return None
    if math.isnan(stec):
        return None
    measurement.process(dog)
    el, az = helpers.get_el_az(rec_pos, measurement.sat_pos)
    # ignore things too low in the sky
    if el < el_cut or math.isnan(el):
        return None
    s_to_v = s_to_v_factor(el, ionh=ionh)
    return stec * s_to_v, ion_loc(rec_pos, measurement.sat_pos), s_to_v

def s_to_v_factor(el, ionh=IONOSPHERE_H):
    # TODO : THIS ONLY WORKS IF TEC IS POSITIVE!!!!!
#    return el
#    return 1
    zenith = math.asin(constants.EARTH_RADIUS * math.sin(math.pi/2 + el) / IONOSPHERE_MAX_D)
    a = 0.9782  # constant magic number
#    return 1/(1 + 2*((0.10472+zenith)/1.5708)**3)
#    return 1/math.cos(zenith)**0.4
#    return 1/math.cos(math.asin(math.sin(zenith) * constants.EARTH_RADIUS/IONOSPHERE_MAX_D ))
    return math.sqrt(1 - (constants.EARTH_RADIUS * math.sin(a * zenith) / (IONOSPHERE_MAX_D)) ** 2)

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

def calc_tec_real(measurement):
    """
    Calculates the slant TEC for a set of observable
    or None if we are missing required observable
    """
    if measurement is None:
        return None
    observable = measurement.observables
    band = measurement.prn[0]  # GPS/GLONASS/GALILEO
    freqs = F_lookup[measurement.prn[0]]

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

    delay_factor = freqs[1]**2/(freqs[0]**2 - freqs[1]**2)

    phase_diff_meters = C * (
        observable[band_1]/freqs[0] - observable[band_2]/freqs[1]
    )
    return phase_diff_meters * m_to_TEC * delay_factor

def calc_tec(measurement):
    """
    Calculates the slant TEC for a set of observable
    or None if we are missing required observable
    """
    if measurement is None:
        return None
    observable = measurement.observables
    band = measurement.prn[0]  # GPS/GLONASS/GALILEO
    freqs = F_lookup[measurement.prn[0]]

    band_1 = 'C1C'
    if measurement.prn[0] == 'E':
        band_2 = 'L5C'
    else:
        band_2 = 'C2C'
    

    if (
        math.isnan(observable.get(band_1, math.nan))
        or math.isnan(observable.get(band_2, math.nan))
    ):
        # missing info: can't do the calculation
        return None

    delay_factor = freqs[1]**2/(freqs[0]**2 - freqs[1]**2)

    phase_diff_meters = C * (
        observable[band_1]/freqs[0] - observable[band_2]/freqs[1]
    )
    range_diff = m_to_TEC * delay_factor * (observable['C2C'] - observable['C1C'])
    return range_diff * m_to_TEC * delay_factor

def _calc_tec(measurement):
    """
    Calculates the slant TEC for a set of observable
    or None if we are missing required observable
    """
    observable = measurement.observables
    band = measurement.prn[0]  # GPS/GLONASS/GALILEO
    freqs = F_lookup[measurement.prn[0]]
    
    needed = ('L1C', 'L2C', 'C1C', 'C2C')
    for obs in needed:
        if math.isnan(observable.get(obs, math.nan)):
            # missing info: can't do the calculation
            return None
    
    delay_factor = freqs[1]**2/(freqs[0]**2 - freqs[1]**2)

    #range_diff = m_to_TEC * delay_factor * (observable['C2C'] - observable['C1C'])
    phase_diff = m_to_TEC * delay_factor * C * (observable['L2C']/freqs[1] - observable['L1C']/freqs[0])
    return phase_diff

def melbourne_wubbena(measurement):
    """
    Wide lane measurement, should not change "much" for GPS, except
    due to cycle slips.
    GLONASS is different though... and this doesn't work so well
    """
    if measurement is None:
        return math.nan
    observable = measurement.observables
    chan2 = 'C2P' if not math.isnan(observable['C2P']) else 'C2C'
    freqs = F_lookup[measurement.prn[0]]
    phase = C/(freqs[0] - freqs[1])*(observable['L1C'] - observable['L2C'])
    pseudorange = 1/(freqs[0] + freqs[1])*(freqs[0]*observable['C1C'] + freqs[1]*observable[chan2])
    return phase - pseudorange
